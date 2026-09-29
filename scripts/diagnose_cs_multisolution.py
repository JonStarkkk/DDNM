"""Evaluation only: call the unchanged official DDNM sampler and CS operator."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from functions.svd_ddnm import ddnm_diffusion
from functions.svd_operators import CS
from guided_diffusion.script_util import create_model


def namespace(value):
    return SimpleNamespace(**{k: namespace(v) if isinstance(v, dict) else v
                              for k, v in value.items()})


class DiagnosticOperator:
    """Observe clean predictions without changing the official DC calculation."""
    def __init__(self, operator, y, mode):
        self.operator, self.y, self.mode = operator, y, mode
        self.records = []

    def A(self, x):
        self.prediction = x
        return self.operator.A(x)

    def A_pinv(self, residual):
        correction = self.operator.A_pinv(residual)
        if self.mode == "none":
            correction = torch.zeros_like(correction)
        corrected = self.prediction - correction
        after = self.operator.A(corrected) - self.y
        self.records.append({"pre_dc_rms": residual.square().mean().sqrt().item(),
                             "post_dc_rms": after.square().mean().sqrt().item()})
        return correction


def measurement_metrics(operator, x, y):
    residual = operator.A(x.reshape(1, -1)) - y
    return {"l2": residual.norm().item(),
            "relative": (residual.norm() / y.norm().clamp_min(1e-12)).item(),
            "rms": residual.square().mean().sqrt().item()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--ratio", type=float, default=0.1)
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--eta", type=float, default=0.85)
    parser.add_argument("--phi_seed", type=int, default=1234)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--dc", choices=["ddnm", "none"], default="ddnm")
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config = yaml.safe_load((ROOT / "configs/imagenet_256.yml").read_text())
    config["time_travel"]["T_sampling"] = args.steps
    model = create_model(**config["model"])
    # Match upstream conversion-before-load and strict checkpoint loading.
    model.convert_to_fp16()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True), strict=True)
    model.to(device).eval().requires_grad_(False)
    torch.manual_seed(args.phi_seed)
    operator = CS(3, 256, args.ratio, device)
    phi = operator.Vt_small[:operator.cs_size]
    error = phi @ phi.T - torch.eye(operator.cs_size, device=device)
    x = torch.randn(1, 3 * 256 * 256, device=device)
    q = torch.randn_like(operator.A(x))
    adjoint_error = ((operator.A(x) * q).sum() - (x * operator.At(q)).sum()).abs()
    y_test = operator.A(x)
    candidate = torch.randn_like(x)
    projected = candidate - operator.A_pinv(operator.A(candidate) - y_test)
    projection = measurement_metrics(operator, projected, y_test)
    tests = {"orthogonality_frobenius": error.norm().item(),
             "orthogonality_max_abs": error.abs().max().item(),
             "adjoint_abs_error": adjoint_error.item(), "projection": projection}
    print(json.dumps({"operator_tests": tests}), flush=True)
    # Do not re-orthogonalize upstream's GPU-SVD operator to hide its errors.
    tests["strict_1e4_orthogonality_pass"] = error.abs().max().item() < 1e-4
    tests["strict_1e4_projection_pass"] = projection["relative"] < 1e-4
    if not all((tests["strict_1e4_orthogonality_pass"], tests["strict_1e4_projection_pass"])):
        print("WARNING: upstream GPU-SVD operator is not accurate to 1e-4; retained unchanged.", flush=True)
    assert torch.isfinite(error).all() and projection["relative"] < 0.01, tests
    metadata = vars(args) | {"model": "official ImageNet256 unconditional Guided Diffusion",
                            "block_size": 32, "measurements_per_block": operator.cs_size,
                            "effective_ratio": operator.cs_size / 1024,
                            "domain": "RGB [-1,1]", "precision": "upstream FP16 torso",
                            "resize": "PIL bicubic direct resize to 256x256 (not full-image benchmark)",
                            "noise": "eta>0: seed changes initial and subsequent sampling noise; eta=0: initial noise only",
                            "phi_sha256": hashlib.sha256(phi.cpu().numpy().tobytes()).hexdigest(),
                            "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
                            "operator_tests": tests}
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2))
    beta = torch.linspace(0.0001, 0.02, 1000, device=device)
    reports = []
    for path_string in args.inputs:
        path = Path(path_string)
        folder = out / path.stem
        folder.mkdir(exist_ok=True)
        image = Image.open(path).convert("RGB").resize((256, 256), Image.Resampling.BICUBIC)
        image.save(folder / "gt.png")
        gt_array = np.asarray(image, dtype=np.float32) / 255
        gt = torch.from_numpy(gt_array).permute(2, 0, 1).unsqueeze(0).to(device)
        gt_internal = 2 * gt - 1
        y = operator.A(gt_internal.reshape(1, -1))
        torch.save({"phi": phi.cpu(), "y": y.cpu()}, folder / "measurement.pt")
        samples, metrics = [], []
        torch.cuda.reset_peak_memory_stats()
        for index in range(args.num_samples):
            seed = args.seed + index
            torch.manual_seed(seed)
            initial = torch.randn_like(gt)
            observer = DiagnosticOperator(operator, y, args.dc)
            torch.cuda.synchronize()
            start = time.perf_counter()
            result, _ = ddnm_diffusion(initial, model, beta, args.eta, observer, y,
                                       config=namespace(config))
            raw = result[0].to(device)
            torch.cuda.synchronize()
            seconds = time.perf_counter() - start
            visible = ((raw + 1) / 2).clamp(0, 1)
            array = visible[0].permute(1, 2, 0).cpu().numpy()
            torch.save(raw.cpu(), folder / f"sample_{index:02d}_raw.pt")
            Image.fromarray(np.round(array * 255).astype(np.uint8)).save(folder / f"sample_{index:02d}.png")
            item = {"index": index, "seed": seed, "psnr": float(peak_signal_noise_ratio(gt_array, array, data_range=1)),
                    "ssim": float(structural_similarity(gt_array, array, channel_axis=2, data_range=1)),
                    "measurement_raw": measurement_metrics(operator, raw, y),
                    "measurement_clamped": measurement_metrics(operator, 2 * visible - 1, y),
                    "runtime_seconds": seconds, "trajectory": observer.records}
            metrics.append(item)
            samples.append(raw)
            print(json.dumps({"image": path.name, **{k: v for k, v in item.items() if k != "trajectory"}}), flush=True)
        pairs = []
        for i in range(len(samples)):
            for j in range(i + 1, len(samples)):
                delta = samples[i] - samples[j]
                measured = operator.A(delta.reshape(1, -1))
                pairs.append({"i": i, "j": j, "rms_01_scale": (delta / 2).square().mean().sqrt().item(),
                              "l2_internal": delta.norm().item(), "A_delta_l2": measured.norm().item(),
                              "null_ratio": (measured.norm() / delta.norm().clamp_min(1e-12)).item()})
        report = {"image": str(path), "samples": metrics, "pairs": pairs,
                  "mean_sample_psnr": float(np.mean([s["psnr"] for s in metrics])),
                  "peak_memory_gib": torch.cuda.max_memory_allocated() / 2**30}
        (folder / "metrics.json").write_text(json.dumps(report, indent=2))
        reports.append(report)
    (out / "summary.json").write_text(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
