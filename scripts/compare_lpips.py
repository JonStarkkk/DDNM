"""Read-only LPIPS comparison of saved 10% and 25% DDNM samples."""
import json
from itertools import combinations
from pathlib import Path

import lpips
import numpy as np
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1] / "runs"
MODES = ("cs010_official_eta085", "cs025_official_eta085")
IMAGES = ("Monarch", "ILSVRC2012_val_00047724")


def read_rgb(path, device):
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device) * 2 - 1


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metric = lpips.LPIPS(net="squeeze", version="0.1").eval().to(device)
    results = []
    with torch.inference_mode():
        for mode in MODES:
            for name in IMAGES:
                folder = ROOT / mode / name
                gt = read_rgb(folder / "gt.png", device)
                samples = [read_rgb(folder / f"sample_{i:02d}.png", device) for i in range(4)]
                to_gt = [metric(x, gt).item() for x in samples]
                pairwise = [{"i": i, "j": j, "lpips": metric(samples[i], samples[j]).item()}
                            for i, j in combinations(range(4), 2)]
                result = {"mode": mode, "image": name, "backbone": "squeeze", "version": "0.1",
                          "input": "saved RGB PNG normalized to [-1,1]",
                          "to_gt": to_gt, "mean_to_gt": float(np.mean(to_gt)),
                          "pairwise": pairwise,
                          "mean_pairwise": float(np.mean([p["lpips"] for p in pairwise]))}
                results.append(result)
                print(json.dumps(result), flush=True)
    output = ROOT / "lpips_10_vs_25.json"
    output.write_text(json.dumps(results, indent=2))
    print(f"saved {output}")


if __name__ == "__main__":
    main()
