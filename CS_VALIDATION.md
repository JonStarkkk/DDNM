# Official DDNM: same-measurement multi-sample diagnostic

Upstream: `wyhuai/DDNM`, commit `00b58eac7843a4c99114fd8fa42da7aa2b6808af`.
Fork: https://github.com/JonStarkkk/DDNM

This evaluation adds only an independent script. No upstream model, CS operator,
or sampling function is edited. It directly calls `functions.svd_ddnm.ddnm_diffusion`
and the official `functions.svd_operators.CS` class.

## Protocol

- Frozen unconditional ImageNet256 Guided Diffusion checkpoint; strict load.
- Official FP16 torso, 1000 linear beta entries, 100 reverse sampling nodes.
- Official 32x32 block CS, independent RGB channel measurements, shared matrix.
- Requested ratio 0.1; actual block measurement count 102/1024 = 0.099609375.
- Fixed operator seed 1234. Same image, operator, and measurement across samples.
- Four sequential trajectories with seeds 2026--2029. No candidate interaction.
- Default eta=0.85 changes both initial and intermediate trajectory noise.
- Separate eta=0 experiment isolates initial-noise variation.
- Two-sample no-DC control returns zero correction without changing the upstream
  sampler; this is an unconditional generation control, not a reconstruction method.
- Input images are converted to RGB and directly bicubic-resized to 256x256.
  These are pilot diagnostics, NOT a full-resolution Set11 benchmark comparable
  to earlier DMP/IDM numbers.
- CS measurements use the internal [-1,1] domain. PSNR/SSIM use clamped [0,1]
  floating-point RGB images. Raw and clamped measurement residuals are separate.

## Numerical boundary

The official single-precision CUDA SVD produced a maximum row-orthogonality
error of 5.38e-4 and a random-projection relative residual of 7.12e-4 in this
environment. It therefore failed the strict 1e-4 checks. We record the failures
and retain the operator unchanged rather than silently re-orthogonalizing it.
Actual image sampling residuals must be read from the result files independently.

The diagnostic script can run in the existing `dmp-faithful` environment without
installing the unrelated dataset-loader dependencies of upstream `main.py`.

## Ubuntu locations

Repository: `/home/ps/Downloads/DDNM-CS-validation`

Weight (reused, not copied):
`/home/ps/Downloads/LiteDC-UDiT-CS/pretrained/256x256_diffusion_uncond.pt`

Results:
- `runs/smoke_5steps`
- `runs/cs010_official_eta085`
- `runs/cs010_initial_only_eta0`
- `runs/cs010_no_dc`

Each image folder includes GT and sample PNGs, original floating-point samples,
the fixed measurement and matrix, and metrics including all pairwise differences.
The trace records pre/post-DC clean-prediction residual at each reverse step.

Re-run the small validation with `bash scripts/run_cs_validation.sh` (GPU0).
Do not run this if another experiment occupies GPU0.

## Interpretation

Distinct samples with small residual demonstrate empirical measurement-compatible
diversity, not exact Bayesian posterior sampling, coverage of all feasible images,
or calibrated uncertainty. Visual plausibility must be inspected separately.
