#!/usr/bin/env bash
set -euo pipefail
cd /home/ps/Downloads/DDNM-CS-validation
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2
python_bin=/home/ps/miniconda3/envs/dmp-faithful/bin/python
checkpoint=/home/ps/Downloads/LiteDC-UDiT-CS/pretrained/256x256_diffusion_uncond.pt
inputs=(/home/ps/Downloads/IDM/data/Set11/Monarch.tif exp/datasets/imagenet/imagenet/ILSVRC2012_val_00047724.JPEG)
mkdir -p runs
"$python_bin" -u scripts/diagnose_cs_multisolution.py --checkpoint "$checkpoint" --inputs "${inputs[@]}" --output_dir runs/cs010_official_eta085 --steps 100 --num_samples 4 --eta 0.85 > runs/cs010_official_eta085.log 2>&1
"$python_bin" -u scripts/diagnose_cs_multisolution.py --checkpoint "$checkpoint" --inputs "${inputs[@]}" --output_dir runs/cs010_initial_only_eta0 --steps 100 --num_samples 4 --eta 0 > runs/cs010_initial_only_eta0.log 2>&1
"$python_bin" -u scripts/diagnose_cs_multisolution.py --checkpoint "$checkpoint" --inputs "${inputs[@]}" --output_dir runs/cs010_no_dc --steps 100 --num_samples 2 --eta 0.85 --dc none > runs/cs010_no_dc.log 2>&1
printf 'DDNM_VALIDATION_COMPLETE\n'
