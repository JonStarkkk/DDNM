# 25%与10%分块CS：受控DDNM多样性比较

## 设置

官方DDNM、冻结ImageNet256 unconditional Guided Diffusion，100步，eta=0.85，
无时间回跳。两张图各4个样本，seed=2026–2029，RGB，直接bicubic缩放到256×256。
官方32×32分块算子，phi_seed=1234。

本次只改变测量行数：10%对应102/1024，25%对应256/1024。
实际保存矩阵已比较：`phi_10 == phi_25[:102]`，完全相同，最大误差0。
因此是同一正交基的嵌套测量，不是独立随机矩阵对比。

## 结果（4个样本平均）

|图像|采样率|PSNR|SSIM|候选间平均RMS（[0,1]尺度）|最大原始相对测量残差|
|---|---:|---:|---:|---:|---:|
|Monarch|10%|20.2599|0.5507|0.07761|6.34e-6|
|Monarch|25%|27.5750|0.8282|0.04122|5.99e-6|
|ImageNet公鸡|10%|25.4657|0.6717|0.04204|6.93e-6|
|ImageNet公鸡|25%|28.8894|0.8117|0.03321|6.40e-6|

Monarch提高7.3152 dB，ImageNet提高3.4237 dB。
25%时差异泄漏比平均为2.16e-5和3.91e-5，差异仍近似位于测量零空间。
每个样本约8.3秒，显存峰值约1.55 GiB，测试完成后GPU已释放。

## 视觉检查及结论边界

已检查两张图的sample_00与sample_01。
25%下Monarch翅膀轮廓、斑点和主体明显更清晰，背景粗颗粒噪声比10%减少；
ImageNet公鸡的羽毛、轮廓和面部细节也更清晰。不同样本主体一致，细节、纹理和背景仍有差异。

可以说：增加嵌套测量改善质量并缩小候选差异，但未发生完全collapse。
不能说：这些差异全部是合理后验模式，或已经排除了残余噪声/伪影。
尚未计算LPIPS、执行多人视觉评审或不确定性校准，且只有两张图。

原始浮点结果与clamp结果必须区分：25%公鸡的clamp后最大相对残差约0.002026，
高于原始结果6.40e-6；PNG的量化还可能产生额外误差。

本实验不是论文25% Walsh–Hadamard/ImageNet1K表格的复现，不能直接对比论文均值。

## 保存位置

Ubuntu：`/home/ps/Downloads/DDNM-CS-validation/runs/cs025_official_eta085/`

本地：`runs/cs025_official_eta085/`

逐图目录有GT、4张重建PNG、原始tensor、实际矩阵和测量、完整metrics.json。
总汇总为summary.json。25%的measurement.pt保存了实际256行矩阵，可自行核对嵌套关系。

## 运行命令（Ubuntu仓库根目录）

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 /home/ps/miniconda3/envs/dmp-faithful/bin/python -u scripts/diagnose_cs_multisolution.py --checkpoint /home/ps/Downloads/LiteDC-UDiT-CS/pretrained/256x256_diffusion_uncond.pt --inputs /home/ps/Downloads/IDM/data/Set11/Monarch.tif exp/datasets/imagenet/imagenet/ILSVRC2012_val_00047724.JPEG --output_dir runs/cs025_official_eta085 --ratio 0.25 --steps 100 --num_samples 4 --eta 0.85
```
