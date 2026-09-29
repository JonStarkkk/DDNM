# DDNM 10%与25%压缩感知结果的LPIPS比较

## 指标协议

- 使用官方`lpips`包0.1.4、`LPIPS(net="squeeze", version="0.1")`。
- ImageNet预训练SqueezeNet1_1特征主干，LPIPS v0.1校准权重。
- 对**保存的RGB PNG**读取并归一化到`[-1,1]`。这与视觉展示一致；
  不是未截断的浮点`sample_*_raw.pt`。两种输入协议不要混用。
- 候选对真值：`LPIPS(sample_i, gt)`，四个样本求均值，越低通常越接近真值。
- 候选间：六个无序样本对的LPIPS均值，越高表示特征差异越大，
  但不等于更好的、合理的后验多样性。
- 输入分别为Set11 Monarch与官方仓库自带的ImageNet公鸡，统一直接缩放到256×256。
- 其余设置参见`CS_25_PERCENT_COMPARISON.md`。25%矩阵的前102行与10%完全相同。

## 结果

|图像|测量率|平均候选-真值LPIPS↓|平均候选-候选LPIPS|
|---|---:|---:|---:|
|Monarch|10%|0.4803|0.0817|
|Monarch|25%|0.2280|0.0352|
|ImageNet公鸡|10%|0.2312|0.0502|
|ImageNet公鸡|25%|0.0961|0.0253|

对应PSNR：Monarch由20.26升至27.58 dB，公鸡由25.47升至28.89 dB；
像素RMS差异亦下降。这些指标方向一致：25%测量改善单样本重建质量，
使候选彼此更接近，但没有完全相同。

Monarch的候选-真值LPIPS下降约52.5%，候选间LPIPS下降约57.0%；
公鸡分别下降约58.4%与49.6%。百分比仅描述本次两张图，
LPIPS不是线性“质量百分数”。

## 解释边界

LPIPS没有证明10%候选差异是真实后验歧义。10%图像存在明显颗粒噪声，
这些伪影本身也会抬高候选间LPIPS。25%差异下降可能既来自条件不确定性减小，
也来自噪声伪影减少，无法仅凭LPIPS分离。

本实验也没有独立的人类视觉评价、FID/KID或后验校准。
候选-真值LPIPS较低不保证测量一致；本项目测量残差需另行报告。

## 文件

Ubuntu：`/home/ps/Downloads/DDNM-CS-validation/runs/lpips_10_vs_25.json`

本地：`runs/lpips_10_vs_25.json`，含所有逐候选与逐对数值。

可在Ubuntu仓库根目录用现成依赖复算：

```bash
CUDA_VISIBLE_DEVICES=0 /home/ps/miniconda3/envs/dmp-faithful/bin/python scripts/compare_lpips.py
```
