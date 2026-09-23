# VC-Attention 0.2.0：论文数值对齐记录

对照 [VC-Attention v1](https://arxiv.org/html/2609.15810v1) 的公式 4、6、7
与附录 B，修正 0.1.0 的遗漏。当前是根据论文重建的 CUDA 数值实现，
尚未复现作者的融合内核、完整模型质量或性能结果。

| 项目 | 0.2.0 实现 |
| --- | --- |
| Q/K 预处理 | 默认开启 K 每通道去均值和 Q/K 共同的归一化 Hadamard，再做 E4M3 量化 |
| RoPE | 接口接收已经完成 RoPE 的 Q/K；调用方负责位置编码，不重复应用 |
| V-Smooth | 每 batch/head 聚类、K/V 同序排列、默认 128-token 块内去均值 |
| V 残差 | 使用 FP32 块均值计算残差，再按块、按通道量化 E4M3 |
| 均值存储 | 实际以 FP16 保存 `mu / V_scale`，在线恢复时与 PV 残差共用一次 scale；BF16/FP32 为可选实验配置 |
| ExpCast | 原生 CUDA 使用单次 FP32 `__fmaf_rn`，再 RNE 和字节重解释；参考以 FP64 中间运算模拟该 FP32 舍入 |
| 普通 exp | PV 消费舍入后的 FP8 P；分母及均值恢复使用 FP32 exp 的行和 |
| ExpCast 行和 | 分母与均值恢复共同使用解码后的 P |
| 去噪调度 | 前 `ceil(总步数 / 4)` 步分组/去均值，间隔 4 步刷新，warm-start 中心；窗口之后保留排列 |

K 去均值与共同正交旋转在精确算术中保持 dense attention 输出。
当前先对 post-RoPE K 去均值，再旋转，二者在线性代数上可交换；
不能据此声称与作者未核对过的融合代码逐位相同。

## 明确保留的实现选择

- 聚类 8 类、2 次 Lloyd 迭代、等间距初始中心；论文未给出足够细节来确认这些具体参数。
- 使用确定性的 Sylvester Hadamard；未假定作者的随机符号或 seed。
  非 2 的幂次的 head dimension 补零后旋转，softmax 仍按原始维度缩放。
- Q/K 继续使用现有 E4M3 和 128/256-token 量化块；不能视为已逐项核对作者 host kernel 的所有量化细节。
- 极小 V 残差可能使 `mu / scale` 超过均值格式范围，因此对 scale 增加有限存储下界。
  这是防溢出的实现选择，不是论文报告的调参结果。
- `Config(k_smooth=False, qk_hadamard=False)` 可以做预处理消融，仍采用新版 ExpCast 和 V 均值契约，不能称为完整恢复 0.1.0。
- QK/PV 仍调用 cuBLASLt FP8，预处理及 attention 尚未完全融合。本轮不预设性能提升。

所有 benchmark 的 VC 消融组默认共用新增 Q/K 预处理，JSON 中会记录配置。
本轮验证 CPU/CUDA；NPU 路径尚未重新验证，新的参考 ExpCast 需要 FP64 中间运算，
不应假定不支持该运算的设备可直接运行。

## 验证和远端复验

本地 Windows / CUDA 12.1 / PyTorch 2.3.1：扩展已编译 `sm_86`、`sm_90`，
安装后测试 **115 passed, 30 skipped**。其中实际 CUDA 预处理与 ExpCast 测试
运行在 RTX 3060 上；跳过的是 28 项 Hopper attention 测试和 2 项 NPU 测试。
Hadamard 使用独立显式矩阵对照，FMA 包含与 C `fmaf` 核对过的固定边界字节，
V 均值测试检查先缩放再保存与近常量块防溢出。

此前用户提供的 H200 **60 passed** 是 0.1.0 的结果，不覆盖本次改动。
0.2.0 原生扩展接口已变更，必须重新编译；旧扩展会明确要求重建：

```bash
git pull --ff-only
bash scripts/validate_h800.sh
python3 scripts/benchmark_cuda.py --backends vc --vc-ablation --vc-reference-check --check-queries 0 --output results/h200_vc_paper.json
```

重点同时检查 CUDA/FP32、REF/FP32、CUDA/REF。随机输入的误差不能直接等同于论文视频质量；
后续仍需真实模型的 post-RoPE Q/K/V 和固定 prompt/seed 的生成对照。
