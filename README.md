# MiniMax-H3 / A5 VC-Attention 实验

新增 H800 CUDA C++ 后端：`Config(backend="cuda_fp8")`。源码、`pip install .`
构建方式和远端验证步骤见 [CUDA_README.md](CUDA_README.md)。QK/PV 使用真实
cuBLASLt FP8 矩阵乘；H800 上的正确性和性能仍需真机验证，尚不是完整融合内核。
下文原有 A5 实验记录和两种后端的说明保留为历史背景；默认后端仍为 `reference`。

这是可运行的**算法与接线原型**。已完成 CPU 数值测试；A5 自检、真实 H3 张量回放、视频/音频生成、融合内核性能均尚未验证。请先抓真实 Q/K/V 回放，再决定融合开发是否值得继续。

## 实现范围

| 文件 | 用途 |
| --- | --- |
| `core.py` | ExpCast、V 聚类重排、128-token 块去均值、在线均值恢复、步数调度 |
| `npu.py` | A5 原生 FP8 矩阵乘实验后端和设备自检 |
| `lightx2v.py` | LightX2V H3 注册表接入、受限张量抓取、生成入口 |
| `replay.py` | 同输入四组对照，以及单独的 MindIE BF16/FP8 基线 |
| `tests/` | 独立 dense oracle、编码、状态、适配器契约、A5 检查 |
| `results/` | 本机合成数据报告；不是 H3 实测结果 |

两个实验执行后端：

- `reference`：E4M3 编码后解码，用 FP32 矩阵乘进行数值模拟，可在 CPU 或 NPU 上执行。**不是 FP8 硬件测速。**
- `npu_fp8`：调用 `torch_npu.npu_add_quant_matmul_` 的 MX 接口，E8M0 scale 字节固定为 127（即 1），让 Cube 消费原始 E4M3 payload；普通量化 scale 在外部恢复。FP32 累加。此路径先跑设备自检，不支持时直接报错，无静默回退。

两者都以 Python 分块调度，尚未融合到 FIA。每个 tile 都会下发多个算子，**不应把原型时延解释成 ExpCast 的加速能力**。长序列完整 H3 推理可能非常慢；优先回放少量 query，保留完整 K/V。

## 数值契约与论文差异

按 [VC-Attention v1](https://arxiv.org/html/2609.15810v1) 的公式重建，非作者代码：

- ExpCast：`uint8(clamp(round((S-rowmax)*8*log2(e)+119.65),0,120))`，随后按位 `view(float8_e4m3fn)`，概率 scale 为 `1/256`。Eager multiply/add 不保证单次 FMA 舍入，融合后须重新对拍边界字节。
- V-Smooth：每头独立聚类，K/V 同序重排，Q 保持原序；块均值以 FP16 保存，残差按每块每通道 E4M3 量化；`rowmass * mean` 加到在线累加器，随 running-max 一起 rescale。
- 普通 exp 路径沿公式 6，用 FP32 exp 的 rowmass 归一化/补均值，PV 使用舍入后的 FP8 P；ExpCast 路径用解码后的 P 同时计算 PV、rowmass 和归一化。
- 默认前 `ceil(0.25 * 实际去噪步数)` 步去均值，间隔 4 步更新 permutation；之后保留 permutation 并关闭去均值。只缓存排列/中心，每次重新算当前 V 的均值和残差。4 步模型只有第 0 步去均值。
- 聚类默认 8 类、2 次 Lloyd 迭代，确定性等间距初始化。这是可调实验选择，未声称复现作者的聚类内核。
- Q/K 分别按 128/256 tokens、128 channels 块量化。未额外加入 QK Hadamard、K smoothing，避免扩大本轮变量。

`fp8_control / expcast / v_smooth / combined` 四组共享同一原型量化契约。**`fp8_control` 不等于现有 MindIE FP8 FIA**：后者 V/P 的块结构、概率处理和内部累加存在差异，报告中另列 `mindie_fp8`，不能把两者差值全部归因于 V-Smooth/ExpCast。

## 本地复现

进入 MindIE-SD 根目录，使用已有 PyTorch（需 E4M3 dtype）与 pytest：

```bash
export PYTHONPATH="$PWD/examples/minimax-h3:${PYTHONPATH:-}"
python -m pytest examples/minimax-h3/vc_attention/tests -q --noconftest
python -m vc_attention.replay --clustered --output vc-results/clustered.json
python -m vc_attention.replay --output vc-results/gaussian.json
```

`--noconftest` 隔离仓库其他需要 NPU 的 pytest 配置。运行时不会导入本地 MindIE 扩展，除非指定 `--mindie` 或启动原生 MindIE 基线。

## A5 执行顺序

沿用你已经验证可用的 CANN 9.1.0、torch_npu、MindIE-SD 环境，不用本机测试依赖覆盖设备环境。推荐将交付压缩包解到独立目录，避免未编译的 MindIE 源码遮蔽设备上已安装的库：

```bash
mkdir -p /path/to/vc-experiment
tar -xzf vc-attention-h3-experiment.tar.gz -C /path/to/vc-experiment
export PYTHONPATH="/path/to/vc-experiment:${PYTHONPATH:-}"
cd /path/to/vc-experiment
python -c 'import mindiesd; print(mindiesd.__file__)'
```

基线调用基于本次 MindIE `dev` 的 `quant_attention` 公共接口；若设备版本较旧且没有该入口，需要安装对应版本或接到你已验证的 FP8 wrapper，不能把接口不存在解释成 A5 不支持 FP8。下方 pytest 路径按解包方式使用。

先验证新增的 FP8 矩阵乘路径，及本机未跑的原生算子检查：

```bash
python -m vc_attention.npu --device npu:0
python -m pytest vc_attention/tests/test_npu.py -q --noconftest
```

检查包含真实 FP8 乘法、尾块补零、E4M3 reinterpret、RNE、四组结果与 CPU oracle 对拍。这里验证的是**新增路径**，不重复质疑已验证的普通 FP8 Attention。若环境缺 `npu_add_quant_matmul_`，仍可显式用 `reference` 检查算法误差。

### 抓 H3 的真实张量

接入基于 LightX2V 的 MiniMax-H3：在 RoPE 与 Ulysses All-to-All 之后，进入 attention 的位置抓取完整 K/V。默认仅 rank 0、第 0/4 步、第 0/12 层、每次 1 个头、最多 4 份、512 MiB 张量预算。基线输出仍走 MindIE 原生 FP8。

保持你已有的模型、提示词、seed、分辨率和步数；下例中 `H3_A5_CONFIG` 应指向已跑通的 A5 LightX2V 配置。示例 task 在 LightX2V 中拼作 `t2av`。

```bash
export H3_MODEL=/path/to/MiniMax-H3
export H3_A5_CONFIG=/path/to/known-working-a5-config.json
torchrun --nproc_per_node=4 --module vc_attention.lightx2v \
  --mode capture --output-dir vc-results/capture -- \
  --model_cls minimax_h3 --task t2av \
  --model_path "$H3_MODEL" --config_json "$H3_A5_CONFIG" \
  --prompt 'A red scarf moving in the wind by the sea.' --seed 1101 \
  --save_result_path vc-results/capture/baseline.mp4
```

入口生成配置副本，保留其他参数，注册独立 `mindie_vc_experiment`，文本 refiner 单独使用 `npu_flash_attn`。原配置不被覆盖。

当前约束：单序列、dense、非 causal、无 mask/bias/dropout、相同 Q/K/V 头数；不支持 packed、GQA、ring、head-parallel、CFG 或 compile。原配置中 `use_compile`、`warmup` 和 `parallel.seq_p_head_parallel` 必须关闭，所有对照组一致。显式关闭框架 warmup，防止把预热提示词的张量当作真实请求抓取；replay 有独立预热。未知参数显式拒绝，避免静默改变注意力语义。并发服务不在本适配器范围内；每次启动用于单一推理任务。

可以先 `--prepare-only` 检查生成的配置；它不导入 LightX2V、MindIE 或 torch_npu。

### 回放同一份张量

```bash
python -m vc_attention.replay \
  --capture vc-results/capture/captures/rank0_step0_layer0.pt \
  --device npu:0 --backend npu_fp8 --mindie \
  --q-start 0 --q-limit 128 --output vc-results/h3-layer0.json
```

`q-start` 必须位于 128-token 块边界。增加它以覆盖音频、视频、文本等不同位置；`q-limit=0` 表示完整 Q。始终保留全部 K/V，不把裁剪 context 的结果冒充完整 Attention。先跨层/跨步/跨头测 relative RMSE、最大绝对误差、cosine，再选参数；单次 replay 强制处于 smoothing 窗口，用于隔离算法，不能当整段去噪收益。

报告包含原张量 SHA256、形状、scale、运行时、完整采样时延、重复次数、参考定义。原生 MindIE BF16/FP8 与原型四组分列。计时含分组、量化、分块循环，交替反转顺序并同步设备；属于探索结果，无融合加速结论。

### 整段生成对照

数值和设备自检通过后，在小规模场景使用同一入口分别指定以下 `--mode`，每组使用独立 `--output-dir`、`--save_result_path`：

| mode | 作用 |
| --- | --- |
| `mindie_bf16` | 原生高精度生成参考 |
| `mindie_fp8` | 原生 FP8 生成基线，可选 `--fp8-mode C8V16_TILING512` |
| `fp8_control` | 原型不启用两种算法，排除后端量化契约变化 |
| `expcast` | 只开 ExpCast |
| `v_smooth` | 只开 V-Smooth，应用前 25% 步调度 |
| `combined` | 两者一起 |

审查 `audit.rank*.json` 中的 `calls`、`expcast_calls`、`smoothing_calls`、`regroup_calls`，确认开关确实命中。关闭本实验改回原来的 LightX2V 入口/配置即恢复基线。

同 prompt/seed/步数/模型权重比较视频帧、运动、细节及音画同步。可复用仓库 `evals/scripts/quality_compare.py`，对同帧索引的解码图片目录计算 SSIM/PSNR：

```bash
python evals/scripts/quality_compare.py --baseline bf16_frames \
  --config combined_frames --metric ssim --output vc-results/quality.json
```

音频质量须另查，不能只看视频帧得出 H3 整体无损结论。原型完整推理仅用于质量探索；要评估性能，需要把 ExpCast 写进 FIA vector Softmax，把均值恢复接到 PV 的在线累加，并融合 gather/quantize。相关来源位置记录在 `PROVENANCE.json`。
