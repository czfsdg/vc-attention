# 在 H200 / H800 对比五种 Attention

`scripts/benchmark_cuda.py` 默认对比下面五种实现。前三种 FlashAttention
通过各自的包接口直接调用，SageAttention 使用公开的 `sageattn` 自动分派接口。

| 参数名 | Python 调用 | 你当前安装的版本 |
| --- | --- | --- |
| `flash2` | `flash_attn.flash_attn_func` | `flash_attn 2.8.3.post1` |
| `flash3` | `flash_attn_3.flash_attn_interface.flash_attn_func` | `flash_attn_3 3.0.0` |
| `flash4` | `flash_attn.cute.flash_attn_func` | `flash-attn-4 4.0.0b31` |
| `sage` | `sageattention.sageattn` | `sageattention 2.2.0` |
| `vc` | `vc_attention.attention(..., Config(backend="cuda_fp8", expcast=True, v_smooth=True))` | `vc-attention 0.2.0` |

运行报告会读取实际安装版本，并保存接口路径和参数。老版 FA3 源码安装若没有
`flash_attn_3.flash_attn_interface`，会尝试同为 FA3 的 `flash_attn_interface`。
若已找到 FA3 但其内部依赖或 CUDA 扩展损坏，会报错。

## 直接运行

你之前在 H200 通过的是 0.1.0。当前 0.2.0 修改了原生 CUDA 与数值契约，
拉取后须重新编译并自检，再运行测速；已有的四个 Attention 包不需要重装。

```bash
git pull --ff-only
bash scripts/validate_h800.sh
python3 scripts/benchmark_cuda.py \
  --batch 1 --queries 1024 --tokens 1024 --heads 2 --dim 128 \
  --dtype bfloat16 --warmup 5 --repeats 20 --trials 3 \
  --output results/h200_attention_compare.json
```

`queries` 是 Q 长度，`tokens` 是 K/V 长度，`heads` 是注意力头数，`dim`
是每个头的维度。上述输入为 `[B,H,N,D] = [1,2,1024,128]`。
脚本会打印环境、预热进度、测试表格，并将完整结果写入 JSON。
FA4 等实现首次调用可能触发 JIT 编译，需要等待。

可以进一步比较不同序列长度，避免根据单个尺寸判断快慢：

```bash
for n in 1024 2048 4096; do
  python3 scripts/benchmark_cuda.py \
    --queries "$n" --tokens "$n" --heads 2 --dim 128 --dtype bfloat16 \
    --output "results/h200_n${n}.json"
done
```

然后将 `--heads`、`--batch`、Q/KV 长度改成实际模型的尺寸；目前 VC 是多内核
实现，大尺寸可能耗时较长。测试时尽量保证这张 GPU 没有其他计算任务。

## 结果怎么看

| 列 / JSON 字段 | 意义 |
| --- | --- |
| `median ms` / `median_ms` | 每轮重复调用的平均毫秒数，再对多轮取中位数；越小越快 |
| `min ms`、`max ms`、`trial_ms` | 不同轮次的波动范围及原始计时 |
| `speedup` / `speedup_vs_baseline` | 默认 FA3 耗时 ÷ 当前行耗时；大于 1 表示当前行比 FA3 快 |
| `vc_speedup_vs` | 各对手耗时 ÷ VC 耗时；大于 1 表示 VC 更快 |
| `rel RMSE` / `relative_rmse_vs_fp32` | 相对 FP32 参考输出的均方根误差，数值为比例；0.01 表示 1% |
| `max abs` / `max_abs_error_vs_fp32` | 相对 FP32 参考输出的最大绝对误差 |

例如，**假设** FA3 为 1 ms、VC 为 2 ms，则 VC 相对 FA3 为 `0.5x`，
即 VC 用时是 FA3 的 2 倍。这只是读表举例，不是实测数据。
当前 VC 尚未融合为单个 FlashAttention 内核，实测前不预设它更快。

## 比较范围和计时方式

- 所有实现使用相同 Q/K/V 数值、FP16 或 BF16 输入，进行 dense、noncausal、
  无 dropout 的 MHA 前向推理；不测 backward、GQA、稀疏或 causal attention。
  各实现内部的量化格式与累加精度不同，因此同时报告误差。
- FA2/3/4 使用连续 `[B,N,H,D]` 输入，Sage/VC 使用连续 `[B,H,N,D]` 输入。
  所需布局副本在计时前生成；结果布局转换和正确性检查也不计时。
- 计时覆盖 Python API 调用、内部量化、平滑、分配及 GPU 运算。
  VC 每次都执行第 0 步的 V-Smooth 聚类，不传入跨步布局缓存；
  Sage 的内部预处理也计入耗时。此结果不能代表 VC 缓存复用后的后续步性能。
- 每个实现先预热，排除首次 JIT 编译；每轮按固定种子打乱执行顺序。
  使用 `perf_counter` 并在每轮开始/结束同步 CUDA，测量完整调用的墙钟耗时。
  未启用 CUDA Graph，也未把结果解释为单个 GPU kernel 的耗时。
- 误差默认检查均匀采样的最多 128 个 query，覆盖所有 batch/head、完整 K/V。
  FP32 参考关闭 TF32 并按 query 分块计算，避免存储整个 N×N 分数矩阵。
  JSON 保存实际 query 索引；`--check-queries 0` 可检查全部 query。
  表格报告预热及各轮最后一个输出中观察到的最差误差，不设置通用精度通过阈值。
- 输入是随机合成数据，不能代替真实模型质量评估。四个库的实际兼容性和耗时
  需要在你的 H200 环境运行；本地只验证适配、参考结果和计时逻辑。

## 排查单个库和 VC 消融

任何选定实现导入或运行失败都会带后端名称报错，不会静默换成 SDPA。
可以明确选择子集定位问题：

```bash
python3 scripts/benchmark_cuda.py --backends flash3 vc --output results/fa3_vs_vc.json
python3 scripts/benchmark_cuda.py --backends flash4 --output results/fa4_only.json
python3 scripts/benchmark_cuda.py --backends sage vc --baseline sage --output results/sage_vs_vc.json
```

`--baseline` 必须在 `--backends` 中；未指定时优先使用 FA3，否则使用第一个后端。
加 `--vc-ablation` 会在 VC 完整实现之外加入 FP8 control、仅 ExpCast、
仅 V-Smooth 三组，用来观察两个算法改动各自的时延和误差影响：

```bash
python3 scripts/benchmark_cuda.py --vc-ablation --output results/h200_ablation.json
```

## 区分实现差异与量化算法误差

在同一份 Q/K/V 上运行原生 CUDA、同配置的 Python 量化参考、标准 FP32
Attention，才能判断约 5% 的误差是否也存在于量化算法参考中。
拉取更新后先重建新版扩展：

```bash
git pull --ff-only
bash scripts/validate_h800.sh
python3 scripts/benchmark_cuda.py --backends vc --vc-ablation --vc-reference-check --check-queries 0 --output results/h200_vc_reference.json
```

默认形状仍为 `[1,2,1024,128]`，BF16 输入、固定随机种子 1234。
0.2.0 的四组都默认开启 K 去均值、Q/K Hadamard，并使用新版 FMA 与缩放均值契约；
因此不能把新旧结果差异单独归因于某一个改动。详见 [PAPER_ALIGNMENT.md](PAPER_ALIGNMENT.md)。
`--vc-reference-check` 会在全部测速结束后，分别运行四组同配置的
`Config(backend="reference")`。只改 backend，其余分块、量化、ExpCast、V-Smooth、
聚类及均值精度参数保持一致。参考在同一张 GPU 上运行，FP8 编码/解码后使用
FP32 矩阵乘，关闭 TF32；它的耗时不进入性能表。

末尾新增的表格以**百分数**显示相对均方根误差：

| 列 | 含义 |
| --- | --- |
| `CUDA/FP32 %` | 原生 CUDA 与标准 FP32 Attention 的差异 |
| `REF/FP32 %` | Python 量化参考与标准 FP32 Attention 的差异 |
| `CUDA/REF %` | 原生 CUDA 与同配置 Python 量化参考的差异 |
| `CUDA/REF max abs` | 原生 CUDA 与 Python 量化参考的最大绝对差异 |

例如，前两列都约 5%、第三列远小于 5%，支持主要误差来自当前量化算法的解释。
若第三列也明显偏大，需要进一步检查原生实现、聚类排列和数值舍入。
两个实现都会独立执行预处理，因此第三列衡量整个实现的差异，并非仅矩阵乘内核。
不能通过相减前两列来计算第三列，三者都直接比较对应的输出张量。
参考接近也不代表模型质量已经达标，这个诊断不设置通用通过阈值。

对拍使用每组最后一轮最后一次原生调用的输出，三种误差对应同一组输出；
上方性能表的误差仍为各次检查中观察到的最差值，两表可能略有不同。
对拍始终先处理完整 Q/K/V，再按 `--check-queries` 抽取输出；不能先裁剪 Q，
否则会改变 Q 的块量化 scale。上述命令检查全部 query。完整诊断保存在 JSON
的 `vc_reference_check` 字段中，JSON 内的相对误差仍为比例，`0.01` 表示 1%。

接口参考：[FlashAttention 官方文档](https://github.com/Dao-AILab/flash-attention#readme)、
[FA3 接口源码](https://github.com/Dao-AILab/flash-attention/blob/main/hopper/flash_attn_interface.py)、
[FA4 接口源码](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/cute/interface.py)、
[SageAttention 2.2.0 接口源码](https://github.com/thu-ml/SageAttention/blob/v2.2.0/sageattention/core.py)。
