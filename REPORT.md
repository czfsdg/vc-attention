# 本轮实验记录

状态：**本地算法测试完成，A5/H3 实测待执行**。设备不可从本机访问，未执行远程部署、CANN 编译或生成视频。

实现新增于独立的 `MindIE-SD-vc-attention/examples/minimax-h3/vc_attention`，原有 MiniMax/FastVideo 工作目录未改。MindIE 公共 API、原有 FIA kernel、版本源和框架核心文件均未修改。

## 数值结果（合成数据，仅用于验证算法行为）

统一 seed=20260917，B=1，H=2，D=128，完整 K/V 长度 1024，Q 取前 128 行；FP32 streaming reference；每个原型采用同一 E4M3 量化契约，V-Smooth 本次处于启用窗口。

| 合成输入 | FP8 原型对照 | + ExpCast | + V-Smooth | 两者一起 |
| --- | ---: | ---: | ---: | ---: |
| 有聚类结构的 V | 1.5905% | 1.6144% | 0.8005% | 0.8541% |
| 高斯随机 V | 5.1165% | 5.3739% | 5.1660% | 5.4135% |

表中为输出 relative RMSE（越小越好）。聚类构造刻意提供了可被去均值去除的结构；这一结果说明实现能利用该结构，不代表 H3 真实收益。高斯输入略退化，也说明不能预设 V-Smooth 总会改善精度。ExpCast 有近似误差，是否值得使用取决于真实 A5 融合后的性能与 H3 质量权衡。

原始数值和本机原型耗时保存在 `results/synthetic_clustered.json`、`results/synthetic_gaussian.json`。没有将 CPU/Python 原型耗时换算成 A5 加速比。

## 验证证据与边界

- Test-first：先写 core 测试，首次 pytest 因 `vc_attention.core` 尚未实现而退出 2；实现后修复 PyTorch 2.2 的 `argsort(stable=True)` keyword-only 兼容问题，核心测试通过。
- 后续加入 LightX2V adapter 契约检查、原生 MindIE 参数 spy、ExpCast 仅保留逐行 exp 检查、真实 A5 测试。最新 pytest 和 Ruff 输出见 `results/pytest.txt`、`results/ruff.txt`。
- CPU 检查覆盖独立 dense oracle、ExpCast IEEE 字节例子/NumPy 解码 oracle、running-max 更新中的均值恢复、常量 V、非整齐尾块、请求隔离、shape 变化、调度窗口、拒绝未知 mask/causal/packed 语义、capture 保留完整 K/V。
- **真实 NPU 测试在本机跳过**。mock 只证明 adapter 调用契约，没有当作 A5 算子通过。
- 最新测试为 **28 passed / 2 skipped**，Ruff 通过，`--prepare-only` 配置生成与 replay CLI 检查通过。Markdownlint 未执行成功：本机没有 npx；已人工检查新增文档的代码围栏和命令。
- 没有自研融合算子，本轮复用公开 torch_npu API。未执行 CANN 编译、msprof、H3 全模型或视频/音频质量判定。

## 复盘与下一阶段

本轮只开发隔离实验模块，未引入公共 API；未改变现有版本源、文档入口或 contributor workflow。未对原仓文件做格式化。框架接线采用运行期注册新后端，需锁定 LightX2V 版本，属于实验接入而非上游能力声明。

仓库建议在远端形成“实现 → 编译/部署 → pytest”闭环；因用户明确 A5 暂不可访问，当前只完成本地数值/接口检查，并提供可执行设备自检与抓取回放入口。后续 adapter 测试是在实现后补充，未将它们写成 test-first 证据。外部 cannbot 未加载：本轮未写 native DSL 融合 kernel，不进入该开发阶段。

下一阶段先在 A5 执行 `README.md` 的 preflight/capture/replay，判断真实 H3 层上的 V 结构和误差，再融合 FIA Softmax/PV 路径。只有完成实际设备性能和生成质量对照，才能回答“接入后快多少、画质如何”。
