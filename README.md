# Megatron SFT 显存估算器

面向 Megatron-LM 全参数 SFT 的浏览器端显存规划工具。根据模型结构、GPU 规格、批次设置和并行拓扑，估算最繁忙 Pipeline Stage 的单卡峰值显存。

## 功能

- 内置 Qwen3.5 Dense 与 MoE 模型配置，也可直接编辑 Hugging Face 风格的模型 JSON。
- 支持 TP、PP、DP、CP、EP、ETP 等并行维度及拓扑合法性检查。
- 分别估算模型权重、梯度、Adam 优化器状态、激活值和运行时预留。
- 对比关闭、选择性和全量激活重计算，以及分布式与常规 Adam 的显存差异。
- 展示各 PP Rank 的层分配、参数分布和峰值显存，便于定位不均衡 Stage。
- 提供参数量拆解、关键网络维度和并行策略说明。

## 本地运行

需要 Node.js `>=22.13.0`。

```bash
npm install
npm run dev
```

其他常用命令：

```bash
npm run lint   # 静态检查
npm run build  # 生产构建
npm test       # 构建并检查服务端渲染结果
```

## 使用方式

1. 从 Qwen3.5 Collection 选择预设，或展开“模型 JSON”粘贴自定义配置。
2. 设置 GPU 数量、单卡显存、Micro Batch 和序列长度。
3. 配置 TP、PP、EP、CP、ETP；DP 根据 GPU 总数自动计算。
4. 选择激活重计算和分布式优化器策略，并设置额外运行时预留。
5. 查看单卡峰值、显存组成、PP Rank 细分和不同策略下的快速对比。

默认模型配置位于 [`config.json`](./config.json)。自定义 JSON 可以将模型字段放在顶层，也可以放在 `text_config` 中。常用字段包括：

- 基础结构：`hidden_size`、`num_hidden_layers`、`num_attention_heads`、`num_key_value_heads`、`vocab_size`
- Dense MLP：`intermediate_size`
- MoE：`num_experts`、`num_experts_per_tok`、`moe_intermediate_size`、`shared_expert_intermediate_size`
- 混合注意力：`layer_types`、`head_dim` 及 `linear_*` 字段
- 其他：`tie_word_embeddings`、`mtp_num_hidden_layers`

## 估算口径

估算器以 BF16 全参数训练为默认口径：

- 模型权重：每参数 2 Bytes
- FP32 主梯度：每参数 4 Bytes
- FP32 主权重与 Adam 一、二阶状态：每参数 12 Bytes
- 分布式优化器：Dense 状态按 DP 切分，专家状态按专家数据并行域切分
- 激活值：根据层类型、每卡 Token 数、并行维度、Pipeline 在途数量和重计算策略近似估算
- 单卡峰值：取所有 PP Rank 中总占用最高的 Rank，并计入用户设置的运行时预留

拓扑关系为：

```text
Attention：DP = GPU 总数 / (TP × PP × CP)
MoE Parallel Folding：EDP = GPU 总数 / (ETP × EP × PP)
每卡 Tokens = Micro Batch × Sequence Length / CP
```

这里的 ETP 是 MoE 层的专家内张量并行；它与 EP、PP 共同决定专家数据并行域 EDP。若没有启用 Parallel Folding，且 ETP 与 TP 绑定，则上式退化为常见的 `EDP = DP / EP`（CP 为 1 时）。

内部按 `1024³ Bytes` 换算容量，界面统一显示为 GB。

## 注意事项

结果用于训练方案的前期规划，不等同于实际运行峰值。CUDA/NCCL 缓冲、算子实现、张量对齐、激活生命周期、通信重叠、框架版本和显存碎片都会影响真实占用。建议保留足够余量，并用目标集群上的短任务实测校准“单卡额外预留”。

当前参数量与显存估算仅覆盖文本模型部分，不包含视觉编码器。

## 主要文件

- [`app/Estimator.tsx`](./app/Estimator.tsx)：估算逻辑与交互界面
- [`app/qwen35-presets.ts`](./app/qwen35-presets.ts)：Qwen3.5 模型预设
- [`config.json`](./config.json)：默认模型配置
- [`app/globals.css`](./app/globals.css)：页面样式
