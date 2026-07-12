"use client";

import { useMemo, useState } from "react";
import { QWEN35_COLLECTION, qwen35Config } from "./qwen35-presets";

type JsonObject = Record<string, unknown>;

type Inputs = {
  totalGpus: number;
  gpuMemory: number;
  microBatch: number;
  sequenceLength: number;
  tp: number;
  pp: number;
  ep: number;
  cp: number;
  etp: number;
  recompute: "none" | "selective" | "full";
  distributedOptimizer: boolean;
  overhead: number;
};

type ModelShape = {
  name: string;
  hidden: number;
  layers: number;
  layerTypes: string[];
  fullAttentionLayers: number;
  linearAttentionLayers: number;
  heads: number;
  headDim: number;
  kvHeads: number;
  experts: number;
  topK: number;
  moeIntermediate: number;
  denseIntermediate: number;
  sharedIntermediate: number;
  vocab: number;
  tied: boolean;
  mtpLayers: number;
  linearKeyHeads: number;
  linearKeyDim: number;
  linearValueHeads: number;
  linearValueDim: number;
};

const DEFAULT_INPUTS: Inputs = {
  totalGpus: 64,
  gpuMemory: 80,
  microBatch: 1,
  sequenceLength: 4096,
  tp: 4,
  pp: 2,
  ep: 8,
  cp: 1,
  etp: 4,
  recompute: "selective",
  distributedOptimizer: true,
  overhead: 10,
};

const PARALLEL_GUIDE = [
  {
    id: "TP",
    name: "Tensor Parallel",
    summary: "切分层内矩阵",
    effect: "沿隐藏维或 FFN 维切分注意力与 MLP 权重，每层通过集合通信拼接结果。",
    advantage: "直接降低单卡权重、梯度和部分激活；单层放不下时最有效。",
    tradeoff: "每层都有 All-Reduce / All-Gather，跨节点扩展容易受带宽和时延限制。",
    fit: "大 hidden / 大 FFN 模型；优先放在 NVLink 或 NVSwitch 域内。",
    relation: "P_dense ∝ 1 / TP；部分 activation ∝ 1 / TP",
  },
  {
    id: "PP",
    name: "Pipeline Parallel",
    summary: "按连续层切 Stage",
    effect: "把 Transformer 层连续分配到多个 Stage，只在 Stage 边界传递激活与梯度。",
    advantage: "通信频率低于 TP，适合跨节点扩展；模型越深越容易均衡切层。",
    tradeoff: "存在 pipeline bubble、在途激活和首尾层不均衡，需要足够 micro-batch 隐藏空泡。",
    fit: "深层模型、多节点训练，或 TP 已达到单机高速互联上限时。",
    relation: "P_layer ≈ 1 / PP；activation 受 layers/rank 与 in-flight 共同影响",
  },
  {
    id: "DP",
    name: "Data Parallel",
    summary: "复制模型、切分 Batch",
    effect: "每个 DP Rank 持有同一份模型并处理不同样本，反向后同步梯度。",
    advantage: "吞吐扩展直接；配合 Distributed Optimizer 可按 DP 切分 Adam 状态。",
    tradeoff: "权重与梯度仍复制，梯度同步量大；全局 Batch 会随 DP 增长。",
    fit: "模型已能放入一个模型并行组，希望增加吞吐与数据规模时。",
    relation: "M_optim,dense ∝ 1 / DP；weights 不随普通 DP 降低",
  },
  {
    id: "CP",
    name: "Context Parallel",
    summary: "切分序列长度",
    effect: "把同一序列的 token 分到多个 Rank，协同完成长上下文注意力计算。",
    advantage: "直接降低每卡 token、注意力激活与长序列中间状态。",
    tradeoff: "引入注意力 P2P / All-to-All 通信，依赖内核支持并增加调度复杂度。",
    fit: "长上下文训练中 activation 或 attention workspace 成为主要瓶颈时。",
    relation: "Tokens/GPU = MBS × SeqLen / CP",
  },
  {
    id: "EP",
    name: "Expert Parallel",
    summary: "跨 Rank 分布专家",
    effect: "把不同 MoE 专家放到不同 Rank，token 经路由后通过 All-to-All 发往目标专家。",
    advantage: "显著降低单卡独占专家权重，并可随专家数量扩展总容量。",
    tradeoff: "token dispatch 通信重且易受负载不均影响；专家数据并行域缩小为 DP / EP。",
    fit: "专家数量多、路由较均衡且集群具备高带宽 All-to-All 的 MoE 模型。",
    relation: "P_expert ∝ 1 / EP；EDP = DP / EP",
  },
  {
    id: "ETP",
    name: "Expert Tensor Parallel",
    summary: "切分单个专家 MLP",
    effect: "在一个专家内部继续切分 gate / up / down 投影，和 EP 的专家分布正交组合。",
    advantage: "当单个专家仍过大时继续降低专家权重与中间激活。",
    tradeoff: "增加专家内部集合通信，EP × ETP 拓扑和通信组配置更复杂。",
    fit: "专家 FFN 本身较大；通常让 ETP 留在单机高速互联域内。",
    relation: "P_expert, A_moe ∝ 1 / (EP × ETP)",
  },
  {
    id: "SP",
    name: "Sequence Parallel",
    summary: "随 TP 切非线性激活",
    effect: "沿序列维切分 LayerNorm、Dropout 等未被 TP 切分的激活，通常与 TP 联动。",
    advantage: "进一步降低每卡 activation，并减少 TP 区域中的重复激活存储。",
    tradeoff: "在层边界增加 Reduce-Scatter / All-Gather；不能独立替代 TP 或 CP。",
    fit: "TP > 1 且激活占比较高的 Megatron 训练；当前估算将其并入 TP 近似。",
    relation: "A_norm/dropout ≈ 1 / TP（启用 SP 时）",
  },
] as const;

const GiB = 1024 ** 3;
const n = (value: unknown, fallback = 0) =>
  typeof value === "number" && Number.isFinite(value) ? value : fallback;

function textConfigOf(config: JsonObject): JsonObject {
  const nested = config.text_config;
  return nested && typeof nested === "object" ? (nested as JsonObject) : config;
}

function shapeFromConfig(config: JsonObject): ModelShape {
  const c = textConfigOf(config);
  const configuredLayerTypes = Array.isArray(c.layer_types)
    ? c.layer_types.map(String)
    : Array(n(c.num_hidden_layers, 1)).fill("full_attention");
  const layers = n(c.num_hidden_layers, configuredLayerTypes.length);
  const layerTypes = Array.from(
    { length: layers },
    (_, index) => configuredLayerTypes[index] ?? "full_attention",
  );
  const architecture = Array.isArray(config.architectures)
    ? String(config.architectures[0] ?? "Megatron 模型")
    : String(c.model_type ?? "Megatron 模型");

  return {
    name: architecture.replace(/ForConditionalGeneration|ForCausalLM/g, ""),
    hidden: n(c.hidden_size),
    layers,
    layerTypes,
    fullAttentionLayers: layerTypes.filter((item) => item === "full_attention").length,
    linearAttentionLayers: layerTypes.filter((item) => item !== "full_attention").length,
    heads: n(c.num_attention_heads),
    headDim: n(c.head_dim, n(c.hidden_size) / Math.max(n(c.num_attention_heads, 1), 1)),
    kvHeads: n(c.num_key_value_heads, n(c.num_attention_heads)),
    experts: n(c.num_experts),
    topK: n(c.num_experts_per_tok, 1),
    moeIntermediate: n(c.moe_intermediate_size, n(c.intermediate_size)),
    denseIntermediate: n(c.intermediate_size, n(c.moe_intermediate_size)),
    sharedIntermediate: n(c.shared_expert_intermediate_size),
    vocab: n(c.vocab_size),
    tied: Boolean(config.tie_word_embeddings ?? c.tie_word_embeddings),
    mtpLayers: n(c.mtp_num_hidden_layers),
    linearKeyHeads: n(c.linear_num_key_heads),
    linearKeyDim: n(c.linear_key_head_dim),
    linearValueHeads: n(c.linear_num_value_heads),
    linearValueDim: n(c.linear_value_head_dim),
  };
}

function calculate(shape: ModelShape, x: Inputs) {
  const fullQ = shape.hidden * shape.heads * shape.headDim;
  const fullKV = 2 * shape.hidden * shape.kvHeads * shape.headDim;
  const fullO = shape.hidden * shape.heads * shape.headDim;
  const fullAttention = fullQ + fullKV + fullO;

  const linearQK = 2 * shape.hidden * shape.linearKeyHeads * shape.linearKeyDim;
  const linearV = shape.hidden * shape.linearValueHeads * shape.linearValueDim;
  const linearO = shape.hidden * shape.linearValueHeads * shape.linearValueDim;
  const linearGates = shape.hidden * shape.linearValueHeads * 2;
  const linearAttention = linearQK + linearV + linearO + linearGates;

  const expertPerLayer = shape.experts * 3 * shape.hidden * shape.moeIntermediate;
  const denseMlpPerLayer = shape.experts > 0 ? 0 : 3 * shape.hidden * shape.denseIntermediate;
  const sharedPerLayer = 3 * shape.hidden * shape.sharedIntermediate;
  const routerPerLayer = shape.hidden * shape.experts;
  const normPerLayer = shape.hidden * 2;
  const mtpMultiplier = shape.layers > 0 ? 1 + shape.mtpLayers / shape.layers : 1;

  const expertParams = expertPerLayer * shape.layers * mtpMultiplier;
  const fullAttentionParams = shape.fullAttentionLayers * fullAttention * mtpMultiplier;
  const linearAttentionParams = shape.linearAttentionLayers * linearAttention * mtpMultiplier;
  const denseMlpParams = denseMlpPerLayer * shape.layers * mtpMultiplier;
  const denseMoEParams =
    shape.layers * (sharedPerLayer + routerPerLayer + normPerLayer) * mtpMultiplier;
  const denseLayerParams = fullAttentionParams + linearAttentionParams + denseMlpParams + denseMoEParams;
  const embedding = shape.vocab * shape.hidden * (shape.tied ? 1 : 2);
  const denseParams = denseLayerParams + embedding;
  const totalParams = denseParams + expertParams;
  const activeExpertPerLayer =
    (shape.topK * 3 * shape.hidden * shape.moeIntermediate) + sharedPerLayer;
  const activeParams =
    embedding / (shape.tied ? 1 : 2) +
    shape.fullAttentionLayers * fullAttention +
    shape.linearAttentionLayers * linearAttention +
    shape.layers * (activeExpertPerLayer + routerPerLayer + normPerLayer);

  const pp = Math.max(Math.floor(x.pp), 1);
  const topology = x.tp * pp * x.cp;
  const dp = x.totalGpus / Math.max(topology, 1);
  const validTopology = Number.isInteger(dp) && dp >= 1 && Number.isInteger(dp / x.ep);
  const expertDp = Math.max(dp / Math.max(x.ep, 1), 1);

  const tokens = x.microBatch * x.sequenceLength / Math.max(x.cp, 1);
  const hiddenBytes = tokens * shape.hidden * 2;
  const fullLayerActivation = hiddenBytes * (4 + 12 / Math.max(x.tp, 1));
  const linearWidth = shape.linearValueHeads * shape.linearValueDim;
  const linearLayerActivation =
    hiddenBytes * (3.5 + 9 / Math.max(x.tp, 1)) +
    tokens * linearWidth * 2 / Math.max(x.tp, 1);
  const moeActivation = shape.experts > 0
    ? tokens * shape.topK * shape.moeIntermediate * 2 / Math.max(x.etp * x.ep, 1)
    : tokens * shape.denseIntermediate * 2 / Math.max(x.tp, 1);
  const recomputeFactor = x.recompute === "full" ? 0.2 : x.recompute === "selective" ? 0.56 : 1;
  const denseCorePerLayer = denseMlpPerLayer + sharedPerLayer + routerPerLayer + normPerLayer;
  const averageDenseLayer = shape.layers > 0
    ? (shape.fullAttentionLayers * fullAttention +
      shape.linearAttentionLayers * linearAttention +
      shape.layers * denseCorePerLayer) / shape.layers
    : 0;
  const averageActivationLayer = shape.layers > 0
    ? (shape.fullAttentionLayers * fullLayerActivation +
      shape.linearAttentionLayers * linearLayerActivation +
      shape.layers * moeActivation) / shape.layers
    : 0;
  const embeddingMatrix = shape.vocab * shape.hidden;

  // Each PP rank owns a contiguous layer range. The last rank also owns MTP layers.
  const ppRanks = Array.from({ length: pp }, (_, rank) => {
    const layerStart = Math.floor(rank * shape.layers / pp);
    const layerEnd = Math.floor((rank + 1) * shape.layers / pp);
    const rankLayerTypes = shape.layerTypes.slice(layerStart, layerEnd);
    const fullLayers = rankLayerTypes.filter((type) => type === "full_attention").length;
    const linearLayers = rankLayerTypes.length - fullLayers;
    const layerCount = rankLayerTypes.length;
    const mtpLayers = rank === pp - 1 ? shape.mtpLayers : 0;
    const roles: string[] = [];

    let denseGlobal =
      fullLayers * fullAttention +
      linearLayers * linearAttention +
      layerCount * denseCorePerLayer +
      mtpLayers * averageDenseLayer;
    const expertGlobal = (layerCount + mtpLayers) * expertPerLayer;

    if (rank === 0) {
      denseGlobal += embeddingMatrix;
      roles.push("Embedding");
    }
    if (rank === pp - 1 && (!shape.tied || pp > 1)) {
      denseGlobal += embeddingMatrix;
      roles.push(shape.tied ? "Tied LM Head" : "LM Head");
    }
    if (mtpLayers > 0) roles.push(`MTP × ${mtpLayers}`);

    const localDense = denseGlobal / Math.max(x.tp, 1);
    const localExpert = expertGlobal / Math.max(x.etp * x.ep, 1);
    const localParams = localDense + localExpert;
    const modelWeights = localParams * 2 / GiB;
    const gradients = localParams * 4 / GiB;
    const optimizerDistributed =
      (localDense * 12 / Math.max(dp, 1) + localExpert * 12 / expertDp) / GiB;
    const optimizerReplicated = localParams * 12 / GiB;
    const optimizer = x.distributedOptimizer ? optimizerDistributed : optimizerReplicated;
    const modelState = modelWeights + gradients + optimizer;
    const pipelineInflight = Math.max(pp - rank, 1);
    const savedActivation =
      fullLayers * fullLayerActivation +
      linearLayers * linearLayerActivation +
      layerCount * moeActivation +
      mtpLayers * averageActivationLayer;
    const activationNoRecompute = savedActivation * pipelineInflight / GiB;
    const activationSelective = activationNoRecompute * 0.56;
    const activationFull = activationNoRecompute * 0.2;
    const activation = activationNoRecompute * recomputeFactor;

    return {
      rank,
      layerStart,
      layerEnd,
      layerCount,
      fullLayers,
      linearLayers,
      mtpLayers,
      roles,
      denseGlobal,
      expertGlobal,
      localDense,
      localExpert,
      localParams,
      modelWeights,
      gradients,
      optimizer,
      optimizerDistributed,
      optimizerReplicated,
      modelState,
      pipelineInflight,
      activation,
      activationNoRecompute,
      activationSelective,
      activationFull,
      total: modelState + activation + x.overhead,
      totalNoRecompute: modelState + activationNoRecompute + x.overhead,
      totalSelective: modelState + activationSelective + x.overhead,
      totalFull: modelState + activationFull + x.overhead,
      totalDistributed: modelWeights + gradients + optimizerDistributed + activation + x.overhead,
      totalReplicated: modelWeights + gradients + optimizerReplicated + activation + x.overhead,
    };
  });

  const peakRank = ppRanks.reduce((peak, rank) => rank.total > peak.total ? rank : peak);
  const peakNoRecompute = ppRanks.reduce((peak, rank) => rank.totalNoRecompute > peak.totalNoRecompute ? rank : peak);
  const peakSelective = ppRanks.reduce((peak, rank) => rank.totalSelective > peak.totalSelective ? rank : peak);
  const peakFull = ppRanks.reduce((peak, rank) => rank.totalFull > peak.totalFull ? rank : peak);
  const peakDistributed = ppRanks.reduce((peak, rank) => rank.totalDistributed > peak.totalDistributed ? rank : peak);
  const peakReplicated = ppRanks.reduce((peak, rank) => rank.totalReplicated > peak.totalReplicated ? rank : peak);
  const capacity = x.gpuMemory;

  return {
    totalParams,
    activeParams,
    denseParams,
    expertParams,
    fullAttentionParams,
    linearAttentionParams,
    denseMoEParams,
    denseMlpParams,
    embedding,
    expertPerLayer,
    mtpMultiplier,
    localDense: peakRank.localDense,
    localExpert: peakRank.localExpert,
    localParams: peakRank.localParams,
    modelWeights: peakRank.modelWeights,
    gradients: peakRank.gradients,
    optimizer: peakRank.optimizer,
    optimizerDistributed: peakDistributed.optimizerDistributed,
    optimizerReplicated: peakReplicated.optimizerReplicated,
    modelState: peakRank.modelState,
    activation: peakRank.activation,
    activationNoRecompute: peakNoRecompute.activationNoRecompute,
    activationSelective: peakSelective.activationSelective,
    activationFull: peakFull.activationFull,
    recomputeFactor,
    tokens,
    fullLayerActivation,
    linearLayerActivation,
    moeActivation,
    pipelineInflight: peakRank.pipelineInflight,
    overhead: x.overhead,
    total: peakRank.total,
    totalNoRecompute: peakNoRecompute.totalNoRecompute,
    totalSelective: peakSelective.totalSelective,
    totalFull: peakFull.totalFull,
    totalDistributed: peakDistributed.totalDistributed,
    totalReplicated: peakReplicated.totalReplicated,
    capacity,
    headroom: capacity - peakRank.total,
    usage: peakRank.total / Math.max(capacity, 1) * 100,
    dp,
    expertDp,
    validTopology,
    ppRanks,
    peakRank,
  };
}

const compact = (value: number) => {
  const billions = value / 1e9;
  return billions >= 100 ? billions.toFixed(0) : billions >= 10 ? billions.toFixed(1) : billions.toFixed(2);
};
const gb = (value: number) => `${value.toFixed(1)} GB`;

function formatLayerRanges(indices: number[]) {
  if (indices.length === 0) return "无";
  const ranges: string[] = [];
  let start = indices[0];
  let end = indices[0];

  for (const index of indices.slice(1)) {
    if (index === end + 1) {
      end = index;
      continue;
    }
    ranges.push(start === end ? `L${start}` : `L${start}–L${end}`);
    start = index;
    end = index;
  }
  ranges.push(start === end ? `L${start}` : `L${start}–L${end}`);
  return ranges.join("、");
}

function NumberField({
  label,
  value,
  min = 1,
  onChange,
  suffix,
}: {
  label: string;
  value: number;
  min?: number;
  onChange: (value: number) => void;
  suffix?: string;
}) {
  return (
    <label className="field">
      <span>{label}</span>
      <span className="input-wrap">
        <input
          type="number"
          min={min}
          value={value}
          onChange={(event) => onChange(Math.max(min, Number(event.target.value) || min))}
        />
        {suffix ? <em>{suffix}</em> : null}
      </span>
    </label>
  );
}

export function Estimator({ initialConfig }: { initialConfig: JsonObject }) {
  const [inputs, setInputs] = useState(DEFAULT_INPUTS);
  const [configText, setConfigText] = useState(() => JSON.stringify(initialConfig, null, 2));
  const [config, setConfig] = useState<JsonObject>(initialConfig);
  const [configError, setConfigError] = useState("");
  const [showConfig, setShowConfig] = useState(false);
  const [selectedPresetId, setSelectedPresetId] = useState("Qwen3.5-122B-A10B");

  const shape = useMemo(() => shapeFromConfig(config), [config]);
  const result = useMemo(() => calculate(shape, inputs), [shape, inputs]);
  const selectedPreset = QWEN35_COLLECTION.find((preset) => preset.id === selectedPresetId);
  const fullLayerSources = formatLayerRanges(
    shape.layerTypes.map((type, index) => type === "full_attention" ? index : -1).filter((index) => index >= 0),
  );
  const linearLayerSources = formatLayerRanges(
    shape.layerTypes.map((type, index) => type !== "full_attention" ? index : -1).filter((index) => index >= 0),
  );
  const allLayerSources = shape.layers > 0 ? `L0–L${shape.layers - 1}` : "无";
  const update = <K extends keyof Inputs>(key: K, value: Inputs[K]) =>
    setInputs((current) => ({ ...current, [key]: value }));

  const applyConfig = () => {
    try {
      const parsed = JSON.parse(configText) as JsonObject;
      setConfig(parsed);
      setConfigError("");
      setSelectedPresetId("");
    } catch (error) {
      setConfigError(error instanceof Error ? error.message : "JSON 格式不正确");
    }
  };

  const loadPreset = (id: string) => {
    const preset = QWEN35_COLLECTION.find((item) => item.id === id);
    if (!preset) return;
    const nextConfig = qwen35Config(preset);
    setConfig(nextConfig);
    setConfigText(JSON.stringify(nextConfig, null, 2));
    setConfigError("");
    setSelectedPresetId(id);
  };

  const status = !result.validTopology
    ? { label: "并行拓扑无效", tone: "warning" }
    : result.headroom >= 8
      ? { label: "可运行", tone: "good" }
      : result.headroom >= 0
        ? { label: "临界", tone: "warning" }
        : { label: "显存不足", tone: "danger" };

  return (
    <main>
      <header className="topbar">
        <a className="brand" href="#top" aria-label="返回顶部">
          <span className="brand-mark">M</span>
          <span>MEGATRON LAB</span>
        </a>
        <span className="top-note">SFT MEMORY PLANNER · BF16</span>
      </header>

      <section className="hero" id="top">
        <div>
          <p className="eyebrow">FULL-PARAMETER SFT · MEMORY ESTIMATE</p>
          <h1>看清每一GB</h1>
        </div>
        <div className="hero-copy">
          <p>
            基于 Megatron-LM 的混合精度训练口径，估算参数、梯度、优化器状态、激活值与运行时预留。
          </p>
          <div className="model-badge">
            <span>已载入配置</span>
            <strong>{selectedPreset?.id ?? shape.name}</strong>
            <small>{compact(result.totalParams)}B 总参数 · {compact(result.activeParams)}B 激活参数</small>
          </div>
        </div>
      </section>

      <section className="workspace" aria-label="显存估算工作区">
        <aside className="controls">
          <div className="section-heading">
            <span>01</span>
            <div><h2>训练配置</h2><p>调整并行策略与批次设置</p></div>
          </div>

          <div className="control-group preset-collection">
            <h3>Qwen3.5 Collection</h3>
            <label className="field full">
              <span>官方正式模型</span>
              <span className="input-wrap select-wrap">
                <select value={selectedPresetId} onChange={(event) => loadPreset(event.target.value)}>
                  {(["Dense", "MoE"] as const).map((family) => (
                    <optgroup key={family} label={family}>
                      {QWEN35_COLLECTION.filter((preset) => preset.family === family).map((preset) => (
                        <option key={preset.id} value={preset.id}>{preset.id} · {preset.nominalParams}</option>
                      ))}
                    </optgroup>
                  ))}
                  {!selectedPresetId ? <option value="">自定义 JSON</option> : null}
                </select>
              </span>
            </label>
            {selectedPreset ? (
              <div className="preset-summary">
                <span className={selectedPreset.family === "MoE" ? "moe" : "dense"}>{selectedPreset.family}</span>
                <strong>标称 {selectedPreset.nominalParams}</strong>
                <a href={selectedPreset.sourceUrl} target="_blank" rel="noreferrer">Hugging Face 配置</a>
              </div>
            ) : null}
            <p className="preset-note">基于 Qwen 官方 text_config；当前估算不包含视觉编码器。</p>
          </div>

          <button className="config-toggle" type="button" onClick={() => setShowConfig((value) => !value)} aria-expanded={showConfig}>
            <span>模型 JSON</span><span>{showConfig ? "收起 −" : "编辑 +"}</span>
          </button>
          {showConfig ? (
            <div className="json-panel">
              <textarea aria-label="模型 JSON 配置" value={configText} onChange={(e) => setConfigText(e.target.value)} spellCheck={false} />
              {configError ? <p className="json-error">{configError}</p> : null}
              <button type="button" onClick={applyConfig}>应用配置</button>
            </div>
          ) : null}

          <div className="control-group">
            <h3>硬件</h3>
            <div className="field-grid two">
              <NumberField label="GPU 数量" value={inputs.totalGpus} onChange={(v) => update("totalGpus", v)} />
              <label className="field">
                <span>单卡显存</span>
                <span className="input-wrap select-wrap">
                  <select value={inputs.gpuMemory} onChange={(e) => update("gpuMemory", Number(e.target.value))}>
                    <option value={80}>80 GB · H100</option>
                    <option value={96}>96 GB · H20</option>
                    <option value={141}>141 GB · H200</option>
                    <option value={192}>192 GB · B200</option>
                  </select>
                </span>
              </label>
            </div>
          </div>

          <div className="control-group">
            <h3>批次与序列</h3>
            <div className="field-grid two">
              <NumberField label="Micro Batch" value={inputs.microBatch} onChange={(v) => update("microBatch", v)} />
              <NumberField label="序列长度" value={inputs.sequenceLength} onChange={(v) => update("sequenceLength", v)} />
            </div>
          </div>

          <div className="control-group">
            <h3>并行拓扑</h3>
            <div className="parallel-grid">
              {(["tp", "pp", "ep", "cp", "etp"] as const).map((key) => (
                <NumberField key={key} label={key.toUpperCase()} value={inputs[key]} onChange={(v) => update(key, v)} />
              ))}
              <div className="derived-field"><span>DP</span><strong>{Number.isInteger(result.dp) ? result.dp : result.dp.toFixed(2)}</strong></div>
            </div>
            <p className={`topology-note ${result.validTopology ? "" : "invalid"}`}>
              {result.validTopology
                ? `${inputs.tp} TP × ${inputs.pp} PP × ${inputs.cp} CP × ${result.dp} DP = ${inputs.totalGpus} GPUs；EP ${inputs.ep} 在数据并行域内分组。`
                : "GPU 总数需能被 TP × PP × CP 整除，且 DP 需能被 EP 整除。"}
            </p>
          </div>

          <div className="control-group">
            <h3>内存策略</h3>
            <label className="field full">
              <span>激活重计算</span>
              <span className="segmented">
                {([[
                  "none", "关闭"
                ], ["selective", "选择性"], ["full", "全量"]] as const).map(([value, label]) => (
                  <button key={value} type="button" className={inputs.recompute === value ? "active" : ""} onClick={() => update("recompute", value)}>{label}</button>
                ))}
              </span>
            </label>
            <label className="check-row">
              <input type="checkbox" checked={inputs.distributedOptimizer} onChange={(e) => update("distributedOptimizer", e.target.checked)} />
              <span><strong>分布式优化器</strong><small>将 FP32 主权重与 Adam 状态按数据并行域切分</small></span>
            </label>
            <NumberField label="单卡额外预留" value={inputs.overhead} min={0} suffix="GB" onChange={(v) => update("overhead", v)} />
          </div>

          <section className="activation-guide" aria-labelledby="activation-guide-title">
            <div className="activation-guide-heading">
              <div><span>MEMORY MODEL</span><h3 id="activation-guide-title">激活值分层近似</h3></div>
              <small>单层 · 单 micro-batch</small>
            </div>
            <div className="activation-guide-list">
              <details className="activation-topic">
                <summary>
                  <span><strong>Full Attention</strong><small>A<sub>full/layer</sub></small></span>
                  <b>{gb(result.fullLayerActivation / GiB)}</b>
                </summary>
                <div className="activation-topic-body">
                  <code>A<sub>full/layer</sub> ≈ Tokens × H × 2 × (4 + 12 / TP)</code>
                  <dl>
                    <div><dt>来源层</dt><dd><code>layer_types = full_attention</code>：{fullLayerSources}</dd></div>
                    <div><dt>Tokens</dt><dd>MBS × SeqLen / CP = {Math.round(result.tokens).toLocaleString("en-US")}</dd></div>
                    <div><dt>H × 2</dt><dd>hidden elements × BF16 2 Bytes；当前 H = {shape.hidden.toLocaleString("en-US")}</dd></div>
                    <div><dt>4</dt><dd>约 4 份不随 TP 完全切分的 hidden-size 激活，如层输入、残差与归一化结果。</dd></div>
                    <div><dt>12 / TP</dt><dd>约 12 份可沿 TP 切分的注意力投影及反向所需等效激活。</dd></div>
                    <div><dt>假设</dt><dd>使用 Flash Attention 类内核，不保存完整的 SeqLen × SeqLen 注意力矩阵。</dd></div>
                  </dl>
                </div>
              </details>
              <details className="activation-topic">
                <summary>
                  <span><strong>Linear Attention</strong><small>A<sub>linear/layer</sub></small></span>
                  <b>{gb(result.linearLayerActivation / GiB)}</b>
                </summary>
                <div className="activation-topic-body">
                  <code>A<sub>linear/layer</sub> ≈ Tokens × H × 2 × (3.5 + 9 / TP) + Tokens × V × 2 / TP</code>
                  <dl>
                    <div><dt>来源层</dt><dd><code>layer_types = linear_attention</code>：{linearLayerSources}</dd></div>
                    <div><dt>3.5</dt><dd>残差、归一化、门控等不完全随 TP 切分的 hidden-size 等效激活。</dd></div>
                    <div><dt>9 / TP</dt><dd>可沿 TP 切分的线性注意力投影与反向保存项。</dd></div>
                    <div><dt>V</dt><dd>value heads × value dim = {shape.linearValueHeads} × {shape.linearValueDim} = {(shape.linearValueHeads * shape.linearValueDim).toLocaleString("en-US")}</dd></div>
                    <div><dt>特点</dt><dd>不显式保存二次复杂度的注意力矩阵，但需保留线性状态与门控中间量。</dd></div>
                  </dl>
                </div>
              </details>
              <details className="activation-topic">
                <summary>
                  <span><strong>{shape.experts > 0 ? "MoE Experts" : "Dense MLP"}</strong><small>A<sub>{shape.experts > 0 ? "moe" : "mlp"}/layer</sub></small></span>
                  <b>{gb(result.moeActivation / GiB)}</b>
                </summary>
                <div className="activation-topic-body">
                  <code>
                    {shape.experts > 0
                      ? <>A<sub>moe/layer</sub> ≈ Tokens × TopK × I<sub>moe</sub> × 2 / (ETP × EP)</>
                      : <>A<sub>mlp/layer</sub> ≈ Tokens × I<sub>FFN</sub> × 2 / TP</>}
                  </code>
                  {shape.experts > 0 ? (
                    <dl>
                      <div><dt>来源层</dt><dd>{allLayerSources} 每个 Transformer block 的 routed experts。</dd></div>
                      <div><dt>TopK</dt><dd>每 token 路由到 {shape.topK} 个专家，专家中间激活相应复制 {shape.topK} 份。</dd></div>
                      <div><dt>I_moe × 2</dt><dd>专家 intermediate width {shape.moeIntermediate.toLocaleString("en-US")} × BF16 2 Bytes。</dd></div>
                      <div><dt>EP</dt><dd>假设路由均衡，每个 EP Rank 接收约 1 / {inputs.ep} 的 routed tokens。</dd></div>
                      <div><dt>ETP</dt><dd>单个专家 intermediate 维再切为 {inputs.etp} 份。</dd></div>
                      <div><dt>未计入</dt><dd>Shared Expert、Router logits、dispatch metadata、负载偏斜和 All-to-All buffer。</dd></div>
                    </dl>
                  ) : (
                    <dl>
                      <div><dt>来源层</dt><dd>{allLayerSources} 每个 Transformer block 的 Dense gated MLP。</dd></div>
                      <div><dt>I_FFN × 2</dt><dd>Dense MLP intermediate width {shape.denseIntermediate.toLocaleString("en-US")} × BF16 2 Bytes。</dd></div>
                      <div><dt>TP</dt><dd>中间维按 TP={inputs.tp} 切分，不经过 EP 或 ETP。</dd></div>
                      <div><dt>未计入</dt><dd>融合 SwiGLU 内核临时量、通信 workspace 与实现相关的额外保存 Tensor。</dd></div>
                    </dl>
                  )}
                </div>
              </details>
              <details className="activation-topic">
                <summary>
                  <span><strong>PP Rank 汇总</strong><small>A<sub>rank</sub></small></span>
                  <b>Peak PP {result.peakRank.rank}</b>
                </summary>
                <div className="activation-topic-body">
                  <code>A<sub>rank</sub> ≈ Σ(L<sub>type,rank</sub> × A<sub>type/layer</sub>) × N<sub>flight,rank</sub> × r</code>
                  <dl>
                    <div><dt>来源层</dt><dd>Peak PP {result.peakRank.rank}：{result.peakRank.layerCount > 0 ? `L${result.peakRank.layerStart}–L${result.peakRank.layerEnd - 1}` : "无基础层"}{result.peakRank.mtpLayers > 0 ? `，另含 MTP × ${result.peakRank.mtpLayers}` : ""}。</dd></div>
                    <div><dt>层数</dt><dd>分别累计该 Rank 实际持有的 Full、Linear 与 MLP / MoE 层。</dd></div>
                    <div><dt>N_flight</dt><dd>近似为 PP − rank；当前 Peak Rank 为 {result.pipelineInflight} 个在途 micro-batch。</dd></div>
                    <div><dt>r</dt><dd>不重计算 1.00、选择性重计算 0.56、全量重计算 0.20；当前 {result.recomputeFactor.toFixed(2)}。</dd></div>
                    <div><dt>结果</dt><dd>Peak PP {result.peakRank.rank} 当前激活约 {gb(result.activation)}。</dd></div>
                  </dl>
                </div>
              </details>
            </div>
          </section>

          <section className="parallel-guide" aria-labelledby="parallel-guide-title">
            <div className="parallel-guide-heading">
              <div><span>REFERENCE</span><h3 id="parallel-guide-title">并行策略作用表</h3></div>
              <small>固定参考</small>
            </div>
            <div className="parallel-guide-list">
              {PARALLEL_GUIDE.map((method) => (
                <details key={method.id} className="parallel-method" open={method.id === "TP"}>
                  <summary>
                    <code>{method.id}</code>
                    <span><strong>{method.name}</strong><small>{method.summary}</small></span>
                  </summary>
                  <div className="parallel-method-body">
                    <dl>
                      <div><dt>作用</dt><dd>{method.effect}</dd></div>
                      <div><dt>优势</dt><dd>{method.advantage}</dd></div>
                      <div><dt>代价</dt><dd>{method.tradeoff}</dd></div>
                      <div><dt>适用</dt><dd>{method.fit}</dd></div>
                    </dl>
                    <code>{method.relation}</code>
                  </div>
                </details>
              ))}
            </div>
          </section>

        </aside>

        <section className="results" aria-live="polite">
          <div className="section-heading light">
            <span>02</span>
            <div><h2>单卡峰值估算</h2><p>以最繁忙 PP Rank 为准</p></div>
          </div>

          <div className="total-card">
            <div className="total-topline">
              <span className={`status ${status.tone}`}><i />{status.label}</span>
              <span>Peak PP {result.peakRank.rank} · {inputs.totalGpus} × {inputs.gpuMemory} GB</span>
            </div>
            <div className="total-number">
              <strong>{result.total.toFixed(1)}</strong><span>GB / GPU</span>
            </div>
            <div className="meter" aria-label={`显存占用 ${result.usage.toFixed(0)}%`}>
              <span style={{ width: `${Math.min(result.usage, 100)}%` }} />
              <i style={{ left: "90%" }} />
            </div>
            <div className="meter-labels"><span>0 GB</span><span>安全线 {Math.round(inputs.gpuMemory * 0.9)} GB</span><span>{inputs.gpuMemory} GB</span></div>
            <p className={`headroom ${result.headroom >= 0 ? "positive" : "negative"}`}>
              {result.headroom >= 0
                ? `预计剩余 ${result.headroom.toFixed(1)} GB，可用于通信缓冲或内核临时空间。`
                : `预计超出 ${Math.abs(result.headroom).toFixed(1)} GB，需要增加并行度或开启更强重计算。`}
            </p>
            <div className="compact-advice">
              <b>建议</b>
              <span>
                {!result.validTopology
                  ? "先让 GPU 数量满足 TP × PP × CP 与 EP 的整除关系。"
                  : result.activation > result.modelState * 0.45
                    ? "激活占比较高，优先增加 CP 或使用全量重计算。"
                    : result.headroom < 8
                      ? "余量偏紧，优先提高 TP / PP / EP 或增加单卡显存。"
                      : "当前余量健康，建议用实测峰值校准运行时预留。"}
              </span>
              {!inputs.distributedOptimizer ? <span>开启分布式优化器可显著降低 Adam 状态。</span> : null}
            </div>
          </div>

          <div className="breakdown">
            <article className="weights"><span>模型权重</span><strong>{gb(result.modelWeights)}</strong><small>BF16 · 2 B / param</small><div style={{ width: `${result.total ? result.modelWeights / result.total * 100 : 0}%` }} /></article>
            <article className="gradients"><span>梯度</span><strong>{gb(result.gradients)}</strong><small>FP32 main grad · 4 B / param</small><div style={{ width: `${result.total ? result.gradients / result.total * 100 : 0}%` }} /></article>
            <article className="optimizer"><span>优化器状态</span><strong>{gb(result.optimizer)}</strong><small>{inputs.distributedOptimizer ? "Distributed Adam" : "Replicated Adam"} · 12 B</small><div style={{ width: `${result.total ? result.optimizer / result.total * 100 : 0}%` }} /></article>
            <article className="activations"><span>激活值</span><strong>{gb(result.activation)}</strong><small>{inputs.recompute === "full" ? "全量重计算" : inputs.recompute === "selective" ? "选择性重计算" : "不重计算"}</small><div style={{ width: `${result.total ? result.activation / result.total * 100 : 0}%` }} /></article>
            <article className="runtime"><span>运行时预留</span><strong>{gb(result.overhead)}</strong><small>NCCL · kernel · 碎片</small><div style={{ width: `${result.total ? result.overhead / result.total * 100 : 0}%` }} /></article>
          </div>

          <div className="composition" aria-label="显存组成比例">
            <span className="weights" style={{ width: `${result.modelWeights / result.total * 100}%` }} title={`模型权重 ${gb(result.modelWeights)}`} />
            <span className="gradients" style={{ width: `${result.gradients / result.total * 100}%` }} title={`梯度 ${gb(result.gradients)}`} />
            <span className="optimizer" style={{ width: `${result.optimizer / result.total * 100}%` }} title={`优化器 ${gb(result.optimizer)}`} />
            <span className="activations" style={{ width: `${result.activation / result.total * 100}%` }} title={`激活值 ${gb(result.activation)}`} />
            <span className="runtime" style={{ width: `${result.overhead / result.total * 100}%` }} title={`运行时预留 ${gb(result.overhead)}`} />
          </div>

          <div className="details-grid">
            <article>
              <div className="detail-title"><span>03</span><h3>模型剖面</h3></div>
              <dl>
                <div><dt>总参数</dt><dd>{compact(result.totalParams)}B</dd></div>
                <div><dt>单 token 激活</dt><dd>{compact(result.activeParams)}B</dd></div>
                <div><dt>峰值 Rank 本地参数</dt><dd>{compact(result.localParams)}B</dd></div>
                <div><dt>{shape.experts > 0 ? "MoE 专家参数" : "Dense MLP 参数"}</dt><dd>{compact(shape.experts > 0 ? result.expertParams : result.denseMlpParams)}B</dd></div>
                <div><dt>注意力层</dt><dd>{shape.linearAttentionLayers} Linear + {shape.fullAttentionLayers} Full</dd></div>
                <div><dt>{shape.experts > 0 ? "专家路由" : "前馈层"}</dt><dd>{shape.experts > 0 ? `${shape.topK} / ${shape.experts}` : "Dense gated MLP"}</dd></div>
              </dl>
              <details className="profile-disclosure">
                <summary><span>关键网络维度</span><strong>H = {shape.hidden.toLocaleString("en-US")}</strong></summary>
                <div className="dimension-list">
                  <div><span>Hidden state</span><code>[B, S, H] = [B, S, {shape.hidden.toLocaleString("en-US")}]</code></div>
                  <div><span>网络深度</span><code>{shape.layers} blocks + {shape.mtpLayers} MTP</code></div>
                  <div><span>Full Q / O</span><code>H ↔ {shape.heads} heads × {shape.headDim} dim</code></div>
                  <div><span>Full K / V</span><code>H → {shape.kvHeads} KV heads × {shape.headDim} dim</code></div>
                  <div><span>Linear Q / K</span><code>H → {shape.linearKeyHeads} heads × {shape.linearKeyDim} dim</code></div>
                  <div><span>Linear V / O</span><code>H ↔ {shape.linearValueHeads} heads × {shape.linearValueDim} dim</code></div>
                  <div><span>{shape.experts > 0 ? "Expert MLP" : "Dense MLP"}</span><code>H → {shape.experts > 0 ? shape.moeIntermediate : shape.denseIntermediate} → H · 3 matrices</code></div>
                  {shape.experts > 0 ? <div><span>MoE routing</span><code>{shape.experts} experts · TopK {shape.topK} · shared I = {shape.sharedIntermediate}</code></div> : null}
                  <div><span>Embedding</span><code>V × H = {shape.vocab.toLocaleString("en-US")} × {shape.hidden.toLocaleString("en-US")} · {shape.tied ? "tied" : "untied"}</code></div>
                </div>
              </details>
            </article>
            <article className="rank-panel">
              <div className="detail-title"><span>04</span><h3>PP Rank 显存细分</h3></div>
              <p className="rank-assumption">连续切层；PP 0 在途激活最多，末 Rank 承载 MTP 与输出层。</p>
              <div className="rank-list">
                {result.ppRanks.map((rank) => (
                  <details key={rank.rank} className={`rank-disclosure ${rank.rank === result.peakRank.rank ? "peak" : ""}`} open={rank.rank === result.peakRank.rank}>
                    <summary>
                      <span>PP {rank.rank}</span>
                      <small>{rank.layerCount > 0 ? `L${rank.layerStart}–${rank.layerEnd - 1}` : "No blocks"}</small>
                      <strong>{gb(rank.total)}</strong>
                    </summary>
                    <div className="rank-detail">
                      <div className="rank-meta">
                        <span>{rank.linearLayers} Linear</span>
                        <span>{rank.fullLayers} Full</span>
                        <span>{rank.pipelineInflight} in-flight</span>
                        {rank.roles.map((role) => <span key={role}>{role}</span>)}
                      </div>
                      <dl>
                        <div><dt>本地参数</dt><dd>{compact(rank.localParams)}B</dd></div>
                        <div><dt>模型权重</dt><dd>{gb(rank.modelWeights)}</dd></div>
                        <div><dt>梯度</dt><dd>{gb(rank.gradients)}</dd></div>
                        <div><dt>优化器状态</dt><dd>{gb(rank.optimizer)}</dd></div>
                        <div><dt>激活值</dt><dd>{gb(rank.activation)}</dd></div>
                        <div><dt>运行时预留</dt><dd>{gb(inputs.overhead)}</dd></div>
                      </dl>
                    </div>
                  </details>
                ))}
              </div>
            </article>
          </div>

          <section className="formula-section" aria-labelledby="formula-title">
            <div className="formula-heading">
              <div className="detail-title"><span>05</span><h3 id="formula-title">粗略估算公式</h3></div>
              <p>公式会随当前并行拓扑、序列长度和内存策略实时更新。</p>
            </div>

            <div className="formula-context">
              <span>P<sub>dense,local</sub> = {compact(result.localDense)}B</span>
              <span>P<sub>expert,local</sub> = {compact(result.localExpert)}B</span>
              <span>P<sub>local</sub> = {compact(result.localParams)}B</span>
              <span>Tokens / GPU = {Math.round(result.tokens).toLocaleString("en-US")}</span>
            </div>

            <section className="parameter-logic" aria-labelledby="parameter-logic-title">
              <div className="parameter-logic-heading">
                <h4 id="parameter-logic-title">参数量拆解</h4>
                <p>先计算全局参数，再按并行维度切分到单卡。</p>
              </div>
              <div className="parameter-logic-grid">
                <article>
                  <header>
                    <span>Dense</span>
                    <strong>{compact(result.denseParams)}B 全局</strong>
                  </header>
                  <code>
                    {shape.experts > 0
                      ? <>P<sub>dense</sub> = [L<sub>full</sub>P<sub>full</sub> + L<sub>linear</sub>P<sub>linear</sub> + L(P<sub>shared</sub> + P<sub>router</sub> + P<sub>norm</sub>)] × m<sub>MTP</sub> + P<sub>embed</sub></>
                      : <>P<sub>dense</sub> = [L<sub>full</sub>P<sub>full</sub> + L<sub>linear</sub>P<sub>linear</sub> + L(P<sub>MLP</sub> + P<sub>norm</sub>)] × m<sub>MTP</sub> + P<sub>embed</sub></>}
                  </code>
                  <dl>
                    <div><dt>Full attention：{shape.fullAttentionLayers} 层，Q + KV + O</dt><dd>{compact(result.fullAttentionParams)}B</dd></div>
                    <div><dt>Linear attention：{shape.linearAttentionLayers} 层，QK + V + O + gates</dt><dd>{compact(result.linearAttentionParams)}B</dd></div>
                    <div><dt>{shape.experts > 0 ? "共享专家、router、RMSNorm" : "Dense gated MLP、RMSNorm"}</dt><dd>{compact(shape.experts > 0 ? result.denseMoEParams : result.denseMlpParams + result.denseMoEParams)}B</dd></div>
                    <div><dt>词嵌入{shape.tied ? "（权重绑定）" : "（输入/输出各一份）"}</dt><dd>{compact(result.embedding)}B</dd></div>
                  </dl>
                  <p>P<sub>dense,local</sub> = {compact(result.peakRank.denseGlobal)}B on PP {result.peakRank.rank} / {inputs.tp} TP = <b>{compact(result.localDense)}B</b></p>
                </article>
                {shape.experts > 0 ? (
                  <article>
                    <header>
                      <span>Experts</span>
                      <strong>{compact(result.expertParams)}B 全局</strong>
                    </header>
                    <code>P<sub>expert</sub> = L × m<sub>MTP</sub> × N<sub>expert</sub> × 3 × H × I<sub>moe</sub></code>
                    <dl>
                      <div><dt>每专家 gated MLP：3 × {shape.hidden.toLocaleString("en-US")} × {shape.moeIntermediate.toLocaleString("en-US")}</dt><dd>{compact(result.expertPerLayer / Math.max(shape.experts, 1))}B</dd></div>
                      <div><dt>每层全部专家：{shape.experts} experts</dt><dd>{compact(result.expertPerLayer)}B</dd></div>
                      <div><dt>有效层数：{shape.layers} × m<sub>MTP</sub> = {(shape.layers * result.mtpMultiplier).toFixed(0)}</dt><dd>{compact(result.expertParams)}B</dd></div>
                      <div><dt>TopK = {shape.topK} 仅影响激活专家与激活值</dt><dd>不减少权重</dd></div>
                    </dl>
                    <p>P<sub>expert,local</sub> = {compact(result.peakRank.expertGlobal)}B on PP {result.peakRank.rank} / ({inputs.etp} ETP × {inputs.ep} EP) = <b>{compact(result.localExpert)}B</b></p>
                  </article>
                ) : (
                  <article>
                    <header>
                      <span>Dense MLP</span>
                      <strong>{compact(result.denseMlpParams)}B 全局</strong>
                    </header>
                    <code>P<sub>MLP</sub> = L × m<sub>MTP</sub> × 3 × H × I<sub>FFN</sub></code>
                    <dl>
                      <div><dt>每层 gated MLP：3 × {shape.hidden.toLocaleString("en-US")} × {shape.denseIntermediate.toLocaleString("en-US")}</dt><dd>{compact(result.denseMlpParams / Math.max(shape.layers * result.mtpMultiplier, 1))}B</dd></div>
                      <div><dt>有效层数：{shape.layers} × m<sub>MTP</sub> = {(shape.layers * result.mtpMultiplier).toFixed(0)}</dt><dd>{compact(result.denseMlpParams)}B</dd></div>
                      <div><dt>TP 切分前馈层权重与中间激活</dt><dd>{inputs.tp} TP</dd></div>
                      <div><dt>无独占专家</dt><dd>P<sub>expert</sub> = 0</dd></div>
                    </dl>
                    <p>P<sub>MLP,local</sub> 已包含在 P<sub>dense,local</sub> 中；不按 EP 或 ETP 再切分。</p>
                  </article>
                )}
              </div>
            </section>

            <div className="formula-grid">
              <article>
                <header><span>01</span><h4>模型权重</h4><strong>{gb(result.modelWeights)}</strong></header>
                <code>M<sub>weight</sub> = P<sub>local</sub> × 2 B ÷ 2³⁰</code>
                <p>{compact(result.localParams)}B × 2 B ÷ 2³⁰ = {gb(result.modelWeights)}</p>
              </article>
              <article>
                <header><span>02</span><h4>梯度</h4><strong>{gb(result.gradients)}</strong></header>
                <code>M<sub>grad</sub> = P<sub>local</sub> × 4 B ÷ 2³⁰</code>
                <p>{compact(result.localParams)}B × 4 B ÷ 2³⁰ = {gb(result.gradients)}</p>
              </article>
              <article>
                <header><span>03</span><h4>优化器状态</h4><strong>{gb(result.optimizer)}</strong></header>
                {inputs.distributedOptimizer ? (
                  <>
                    <code>M<sub>optim</sub> = (P<sub>dense</sub> × 12 / DP + P<sub>expert</sub> × 12 / EDP) ÷ 2³⁰</code>
                    <p>EDP = DP / EP = {result.expertDp.toFixed(0)}；当前为 {gb(result.optimizer)}</p>
                  </>
                ) : (
                  <>
                    <code>M<sub>optim</sub> = P<sub>local</sub> × 12 B ÷ 2³⁰</code>
                    <p>未切分 Adam 主权重与一二阶矩；当前为 {gb(result.optimizer)}</p>
                  </>
                )}
              </article>
              <article>
                <header><span>04</span><h4>激活值</h4><strong>{gb(result.activation)}</strong></header>
                <code>Tokens = MBS × SeqLen / CP</code>
                <code>A<sub>rank</sub> ≈ (L<sub>full,rank</sub>A<sub>full</sub> + L<sub>linear,rank</sub>A<sub>linear</sub> + L<sub>rank</sub>A<sub>moe</sub>) × N<sub>flight,rank</sub> × r</code>
                <p>Peak PP {result.peakRank.rank}：N<sub>flight</sub> = {result.pipelineInflight}，r = {result.recomputeFactor.toFixed(2)}；当前为 {gb(result.activation)}</p>
              </article>
            </div>

            <div className="scenario-heading">
              <h4>不同配置下的快速对比</h4>
              <p>其他输入保持当前值，仅切换对应策略。</p>
            </div>
            <div className="scenario-grid">
              <article><span>不重计算</span><strong>{gb(result.totalNoRecompute)}</strong><small>激活 {gb(result.activationNoRecompute)} · r = 1.00</small></article>
              <article><span>选择性重计算</span><strong>{gb(result.totalSelective)}</strong><small>激活 {gb(result.activationSelective)} · r ≈ 0.56</small></article>
              <article><span>全量重计算</span><strong>{gb(result.totalFull)}</strong><small>激活 {gb(result.activationFull)} · r ≈ 0.20</small></article>
              <article><span>分布式 Adam</span><strong>{gb(result.totalDistributed)}</strong><small>优化器 {gb(result.optimizerDistributed)}</small></article>
              <article><span>常规 Adam</span><strong>{gb(result.totalReplicated)}</strong><small>优化器 {gb(result.optimizerReplicated)}</small></article>
            </div>

          </section>

          <p className="method-note">
            估算口径：BF16 参数 2 B、FP32 主梯度 4 B、Adam FP32 主权重与一二阶矩 12 B；PP Rank 按连续层范围切分，MTP 放在末 Rank，tied embedding 在首尾 Rank 各驻留一份；在途 micro-batch 近似为 PP − rank。实际峰值还会受 Megatron 版本、Pipeline 调度、Sequence Parallel、NCCL 缓冲、CUDA Graph 与显存碎片影响，请用训练实测校准“额外预留”。
          </p>
        </section>
      </section>
    </main>
  );
}
