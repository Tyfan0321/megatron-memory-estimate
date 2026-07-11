"use client";

import { useMemo, useState } from "react";

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
  fullAttentionLayers: number;
  linearAttentionLayers: number;
  heads: number;
  headDim: number;
  kvHeads: number;
  experts: number;
  topK: number;
  moeIntermediate: number;
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

const GiB = 1024 ** 3;
const n = (value: unknown, fallback = 0) =>
  typeof value === "number" && Number.isFinite(value) ? value : fallback;

function textConfigOf(config: JsonObject): JsonObject {
  const nested = config.text_config;
  return nested && typeof nested === "object" ? (nested as JsonObject) : config;
}

function shapeFromConfig(config: JsonObject): ModelShape {
  const c = textConfigOf(config);
  const layerTypes = Array.isArray(c.layer_types)
    ? c.layer_types.map(String)
    : Array(n(c.num_hidden_layers, 1)).fill("full_attention");
  const architecture = Array.isArray(config.architectures)
    ? String(config.architectures[0] ?? "Megatron 模型")
    : String(c.model_type ?? "Megatron 模型");

  return {
    name: architecture.replace(/ForConditionalGeneration|ForCausalLM/g, ""),
    hidden: n(c.hidden_size),
    layers: n(c.num_hidden_layers, layerTypes.length),
    fullAttentionLayers: layerTypes.filter((item) => item === "full_attention").length,
    linearAttentionLayers: layerTypes.filter((item) => item !== "full_attention").length,
    heads: n(c.num_attention_heads),
    headDim: n(c.head_dim, n(c.hidden_size) / Math.max(n(c.num_attention_heads, 1), 1)),
    kvHeads: n(c.num_key_value_heads, n(c.num_attention_heads)),
    experts: n(c.num_experts),
    topK: n(c.num_experts_per_tok, 1),
    moeIntermediate: n(c.moe_intermediate_size, n(c.intermediate_size)),
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
  const sharedPerLayer = 3 * shape.hidden * shape.sharedIntermediate;
  const routerPerLayer = shape.hidden * shape.experts;
  const moePerLayer = expertPerLayer + sharedPerLayer + routerPerLayer;
  const normPerLayer = shape.hidden * 2;
  const mtpMultiplier = shape.layers > 0 ? 1 + shape.mtpLayers / shape.layers : 1;

  const expertParams = expertPerLayer * shape.layers * mtpMultiplier;
  const denseLayerParams =
    (shape.fullAttentionLayers * fullAttention +
      shape.linearAttentionLayers * linearAttention +
      shape.layers * (sharedPerLayer + routerPerLayer + normPerLayer)) *
    mtpMultiplier;
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

  const topology = x.tp * x.pp * x.cp;
  const dp = x.totalGpus / Math.max(topology, 1);
  const validTopology = Number.isInteger(dp) && dp >= 1 && Number.isInteger(dp / x.ep);
  const expertDp = Math.max(dp / Math.max(x.ep, 1), 1);
  const localDense = denseParams / Math.max(x.tp * x.pp, 1);
  const localExpert = expertParams / Math.max(x.etp * x.ep * x.pp, 1);

  // Megatron mixed-precision convention: BF16 weights (2 B), FP32 main grads (4 B),
  // and 12 B of Adam master weights/moments, sharded across data-parallel ranks.
  const baseBytes = (localDense + localExpert) * 6;
  const optimizerBytes = x.distributedOptimizer
    ? localDense * 12 / Math.max(dp, 1) + localExpert * 12 / expertDp
    : (localDense + localExpert) * 12;
  const modelState = (baseBytes + optimizerBytes) / GiB;

  const tokens = x.microBatch * x.sequenceLength / Math.max(x.cp, 1);
  const hiddenBytes = tokens * shape.hidden * 2;
  const fullLayerActivation = hiddenBytes * (4 + 12 / Math.max(x.tp, 1));
  const linearWidth = shape.linearValueHeads * shape.linearValueDim;
  const linearLayerActivation =
    hiddenBytes * (3.5 + 9 / Math.max(x.tp, 1)) +
    tokens * linearWidth * 2 / Math.max(x.tp, 1);
  const moeActivation =
    tokens * shape.topK * shape.moeIntermediate * 2 / Math.max(x.etp * x.ep, 1);
  const savedPerModel =
    shape.fullAttentionLayers * fullLayerActivation +
    shape.linearAttentionLayers * linearLayerActivation +
    shape.layers * moeActivation;
  const pipelineInflight = Math.max(x.pp, 1);
  const recomputeFactor = x.recompute === "full" ? 0.2 : x.recompute === "selective" ? 0.56 : 1;
  const activation = savedPerModel / Math.max(x.pp, 1) * pipelineInflight * recomputeFactor / GiB;
  const total = modelState + activation + x.overhead;
  const capacity = x.gpuMemory;

  return {
    totalParams,
    activeParams,
    denseParams,
    expertParams,
    localParams: localDense + localExpert,
    modelState,
    activation,
    overhead: x.overhead,
    total,
    capacity,
    headroom: capacity - total,
    usage: total / Math.max(capacity, 1) * 100,
    dp,
    expertDp,
    validTopology,
  };
}

const compact = (value: number) => {
  const billions = value / 1e9;
  return billions >= 100 ? billions.toFixed(0) : billions >= 10 ? billions.toFixed(1) : billions.toFixed(2);
};
const gb = (value: number) => `${value.toFixed(1)} GB`;

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

  const shape = useMemo(() => shapeFromConfig(config), [config]);
  const result = useMemo(() => calculate(shape, inputs), [shape, inputs]);
  const update = <K extends keyof Inputs>(key: K, value: Inputs[K]) =>
    setInputs((current) => ({ ...current, [key]: value }));

  const applyConfig = () => {
    try {
      const parsed = JSON.parse(configText) as JsonObject;
      setConfig(parsed);
      setConfigError("");
    } catch (error) {
      setConfigError(error instanceof Error ? error.message : "JSON 格式不正确");
    }
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
          <h1>在启动训练前，<br />先看清每一 GB。</h1>
        </div>
        <div className="hero-copy">
          <p>
            基于 Megatron-LM 的混合精度训练口径，估算参数、梯度、优化器状态、激活值与运行时预留。
          </p>
          <div className="model-badge">
            <span>已载入配置</span>
            <strong>{shape.name}</strong>
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

          <div className="control-group">
            <h3>硬件</h3>
            <div className="field-grid two">
              <NumberField label="GPU 数量" value={inputs.totalGpus} onChange={(v) => update("totalGpus", v)} />
              <label className="field">
                <span>单卡显存</span>
                <span className="input-wrap select-wrap">
                  <select value={inputs.gpuMemory} onChange={(e) => update("gpuMemory", Number(e.target.value))}>
                    <option value={80}>80 GB · H100</option>
                    <option value={96}>96 GB</option>
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
        </aside>

        <section className="results" aria-live="polite">
          <div className="section-heading light">
            <span>02</span>
            <div><h2>单卡峰值估算</h2><p>以最繁忙 Pipeline Stage 为准</p></div>
          </div>

          <div className="total-card">
            <div className="total-topline">
              <span className={`status ${status.tone}`}><i />{status.label}</span>
              <span>{inputs.totalGpus} × {inputs.gpuMemory} GB</span>
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
          </div>

          <div className="breakdown">
            <article><span>模型状态</span><strong>{gb(result.modelState)}</strong><small>权重 · 主梯度 · Adam</small><div style={{ width: `${result.total ? result.modelState / result.total * 100 : 0}%` }} /></article>
            <article><span>激活值</span><strong>{gb(result.activation)}</strong><small>{inputs.recompute === "full" ? "全量重计算" : inputs.recompute === "selective" ? "选择性重计算" : "不重计算"}</small><div style={{ width: `${result.total ? result.activation / result.total * 100 : 0}%` }} /></article>
            <article><span>运行时预留</span><strong>{gb(result.overhead)}</strong><small>NCCL · kernel · 碎片</small><div style={{ width: `${result.total ? result.overhead / result.total * 100 : 0}%` }} /></article>
          </div>

          <div className="details-grid">
            <article>
              <div className="detail-title"><span>03</span><h3>模型剖面</h3></div>
              <dl>
                <div><dt>总参数</dt><dd>{compact(result.totalParams)}B</dd></div>
                <div><dt>单 token 激活</dt><dd>{compact(result.activeParams)}B</dd></div>
                <div><dt>单卡本地参数</dt><dd>{compact(result.localParams)}B</dd></div>
                <div><dt>MoE 专家参数</dt><dd>{compact(result.expertParams)}B</dd></div>
                <div><dt>注意力层</dt><dd>{shape.linearAttentionLayers} Linear + {shape.fullAttentionLayers} Full</dd></div>
                <div><dt>专家路由</dt><dd>{shape.topK} / {shape.experts}</dd></div>
              </dl>
            </article>
            <article>
              <div className="detail-title"><span>04</span><h3>优化建议</h3></div>
              <ul className="recommendations">
                {!result.validTopology ? <li><b>先修复拓扑</b><span>让 GPU 数量满足并行维度的整除关系。</span></li> : null}
                {result.activation > result.modelState * 0.45 ? <li><b>激活值占比较高</b><span>优先提高 CP 或使用全量重计算。</span></li> : null}
                {result.headroom < 8 ? <li><b>保留更多余量</b><span>提高 TP / PP / EP，或换用更大显存 GPU。</span></li> : <li><b>当前余量健康</b><span>建议实测后将额外预留校准到集群环境。</span></li>}
                {!inputs.distributedOptimizer ? <li><b>开启分布式优化器</b><span>通常能显著降低 Adam 状态占用。</span></li> : null}
              </ul>
            </article>
          </div>

          <p className="method-note">
            估算口径：BF16 参数 2 B、FP32 主梯度 4 B、Adam 主权重与一二阶矩 12 B；激活值按层类型、序列长度、并行度及重计算策略建模。结果不包含框架外的固定占用，已通过“额外预留”单独计入。
          </p>
        </section>
      </section>
    </main>
  );
}
