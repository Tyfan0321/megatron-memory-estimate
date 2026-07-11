type Qwen35Preset = {
  id: string;
  family: "Dense" | "MoE";
  nominalParams: string;
  hidden: number;
  layers: number;
  heads: number;
  kvHeads: number;
  linearValueHeads: number;
  tieEmbeddings: boolean;
  intermediate?: number;
  experts?: number;
  topK?: number;
  moeIntermediate?: number;
  sharedIntermediate?: number;
};

const QWEN35_PRESETS: Qwen35Preset[] = [
  { id: "Qwen3.5-0.8B", family: "Dense", nominalParams: "0.8B", hidden: 1024, layers: 24, heads: 8, kvHeads: 2, linearValueHeads: 16, tieEmbeddings: true, intermediate: 3584 },
  { id: "Qwen3.5-2B", family: "Dense", nominalParams: "2B", hidden: 2048, layers: 24, heads: 8, kvHeads: 2, linearValueHeads: 16, tieEmbeddings: true, intermediate: 6144 },
  { id: "Qwen3.5-4B", family: "Dense", nominalParams: "4B", hidden: 2560, layers: 32, heads: 16, kvHeads: 4, linearValueHeads: 32, tieEmbeddings: true, intermediate: 9216 },
  { id: "Qwen3.5-9B", family: "Dense", nominalParams: "9B", hidden: 4096, layers: 32, heads: 16, kvHeads: 4, linearValueHeads: 32, tieEmbeddings: false, intermediate: 12288 },
  { id: "Qwen3.5-27B", family: "Dense", nominalParams: "27B", hidden: 5120, layers: 64, heads: 24, kvHeads: 4, linearValueHeads: 48, tieEmbeddings: false, intermediate: 17408 },
  { id: "Qwen3.5-35B-A3B", family: "MoE", nominalParams: "35B / 3B active", hidden: 2048, layers: 40, heads: 16, kvHeads: 2, linearValueHeads: 32, tieEmbeddings: false, experts: 256, topK: 8, moeIntermediate: 512, sharedIntermediate: 512 },
  { id: "Qwen3.5-122B-A10B", family: "MoE", nominalParams: "122B / 10B active", hidden: 3072, layers: 48, heads: 32, kvHeads: 2, linearValueHeads: 64, tieEmbeddings: false, experts: 256, topK: 8, moeIntermediate: 1024, sharedIntermediate: 1024 },
  { id: "Qwen3.5-397B-A17B", family: "MoE", nominalParams: "397B / 17B active", hidden: 4096, layers: 60, heads: 32, kvHeads: 2, linearValueHeads: 64, tieEmbeddings: false, experts: 512, topK: 10, moeIntermediate: 1024, sharedIntermediate: 1024 },
];

export const QWEN35_COLLECTION = QWEN35_PRESETS.map((preset) => ({
  ...preset,
  sourceUrl: `https://huggingface.co/Qwen/${preset.id}`,
}));

export function qwen35Config(preset: Qwen35Preset): Record<string, unknown> {
  const isMoE = preset.family === "MoE";
  const textConfig: Record<string, unknown> = {
    hidden_size: preset.hidden,
    num_hidden_layers: preset.layers,
    num_attention_heads: preset.heads,
    head_dim: 256,
    num_key_value_heads: preset.kvHeads,
    vocab_size: 248320,
    tie_word_embeddings: preset.tieEmbeddings,
    mtp_num_hidden_layers: 1,
    linear_num_key_heads: 16,
    linear_key_head_dim: 128,
    linear_num_value_heads: preset.linearValueHeads,
    linear_value_head_dim: 128,
    layer_types: Array.from({ length: preset.layers }, (_, index) =>
      index % 4 === 3 ? "full_attention" : "linear_attention",
    ),
  };

  if (isMoE) {
    textConfig.num_experts = preset.experts;
    textConfig.num_experts_per_tok = preset.topK;
    textConfig.moe_intermediate_size = preset.moeIntermediate;
    textConfig.shared_expert_intermediate_size = preset.sharedIntermediate;
  } else {
    textConfig.intermediate_size = preset.intermediate;
  }

  return {
    architectures: [isMoE ? "Qwen3_5MoeForConditionalGeneration" : "Qwen3_5ForConditionalGeneration"],
    model_type: isMoE ? "qwen3_5_moe" : "qwen3_5",
    text_config: textConfig,
    tie_word_embeddings: preset.tieEmbeddings,
  };
}
