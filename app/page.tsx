import type { Metadata } from "next";
import modelConfig from "../config.json";
import { Estimator } from "./Estimator";

export const metadata: Metadata = {
  title: "Megatron SFT 显存估算器",
  description: "面向 Megatron-LM 全量 SFT 的单卡显存与并行策略估算工具。",
};

export default function Home() {
  return <Estimator initialConfig={modelConfig} />;
}
