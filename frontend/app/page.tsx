import type { Metadata } from "next";
import { AnswerScreen } from "./components/AnswerScreen";

export const metadata: Metadata = {
  title: "답변 | DocLens Trace",
  description: "답변에 쓰인 근거를 원문까지 추적할 수 있는 RAG 데모",
};

export default function Home() {
  return <AnswerScreen />;
}

