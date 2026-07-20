import type { Metadata } from "next";
import {
  AdminTraceScreen,
  type TraceFilter,
} from "../../../components/AdminTraceScreen";

export const metadata: Metadata = {
  title: "검색 기록 | DocLens Trace 관리자",
  description: "RAG 검색·재정렬·인용 검증 과정을 확인하는 관리자 화면",
};

type AdminTracePageProps = {
  params: Promise<{ queryId: string }>;
  searchParams: Promise<{ filter?: string }>;
};

function toFilter(value?: string): TraceFilter {
  return value === "used" || value === "excluded" ? value : "all";
}

export default async function AdminTracePage({
  params,
  searchParams,
}: AdminTracePageProps) {
  const [{ queryId }, { filter }] = await Promise.all([params, searchParams]);

  return <AdminTraceScreen queryId={queryId} filter={toFilter(filter)} />;
}

