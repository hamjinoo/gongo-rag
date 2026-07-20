import Link from "next/link";
import { Brand } from "./Brand";
import {
  isUsed,
  traceDurations,
  traceRows,
  type TraceRow,
} from "../lib/mock-trace";

export type TraceFilter = "all" | "used" | "excluded";

const statusLabel: Record<TraceRow["status"], string> = {
  citation: "인용",
  context: "컨텍스트",
  duplicate: "중복 제거",
  excluded: "순위 제외",
};

function filterRows(filter: TraceFilter) {
  if (filter === "used") return traceRows.filter(isUsed);
  if (filter === "excluded") return traceRows.filter((row) => !isUsed(row));
  return traceRows;
}

export function AdminTraceScreen({
  queryId,
  filter,
}: {
  queryId: string;
  filter: TraceFilter;
}) {
  const rows = filterRows(filter);
  const usedCount = traceRows.filter(isUsed).length;
  const duplicateCount = traceRows.filter(
    (row) => row.status === "duplicate",
  ).length;
  const verifiedCount = traceRows.filter(
    (row) => row.citationVerified,
  ).length;

  const filters: { key: TraceFilter; label: string; count: number }[] = [
    { key: "all", label: "전체", count: traceRows.length },
    { key: "used", label: "사용됨", count: usedCount },
    {
      key: "excluded",
      label: "제외됨",
      count: traceRows.length - usedCount,
    },
  ];

  return (
    <main className="admin-page">
      <header className="admin-header">
        <Brand admin />
        <code>query_id: {queryId}</code>
      </header>

      <div className="admin-content">
        <section className="metric-grid" aria-label="검색 요약">
          <article>
            <h2>전체 소요</h2>
            <strong>
              2.41<small>s</small>
            </strong>
          </article>
          <article>
            <h2>검색된 청크</h2>
            <strong>
              {traceRows.length}<small>건</small>
            </strong>
          </article>
          <article>
            <h2>사용된 청크</h2>
            <strong>
              {usedCount}<small>건</small>
            </strong>
          </article>
          <article>
            <h2>중복 제거</h2>
            <strong>
              {duplicateCount}<small>건</small>
            </strong>
          </article>
          <article className="success">
            <h2>인용 검증</h2>
            <strong>
              {verifiedCount}/{verifiedCount} <small>통과</small>
            </strong>
          </article>
        </section>

        <section className="trace-panel" aria-labelledby="trace-title">
          <div className="trace-toolbar">
            <div>
              <h1 id="trace-title">검색 결과 상세</h1>
              <span>BM25 + 벡터 하이브리드 → rerank</span>
            </div>
            <nav aria-label="검색 결과 필터">
              {filters.map((item) => (
                <Link
                  aria-current={filter === item.key ? "page" : undefined}
                  href={`/admin/trace/${queryId}?filter=${item.key}`}
                  key={item.key}
                >
                  {item.label} {item.count}
                </Link>
              ))}
            </nav>
          </div>

          <div className="table-scroll">
            <table>
              <thead>
                <tr>
                  <th>청크 ID</th>
                  <th>출처</th>
                  <th className="numeric">BM25</th>
                  <th className="numeric">벡터</th>
                  <th className="numeric">rerank</th>
                  <th>상태</th>
                  <th>인용 검증</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((row) => (
                  <tr
                    className={
                      row.status === "citation"
                        ? "cited"
                        : row.status === "duplicate" ||
                            row.status === "excluded"
                          ? "discarded"
                          : undefined
                    }
                    key={row.chunkId}
                  >
                    <td>
                      <code>{row.chunkId}</code>
                    </td>
                    <td>{row.source}</td>
                    <td className="numeric">
                      <code>{row.bm25.toFixed(1)}</code>
                    </td>
                    <td className="numeric">
                      <code>{row.vector.toFixed(3)}</code>
                    </td>
                    <td className="numeric rerank-score">
                      <code>
                        {row.rerank === null ? "—" : row.rerank.toFixed(2)}
                      </code>
                    </td>
                    <td>
                      {row.status === "citation" ? (
                        <Link href={`/verify/${row.citationId}`}>
                          {statusLabel[row.status]} {row.citationId}
                        </Link>
                      ) : (
                        <span className={`status-${row.status}`}>
                          {statusLabel[row.status]}
                        </span>
                      )}
                    </td>
                    <td className="verification">
                      {row.citationVerified ? "✓ 통과" : "—"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <footer className="trace-footer">
            <div>
              {traceDurations.map(([label, duration]) => (
                <span key={label}>
                  {label} {duration}
                </span>
              ))}
            </div>
            <strong>
              citation verification: {verifiedCount}건 모두 원문 스팬 일치
            </strong>
          </footer>
        </section>
      </div>
    </main>
  );
}
