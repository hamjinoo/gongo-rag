import Link from "next/link";
import {
  citations,
  documents,
  type Citation,
  workspace,
} from "../lib/mock-trace";

export function VerifyScreen({ citation }: { citation: Citation }) {
  return (
    <main className="verify-page">
      <header className="verify-header">
        <Link className="back-link" href="/">
          <span aria-hidden="true">←</span> 답변으로 돌아가기
        </Link>
        <span className="header-divider" aria-hidden="true" />
        <p>
          근거 검증 ·{" "}
          <strong>
            주장 {citation.id} {citation.claim.split("→")[0].trim()}
          </strong>
        </p>
        <nav className="citation-navigation" aria-label="근거 이동">
          <span>근거 이동</span>
          {citations.map((item) => (
            <Link
              aria-current={item.id === citation.id ? "page" : undefined}
              href={`/verify/${item.id}`}
              key={item.id}
            >
              {item.id}
            </Link>
          ))}
        </nav>
      </header>

      <div className="verify-layout">
        <aside className="document-sidebar" aria-label="검색 문서 목록">
          <h2>문서 {workspace.documentCount}건</h2>
          <nav>
            {documents.slice(0, 5).map((document) => {
              const itemCitation = citations.find(
                (item) =>
                  item.sourceFilename.replace(/\.[^.]+$/, "") === document,
              );
              const active =
                citation.sourceFilename.replace(/\.[^.]+$/, "") === document;

              return itemCitation ? (
                <Link
                  className={active ? "active" : undefined}
                  href={`/verify/${itemCitation.id}`}
                  key={document}
                >
                  {document}
                </Link>
              ) : (
                <span key={document}>{document}</span>
              );
            })}
            <span className="muted">+ {documents.length - 5}건 더보기</span>
          </nav>
        </aside>

        <section className="document-viewer" aria-labelledby="document-title">
          <div className="document-meta">
            <h1 id="document-title">{citation.sourceFilename}</h1>
            <span>
              {citation.page} / {citation.totalPages} 페이지
            </span>
          </div>
          <article className="document-paper">
            <p className="document-section">{citation.section}</p>
            <p>{citation.before}</p>
            <p>
              <mark id={`evidence-${citation.id}`}>{citation.highlighted}</mark>{" "}
              <span className="claim-label">주장 {citation.id}의 근거</span>{" "}
              {citation.after}
            </p>
            <p>
              표시된 문장은 답변에서 사용한 청크{" "}
              <code>{citation.chunkId}</code>의 원문 구간입니다. 파란색 영역과
              답변의 주장이 같은 내용을 말하는지 직접 확인할 수 있습니다.
            </p>
          </article>
        </section>

        <aside className="claim-panel" aria-labelledby="claim-panel-title">
          <div className="claim-panel-heading">
            <h2 id="claim-panel-title">답변과 근거 연결</h2>
            <p>주장을 누르면 해당 원문으로 이동합니다</p>
          </div>
          <div className="claim-list">
            {citations.map((item) => (
              <Link
                className={item.id === citation.id ? "active" : undefined}
                href={`/verify/${item.id}#evidence-${item.id}`}
                key={item.id}
              >
                <span className="claim-title">
                  <b>{item.id}</b>
                  <strong>{item.claim}</strong>
                </span>
                <span className="claim-source">
                  근거: {item.sourceLabel} p.{item.page} ·{" "}
                  <em>✓ 원문 일치 확인</em>
                </span>
              </Link>
            ))}
          </div>
        </aside>
      </div>
    </main>
  );
}

