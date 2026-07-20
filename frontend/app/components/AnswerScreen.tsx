import Link from "next/link";
import { Brand } from "./Brand";
import {
  citations,
  relatedDocuments,
  workspace,
} from "../lib/mock-trace";

function CitationLink({
  id,
  children,
}: {
  id: string;
  children: React.ReactNode;
}) {
  const citation = citations.find((item) => item.id === id) ?? citations[0];

  return (
    <Link
      className="citation-chip"
      href={`/verify/${id}`}
      aria-label={`근거 ${id} 확인: ${citation.claim}`}
      data-preview={citation.excerpt}
    >
      <span aria-hidden="true">{id}</span>
      {children}
    </Link>
  );
}

export function AnswerScreen() {
  return (
    <main className="answer-page">
      <header className="answer-header">
        <div className="header-inner">
          <Brand />
          <span className="header-divider" aria-hidden="true" />
          <p className="workspace-title">
            {workspace.title}
            <span>· 문서 {workspace.documentCount}건</span>
          </p>
          <label className="secondary-button" title="분석할 문서 선택">
            <span aria-hidden="true">+</span> 문서 추가
            <input
              className="sr-only"
              type="file"
              multiple
              accept=".pdf,.doc,.docx,.hwp,.hwpx,.png,.jpg,.jpeg,.tif,.tiff"
              aria-label="분석할 문서 추가"
            />
          </label>
        </div>
      </header>

      <section className="answer-content" aria-labelledby="question-heading">
        <h1 id="question-heading">{workspace.question}</h1>

        <div className="answer-copy">
          <p>
            1분기 매출은 전년 동기 대비 <mark>8.2% 감소</mark>했습니다.
            아시아 지역 수요 둔화로 해외 매출이 12% 줄었고{" "}
            <CitationLink id="1">실적보고서 p.14</CitationLink>, 원자재 가격
            상승의 판가 반영 지연으로 매출총이익률이 3.1%p 하락했습니다{" "}
            <CitationLink id="2">원가분석 메모 p.3</CitationLink>.
          </p>
          <p>
            또한 신제품 출시 연기로 약 45억 원의 매출이 2분기로 이월된
            점도 영향을 미쳤습니다{" "}
            <CitationLink id="3">출시일정 변경안 p.2</CitationLink>.
          </p>
        </div>

        <section className="evidence-section" aria-labelledby="cited-evidence-title">
          <h2 id="cited-evidence-title">
            답변에 인용된 근거 {citations.length}건
          </h2>
          <div className="evidence-grid">
            {citations.map((citation) => (
              <article className="evidence-card" key={citation.id}>
                <div className="evidence-card-heading">
                  <span className="citation-number">{citation.id}</span>
                  <strong title={citation.sourceFilename}>
                    {citation.sourceFilename}
                  </strong>
                  <span>p.{citation.page}</span>
                </div>
                <p>&ldquo;{citation.excerpt}&rdquo;</p>
                <Link href={`/verify/${citation.id}`}>
                  원문에서 보기 <span aria-hidden="true">→</span>
                </Link>
              </article>
            ))}
          </div>
        </section>

        <section className="related-section" aria-labelledby="related-title">
          <h2 id="related-title">
            추가로 검색된 관련 문서 {relatedDocuments.length}건
          </h2>
          <div className="related-list">
            {relatedDocuments.map((document) => (
              <span className="related-document" key={document.name}>
                {document.name} <small>p.{document.page}</small>
              </span>
            ))}
          </div>
        </section>

        <details className="search-process">
          <summary>
            <span>검색 과정 자세히 보기</span>
            <small>질문 분석 → 검색 → 선별 → 생성 · 4단계 · 2.41초</small>
          </summary>
          <ol>
            <li>
              <strong>질문 분석</strong>
              <span>핵심어와 검색 의도를 정리합니다.</span>
            </li>
            <li>
              <strong>하이브리드 검색</strong>
              <span>BM25와 벡터 검색 결과를 합칩니다.</span>
            </li>
            <li>
              <strong>선별</strong>
              <span>reranker가 답변에 가까운 근거를 다시 정렬합니다.</span>
            </li>
            <li>
              <strong>생성·검증</strong>
              <span>
                근거 안에서만 답하고 인용 문장이 실제 원문과 맞는지 확인합니다.
              </span>
            </li>
          </ol>
          <Link href={`/admin/trace/${workspace.queryId}`}>
            관리자용 검색 기록 열기
          </Link>
        </details>
      </section>

      <form className="follow-up" action="/" method="get" role="search">
        <label className="sr-only" htmlFor="follow-up-question">
          후속 질문
        </label>
        <input
          id="follow-up-question"
          name="question"
          placeholder="후속 질문을 입력하세요…"
          autoComplete="off"
        />
        <button type="submit" aria-label="후속 질문 보내기">
          <span aria-hidden="true">↑</span>
        </button>
      </form>
    </main>
  );
}
