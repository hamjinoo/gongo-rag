export type Citation = {
  id: string;
  claim: string;
  inlineLabel: string;
  sourceFilename: string;
  sourceLabel: string;
  page: number;
  totalPages: number;
  section: string;
  excerpt: string;
  before: string;
  highlighted: string;
  after: string;
  chunkId: string;
};

export type TraceStatus = "citation" | "context" | "duplicate" | "excluded";

export type TraceRow = {
  chunkId: string;
  source: string;
  bm25: number;
  vector: number;
  rerank: number | null;
  status: TraceStatus;
  citationId?: string;
  citationVerified?: boolean;
};

export const workspace = {
  title: "2026 1분기 실적 분석",
  documentCount: 8,
  question: "1분기 매출 감소의 주요 원인은 무엇인가요?",
  queryId: "q_20260720_0412",
};

export const citations: Citation[] = [
  {
    id: "1",
    claim: "아시아 수요 둔화 → 해외 매출 12% 감소",
    inlineLabel: "실적보고서 p.14",
    sourceFilename: "2026_1Q_실적보고서.pdf",
    sourceLabel: "실적보고서",
    page: 14,
    totalPages: 42,
    section: "3. 지역별 매출 분석",
    excerpt:
      "아시아 지역 매출은 환율 영향 및 수요 둔화로 전년 대비 12.0% 감소한 1,842억 원을 기록했습니다.",
    before:
      "국내 매출은 전년 동기와 유사한 수준(2,310억 원, -0.8%)을 유지하였으나, 해외 매출의 부진이 전체 실적 하락을 견인하였다.",
    highlighted:
      "아시아 지역 매출은 환율 영향 및 수요 둔화로 전년 대비 12.0% 감소한 1,842억 원을 기록하였다. 특히 주력 제품군의 판매량 감소가 두드러졌으며,",
    after:
      "이는 현지 경기 둔화와 경쟁 심화가 복합적으로 작용한 결과로 판단된다. 유럽 지역은 신규 유통 계약 효과로 소폭 성장(+2.4%)하였으나 전체 하락분을 상쇄하기에는 부족하였다.",
    chunkId: "ch_1q_014_02",
  },
  {
    id: "2",
    claim: "판가 반영 지연 → 이익률 3.1%p 하락",
    inlineLabel: "원가분석 메모 p.3",
    sourceFilename: "원가분석_내부메모_3월.docx",
    sourceLabel: "원가분석 메모",
    page: 3,
    totalPages: 9,
    section: "2. 원자재 가격과 판가 반영",
    excerpt:
      "원자재 단가 상승분의 판가 전가율은 42% 수준으로, 매출총이익률은 전년 동기 대비 3.1%p 하락했습니다.",
    before:
      "1분기 주요 원자재 평균 매입 단가는 전년 동기 대비 9.7% 상승하였다. 기존 계약 물량은 가격 조정 시점이 늦어 원가 상승분이 먼저 반영되었다.",
    highlighted:
      "원자재 단가 상승분의 판가 전가율은 42% 수준에 그쳤으며, 판가 반영 시차로 매출총이익률은 전년 동기 대비 3.1%p 하락하였다.",
    after:
      "2분기 신규 계약부터 순차적으로 가격 조정이 예정되어 있으나, 원자재 가격 변동에 따라 회복 속도는 달라질 수 있다.",
    chunkId: "ch_memo_003_01",
  },
  {
    id: "3",
    claim: "신제품 출시 연기 → 매출 45억 이월",
    inlineLabel: "출시일정 변경안 p.2",
    sourceFilename: "신제품_출시일정_변경안.pdf",
    sourceLabel: "출시일정 변경안",
    page: 2,
    totalPages: 6,
    section: "1. 일정 변경 영향",
    excerpt:
      "출시 일정 조정에 따라 1분기 인식 예정 매출 45억 원이 2분기로 이월될 예정입니다.",
    before:
      "품질 인증 보완 요청으로 신제품 양산 시작일을 3월 10일에서 4월 8일로 조정한다. 영업·생산·재무 부서가 변경 일정을 공동 검토하였다.",
    highlighted:
      "출시 일정 조정에 따라 1분기 인식 예정 매출 약 45억 원은 2분기로 이월될 예정이다.",
    after:
      "연간 판매 목표는 유지하되 2분기 공급 계획을 확대하고, 주요 고객사에는 변경된 납기 일정을 별도로 안내한다.",
    chunkId: "ch_plan_002_04",
  },
];

export const relatedDocuments = [
  { name: "3월_영업회의록.pdf", page: 5 },
  { name: "경쟁사_동향_리포트.pdf", page: 11 },
];

export const documents = [
  "2026_1Q_실적보고서",
  "원가분석_내부메모_3월",
  "신제품_출시일정_변경안",
  "3월_영업회의록",
  "경쟁사_동향_리포트",
  "환율_영향_분석",
  "제품군별_판매현황",
  "2분기_매출_전망",
];

export const traceRows: TraceRow[] = [
  {
    chunkId: "ch_1q_014_02",
    source: "실적보고서 p.14",
    bm25: 18.4,
    vector: 0.912,
    rerank: 0.94,
    status: "citation",
    citationId: "1",
    citationVerified: true,
  },
  {
    chunkId: "ch_memo_003_01",
    source: "원가분석 메모 p.3",
    bm25: 15.1,
    vector: 0.874,
    rerank: 0.89,
    status: "citation",
    citationId: "2",
    citationVerified: true,
  },
  {
    chunkId: "ch_plan_002_04",
    source: "출시일정 변경안 p.2",
    bm25: 13.8,
    vector: 0.861,
    rerank: 0.87,
    status: "citation",
    citationId: "3",
    citationVerified: true,
  },
  {
    chunkId: "ch_mtg_005_02",
    source: "영업회의록 p.5",
    bm25: 11.2,
    vector: 0.803,
    rerank: 0.81,
    status: "context",
  },
  {
    chunkId: "ch_cmp_011_01",
    source: "경쟁사 리포트 p.11",
    bm25: 10.6,
    vector: 0.781,
    rerank: 0.78,
    status: "context",
  },
  {
    chunkId: "ch_1q_022_01",
    source: "실적보고서 p.22",
    bm25: 12.9,
    vector: 0.822,
    rerank: null,
    status: "duplicate",
  },
  {
    chunkId: "ch_memo_004_03",
    source: "원가분석 메모 p.4",
    bm25: 9.8,
    vector: 0.745,
    rerank: null,
    status: "duplicate",
  },
  {
    chunkId: "ch_fx_007_01",
    source: "환율 영향 분석 p.7",
    bm25: 8.7,
    vector: 0.742,
    rerank: 0.64,
    status: "excluded",
  },
  {
    chunkId: "ch_sales_006_02",
    source: "제품군별 판매현황 p.6",
    bm25: 8.2,
    vector: 0.721,
    rerank: 0.59,
    status: "excluded",
  },
  {
    chunkId: "ch_q2_003_01",
    source: "2분기 매출 전망 p.3",
    bm25: 7.9,
    vector: 0.704,
    rerank: 0.55,
    status: "excluded",
  },
  {
    chunkId: "ch_mtg_009_03",
    source: "영업회의록 p.9",
    bm25: 6.8,
    vector: 0.682,
    rerank: 0.48,
    status: "excluded",
  },
  {
    chunkId: "ch_1q_031_02",
    source: "실적보고서 p.31",
    bm25: 6.1,
    vector: 0.651,
    rerank: 0.41,
    status: "excluded",
  },
];

export const traceDurations = [
  ["질문 분석", "0.18s"],
  ["하이브리드 검색", "0.72s"],
  ["rerank", "0.35s"],
  ["답변 생성", "1.16s"],
] as const;

export function getCitation(id: string) {
  return citations.find((citation) => citation.id === id) ?? citations[0];
}

export function isUsed(row: TraceRow) {
  return row.status === "citation" || row.status === "context";
}
