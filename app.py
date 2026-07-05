"""
app.py — 데모 UI (Streamlit)  [✅ 배관 뼈대 — 계획서: "Streamlit UI 뼈대"는 맡겨도 됨]

⚠️ 11주차 전에 이 파일을 열지 마세요. UI는 이틀 이상 금지 (계획서 금지사항 4).
   데모의 전부: 질문 입력 → 답변 → 근거 chunk 표시. 그 이상 꾸미지 않는다.

실행:
    pip install streamlit
    streamlit run app.py
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from chunker import chunk_fixed          # noqa: E402
from bm25 import BM25                     # noqa: E402
import rag_answer                         # noqa: E402

TEXT_DIR = Path(__file__).resolve().parent / "docs" / "text"


@st.cache_resource
def build_index():
    """문서 로드 + chunking + BM25 색인 (앱 시작 시 1회만)."""
    chunks = []
    for f in sorted(TEXT_DIR.glob("*.txt")):
        chunks += chunk_fixed(f.read_text(encoding="utf-8"), doc_id=f.stem)
    return chunks, BM25([c["text"] for c in chunks])


st.title("📄 gongo-rag — 공고문 질문 답변 데모")
st.caption("한국어 공공문서 RAG · 근거 인용 · 정보 없음 처리")

chunks, bm25 = build_index()
st.sidebar.markdown(f"**색인 현황**\n\n- chunk 수: {len(chunks)}\n- 검색: BM25 (top-3)")

question = st.text_input("질문을 입력하세요", placeholder="예: 신청 자격이 어떻게 되나요?")

if question:
    top = bm25.search(question, k=3)
    retrieved = [chunks[i]["text"] for i, _ in top]

    with st.spinner("답변 생성 중..."):
        try:
            ans = rag_answer.answer(question, retrieved)
        except Exception as e:
            ans = f"(생성 실패: {e} — API 키 확인)"

    st.subheader("답변")
    st.write(ans)

    st.subheader("근거 chunk")
    for rank, (i, score) in enumerate(top, 1):
        with st.expander(f"[근거 {rank}] {chunks[i]['id']} (score {score:.2f})"):
            st.text(chunks[i]["text"])
