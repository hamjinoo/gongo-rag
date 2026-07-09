"""
chunker.py — 텍스트를 검색 단위 조각(chunk)으로 자르기  [✍️ Week 2: 직접 구현]

먼저 읽기: ../01-basics/03-RAG-기초.md 3번 (왜 자르는가)
감 잡기:   놀이터.html의 ✂️ Chunking 위젯 — size/overlap 트레이드오프를 눈으로 먼저
자가 채점:  python tests\\test_chunker.py
GPT 규칙:  이 파일의 TODO를 통째로 구현시키는 것 금지 (WORKFLOW.md).
           내가 짠 코드를 보여주고 리뷰받는 것은 OK.

chunk의 표현 (이 형식을 지켜야 테스트와 다른 모듈이 동작합니다):
    {"id": "doc1-0", "text": "조각 내용...", "start": 0}
      - id:    "{doc_id}-{순번}"
      - start: 원문에서 이 조각이 시작되는 문자 인덱스 (고정 크기 방식에서)
"""


def chunk_fixed(text: str, doc_id: str = "doc", chunk_size: int = 500, overlap: int = 100) -> list[dict]:
    """방식 1: 고정 크기 + 겹침(overlap) 슬라이딩 윈도우.

    overlap이 필요한 이유: 정답 문장이 하필 조각 경계에서 반토막 나는 사고를
    줄이기 위해, 이웃 조각끼리 끝/처음 일부를 공유하게 한다.

    TODO(직접 구현) — 단계 힌트:
      1. 시작 위치 start를 0부터 (chunk_size - overlap) 간격으로 전진시킨다.
         (예: size=500, overlap=100 → start = 0, 400, 800, ...)
      2. 각 start에서 text[start : start + chunk_size]를 잘라 chunk dict를 만든다.
      3. 마지막 조각: 텍스트 끝을 넘어가면 짧게 잘리는데, 그대로 두면 된다.
         단, 빈 문자열 조각은 만들지 말 것.
      4. 반환: chunk dict의 리스트. id는 f"{doc_id}-{i}".

    자가 검증 아이디어: 모든 조각의 (start, start+len(text))를 이으면
    원문 전체가 빠짐없이 덮여야 한다. (테스트가 이걸 확인함)
    """
    # TODO: 여기에 직접 구현
    text_data = []
    for start in range(0, len(text), chunk_size - overlap):
        split_text = text[start:start + chunk_size]

        chunk = {
            "id": f"{doc_id}-{len(text_data)}",
            "text": split_text,
            "start": start
        }

        text_data.append(chunk)

    return text_data

    raise NotImplementedError("chunk_fixed를 직접 구현하세요 (힌트는 docstring에)")


def chunk_by_paragraph(text: str, doc_id: str = "doc", max_chars: int = 800) -> list[dict]:
    """방식 2: 문단(빈 줄) 기반. 의미 단위를 존중하는 방식.

    TODO(직접 구현) — 단계 힌트:
      1. 빈 줄 기준으로 문단을 나눈다.  힌트: text.split("\\n\\n")
         (관찰노트에서 봤듯 공고문의 문단 구분이 지저분할 수 있음 — 완벽 추구 금지)
      2. 각 문단의 앞뒤 공백을 정리하고, 빈 문단은 버린다.
      3. 너무 짧은 문단이 연달아 나오면 max_chars를 넘지 않는 선에서 합친다.
         (제목 한 줄 + 본문이 따로 놀지 않게)
      4. max_chars를 크게 넘는 문단은 그 안에서 다시 강제로 자른다.
         힌트: 이미 만든 chunk_fixed를 재사용할 수 있다!
      5. id는 f"{doc_id}-{i}", start는 이 방식에선 -1로 채워도 된다(추적 어려움).

    생각해볼 것(노트에 기록): 표가 깨진 텍스트에서 이 방식은 어떤 사고를 칠까?
    """
    # TODO: 여기에 직접 구현
    text_data = text.split("\n\n")
    chunk_paragraph = []

    for idx, paragraph in enumerate(text_data):
        clear_text = paragraph.strip()
        if clear_text == "":
            continue

        if len(clear_text) > max_chars:
            chunk_paragraph.extend(chunk_fixed(clear_text, chunk_size=max_chars, overlap=100))
        else:
            chunk = {
                "id": f"{doc_id}-{idx}",
                "text" : clear_text,
                "start": idx
            }
            
            chunk_paragraph.append(chunk)
    return chunk_paragraph




    raise NotImplementedError("chunk_by_paragraph를 직접 구현하세요")


# ──────────────────────────────────────────────────────────────
# 아래는 완성 배관: 내 구현을 실제 공고문에 돌려보는 데모
# 실행: 02-gongo-rag 폴더에서  python src\chunker.py
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from pathlib import Path

    text_dir = Path(__file__).resolve().parents[1] / "docs" / "text"
    txt_files = sorted(text_dir.glob("*.txt"))
    if not txt_files:
        print("docs/text/ 에 텍스트가 없습니다. 먼저 python src\\extract_pdf.py 를 실행하세요.")
        raise SystemExit(0)

    text = txt_files[0].read_text(encoding="utf-8")
    print(f"문서: {txt_files[0].name} ({len(text):,}자)\n")

    try:
        for name, chunks in [
            ("고정 크기(500/100)", chunk_fixed(text, chunk_size=500, overlap=100)),
            ("문단 기반(800)", chunk_by_paragraph(text, max_chars=800)),
        ]:
            sizes = [len(c["text"]) for c in chunks]
            print(f"[{name}] 조각 {len(chunks)}개, 평균 {sum(sizes)//max(len(sizes),1)}자, "
                  f"최소 {min(sizes)}자, 최대 {max(sizes)}자")
            print(f"  첫 조각 미리보기: {chunks[0]['text'][:80]!r}\n")
        print("→ 두 방식의 조각을 몇 개 열어 읽어보고, 어느 쪽이 '질문에 답할 수 있는 단위'인지 노트에 기록하세요.")
    except NotImplementedError as e:
        print(f"아직 구현 전: {e}")
