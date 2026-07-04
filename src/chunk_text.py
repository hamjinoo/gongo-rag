from pathlib import Path
import re


TEXT_DIR = Path("docs/text")
EXPERIMENTS_DIR = Path("experiments")
OUTPUT_PATH = EXPERIMENTS_DIR / "chunks_v0.md"


# ===== PAGE 1 ===== 같은 페이지 구분선 제거용
PAGE_MARKER_RE = re.compile(r"^===== PAGE \d+ =====$")

# - 1 - 같은 페이지 번호 제거용
PAGE_NUMBER_RE = re.compile(r"^\s*-\s*\d+\s*-\s*$")

# Ⅰ 사업개요 / Ⅱ 주요 수행업무 같은 로마숫자 제목
ROMAN_HEADING_RE = re.compile(r"^\s*[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]\s+.+$")

# 1. 관련 근거 / 4. 참여 가능 기관 같은 숫자 제목 또는 번호 목록
NUMBER_HEADING_RE = re.compile(r"^\s*\d+\.\s+.+$")

# ① / ② / ③ 같은 원형 숫자 시작
CIRCLED_START_RE = re.compile(r"^\s*[①②③④⑤⑥⑦⑧⑨⑩]\s*")


def load_text(file_path: Path) -> str:
    return file_path.read_text(encoding="utf-8")


def normalize_spaces(text: str) -> str:
    """
    heading_path에 들어갈 제목의 공백만 정리한다.
    본문 전체에는 함부로 적용하지 않는다.
    """
    return re.sub(r"\s+", " ", text).strip()


def clean_basic_text(text: str) -> str:
    """
    chunking 전에 아주 기본적인 노이즈만 제거한다.

    제거:
    - ===== PAGE 1 =====
    - - 1 -
    - 줄 오른쪽 공백

    아직 하지 않음:
    - 표 복원
    - 줄바꿈 자동 합치기
    - 문장 재구성
    """
    lines = text.splitlines()
    cleaned_lines = []

    for line in lines:
        stripped = line.strip()

        if PAGE_MARKER_RE.match(stripped):
            continue

        if PAGE_NUMBER_RE.match(stripped):
            continue

        cleaned_lines.append(line.rstrip())

    return "\n".join(cleaned_lines)


def count_leading_spaces(line: str) -> int:
    """
    줄 앞 공백 개수를 센다.

    예:
    '4. 참여 가능 기관'       -> 0
    '  1. AIㆍ정보통신...'   -> 2
    """
    return len(line) - len(line.lstrip(" "))


def is_roman_heading(line: str) -> bool:
    """
    Ⅰ 사업개요
    Ⅱ 중소기업AI훈련확산센터 주요 수행업무
    같은 큰 섹션 제목인지 판단한다.
    """
    return bool(ROMAN_HEADING_RE.match(line))


def is_number_heading(line: str) -> bool:
    """
    숫자. 텍스트 형태인지 판단한다.

    예:
    1. 관련 근거
    4. 참여 가능 기관
      1. AIㆍ정보통신...
    """
    return bool(NUMBER_HEADING_RE.match(line))


def is_main_number_heading(line: str) -> bool:
    """
    chunk 시작점이 될 수 있는 숫자 제목인지 판단한다.

    현재 v0 기준:
    - 앞 공백이 0개면 main number heading
    - 앞 공백이 있으면 목록 item

    예:
    4. 참여 가능 기관 -> main heading
      1. AIㆍ정보통신 -> list item
    """
    if not is_number_heading(line):
        return False

    return count_leading_spaces(line) == 0


def is_number_list_item(line: str) -> bool:
    """
    제목 아래에 딸린 번호 목록인지 판단한다.

    예:
      1. AIㆍ정보통신ㆍ스마트제조...
      2. 지방자치단체가 설립한...
      10. 고등교육법...
    """
    if not is_number_heading(line):
        return False

    return count_leading_spaces(line) > 0


def is_topic_heading(line: str) -> bool:
    """
     AI 훈련 확산 지원
     AI훈련코치 선발·관리
     지역 AI 확산 활동
    같은 진짜 중간 섹션 제목인지 판단한다.

    단, 성과 목표 표 안의 행은 제외한다.

    예외 처리:
    - ' AI 훈련 확산 지원      -      -'
    - ' 공단 주치의와 협업, 신규 참여기업 발굴    80개소   -'
    이런 줄은 표의 행이므로 topic heading으로 보지 않는다.
    """
    stripped = line.strip()

    if not stripped.startswith(""):
        return False

    # 표 안의 행은 열 간격 때문에 공백이 여러 개 연속으로 들어간다.
    # 진짜 제목은 보통 단일 공백만 있다.
    if re.search(r"\s{2,}", stripped):
        return False

    return True


def split_circled_heading_line(line: str):
    """
    원형 숫자 제목 줄을 heading과 body로 나눈다.

    예:
    입력:
    ② (문제해결형(PBL) AI+직무 훈련과정 개발) AI훈련코치가 기업의

    출력:
    heading:
    ② (문제해결형(PBL) AI+직무 훈련과정 개발)

    rest:
    AI훈련코치가 기업의

    중요한 점:
    제목 안에 (PBL)처럼 괄호가 한 번 더 들어갈 수 있으므로
    괄호 깊이를 세면서 닫히는 지점을 찾는다.
    """
    stripped = line.strip()

    match = CIRCLED_START_RE.match(stripped)

    if not match:
        return None

    start_index = match.end()

    if start_index >= len(stripped):
        return None

    if stripped[start_index] != "(":
        return None

    depth = 0

    for index in range(start_index, len(stripped)):
        char = stripped[index]

        if char == "(":
            depth += 1

        elif char == ")":
            depth -= 1

            if depth == 0:
                heading = stripped[: index + 1].strip()
                rest = stripped[index + 1 :].strip()
                return heading, rest

    return stripped, ""


def is_main_circled_heading(line: str) -> bool:
    """
    ① (기업 내 AI도입을 위한 훈련로드맵 수립)
    ② (문제해결형(PBL) AI+직무 훈련과정 개발)
    같은 원형 숫자 제목인지 판단한다.

    현재 v0 기준:
    - 앞 공백이 0개여야 한다.
    - 원형 숫자 뒤에 괄호 제목이 있어야 한다.
    """
    if count_leading_spaces(line) != 0:
        return False

    return split_circled_heading_line(line) is not None


def print_heading_candidates(text: str) -> None:
    """
    디버깅용 함수.
    각 줄이 어떤 heading/list item으로 분류되는지 확인한다.
    """
    lines = text.splitlines()

    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue

        leading_spaces = count_leading_spaces(line)

        if is_roman_heading(line):
            print(
                f"[ROMAN_HEADING] line {line_number} "
                f"spaces={leading_spaces}: {line.strip()}"
            )

        elif is_topic_heading(line):
            print(
                f"[TOPIC_HEADING] line {line_number} "
                f"spaces={leading_spaces}: {line.strip()}"
            )

        elif is_main_number_heading(line):
            print(
                f"[MAIN_NUMBER_HEADING] line {line_number} "
                f"spaces={leading_spaces}: {line.strip()}"
            )

        elif is_main_circled_heading(line):
            heading, rest = split_circled_heading_line(line)

            print(
                f"[MAIN_CIRCLED_HEADING] line {line_number} "
                f"spaces={leading_spaces}: {heading}"
            )

            if rest:
                print(f"  [REST] {rest}")

        elif is_number_list_item(line):
            print(
                f"[NUMBER_LIST_ITEM] line {line_number} "
                f"spaces={leading_spaces}: {line.strip()}"
            )


def build_chunks(text: str) -> list[dict]:
    """
    문서 텍스트를 heading 구조 기준으로 chunk로 나눈다.

    v0 규칙:
    1. 로마숫자 제목은 큰 섹션으로 기억한다.
    2.  제목은 중간 섹션으로 기억한다.
    3. 숫자 제목은 chunk 시작점으로 본다.
    4. 원형 숫자 제목도 chunk 시작점으로 본다.
    5. 번호 목록, bullet, 일반 문장은 현재 chunk 본문에 포함한다.
    6. 원형 숫자 제목 뒤에 같은 줄로 붙은 본문은 heading이 아니라 text에 넣는다.
    7. 로마숫자 제목이 바뀌면 이전 chunk를 먼저 저장한다.
    """
    lines = text.splitlines()

    chunks = []

    current_roman_heading = None
    current_topic_heading = None
    current_chunk_heading = None
    current_lines = []

    def make_heading_path() -> str:
        parts = [
            current_roman_heading,
            current_topic_heading,
            current_chunk_heading,
        ]

        cleaned_parts = []

        for part in parts:
            if part:
                cleaned_parts.append(normalize_spaces(part))

        return " > ".join(cleaned_parts)

    def save_current_chunk():
        nonlocal current_chunk_heading
        nonlocal current_lines

        if current_chunk_heading is None:
            return

        chunk_text = "\n".join(current_lines).strip()

        if not chunk_text:
            current_chunk_heading = None
            current_lines = []
            return

        chunks.append(
            {
                "heading_path": make_heading_path(),
                "text": chunk_text,
            }
        )

        current_chunk_heading = None
        current_lines = []

    def start_section_intro_if_needed():
        """
        Ⅳ 선정 평가지표, Ⅴ 기타사항처럼
        숫자 하위 제목이 없는 섹션도 chunk로 남기기 위한 장치.

        예:
        Ⅳ 선정 평가지표 > 섹션 개요
        Ⅴ 기타사항 > 섹션 개요
        """
        nonlocal current_chunk_heading
        nonlocal current_lines

        if current_chunk_heading is None and current_roman_heading is not None:
            current_chunk_heading = "섹션 개요"
            current_lines = []

    for line in lines:
        stripped = line.strip()

        if not stripped:
            continue

        # Ⅰ, Ⅱ, Ⅲ 같은 큰 제목
        # 큰 제목이 바뀌면 이전 chunk를 먼저 저장해야 한다.
        if is_roman_heading(line):
            save_current_chunk()
            current_roman_heading = stripped
            current_topic_heading = None
            current_chunk_heading = None
            current_lines = []
            continue

        #  AI 훈련 확산 지원 같은 중간 제목
        if is_topic_heading(line):
            save_current_chunk()
            current_topic_heading = stripped
            current_chunk_heading = None
            current_lines = []
            continue

        # 1. 관련 근거 / 2. 선정 규모 같은 숫자 제목
        if is_main_number_heading(line):
            save_current_chunk()
            current_chunk_heading = stripped
            current_lines = []
            continue

        # ① (기업 내 AI도입...) 같은 원형 숫자 제목
        if is_main_circled_heading(line):
            save_current_chunk()

            result = split_circled_heading_line(line)

            if result is None:
                current_chunk_heading = stripped
                current_lines = []
                continue

            heading, rest = result

            current_chunk_heading = heading
            current_lines = []

            if rest:
                current_lines.append(rest)

            continue

        # 여기까지 오면 일반 본문, bullet, 번호 목록이다.
        # 현재 chunk가 있으면 그 chunk에 넣는다.
        if current_chunk_heading is not None:
            current_lines.append(line)
            continue

        # 현재 chunk가 없는데 본문이 나오면
        # 로마 섹션 자체의 개요 chunk로 저장한다.
        # 예: Ⅳ 선정 평가지표, Ⅴ 기타사항
        start_section_intro_if_needed()

        if current_chunk_heading is not None:
            current_lines.append(line)

    save_current_chunk()

    return chunks


def print_chunks(chunks: list[dict]) -> None:
    print(f"생성된 chunk 수: {len(chunks)}")

    for index, chunk in enumerate(chunks, start=1):
        print("=" * 80)
        print(f"CHUNK {index:03d}")
        print(chunk["heading_path"])
        print("-" * 80)
        print(chunk["text"])


def save_chunks_to_markdown(chunks: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("# Auto Chunks v0")
    lines.append("")

    for index, chunk in enumerate(chunks, start=1):
        lines.append(f"## CHUNK {index:03d}")
        lines.append("")
        lines.append("### heading_path")
        lines.append("")
        lines.append(chunk["heading_path"])
        lines.append("")
        lines.append("### text")
        lines.append("")
        lines.append("```text")
        lines.append(chunk["text"])
        lines.append("```")
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    txt_files = list(TEXT_DIR.glob("*.txt"))

    if not txt_files:
        print("docs/text 폴더에 txt 파일이 없습니다.")
        return

    file_path = txt_files[0]

    print(f"사용 파일: {file_path}")

    text = load_text(file_path)
    text = clean_basic_text(text)

    chunks = build_chunks(text)

    print_chunks(chunks)
    save_chunks_to_markdown(chunks, OUTPUT_PATH)

    print(f"\n저장 완료: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()