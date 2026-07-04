from pathlib import Path
import re


TEXT_DIR = Path("docs/text")

# ===== PAGE 1 ===== 같은 줄 제거용
PAGE_MARKER_RE = re.compile(r"^===== PAGE \d+ =====$")

# - 1 - 같은 페이지 번호 제거용
PAGE_NUMBER_RE = re.compile(r"^\s*-\s*\d+\s*-\s*$")

# Ⅰ  사업개요 / Ⅱ  주요 수행업무 같은 로마숫자 제목
ROMAN_HEADING_RE = re.compile(r"^\s*[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]\s+.+$")

# 1. 관련 근거 / 4. 참여 가능 기관 / 10. 고등교육법...
NUMBER_HEADING_RE = re.compile(r"^\s*\d+\.\s+.+$")


def load_text(file_path: Path) -> str:
    return file_path.read_text(encoding="utf-8")


def clean_basic_text(text: str) -> str:
    """
    chunking 전에 아주 기본적인 노이즈만 제거한다.

    지금 제거하는 것:
    1. ===== PAGE 1 ===== 같은 페이지 구분선
    2. - 1 - 같은 페이지 번호
    3. 줄 오른쪽 공백

    아직 하지 않는 것:
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
    줄 앞에 공백이 몇 개 있는지 센다.

    예:
    '4. 참여 가능 기관'       -> 0
    '  1. AIㆍ정보통신...'   -> 2
    """
    return len(line) - len(line.lstrip(" "))


def is_roman_heading(line: str) -> bool:
    return bool(ROMAN_HEADING_RE.match(line))


def is_number_heading(line: str) -> bool:
    return bool(NUMBER_HEADING_RE.match(line))


def is_main_number_heading(line: str) -> bool:
    """
    chunk 시작점이 될 수 있는 숫자 제목인지 판단한다.

    예:
    1. 관련 근거
    2. 선정 규모
    3. 모집 절차
    4. 참여 가능 기관
    5. 자격 제한

    기준:
    - 숫자. 형태여야 함
    - 줄 앞 공백이 거의 없어야 함
    """
    if not is_number_heading(line):
        return False

    leading_spaces = count_leading_spaces(line)

    return leading_spaces == 0


def is_number_list_item(line: str) -> bool:
    """
    제목 아래에 딸린 번호 목록인지 판단한다.

    예:
      1. AIㆍ정보통신ㆍ스마트제조...
      2. 지방자치단체가 설립한...
      3. 국가인적자원개발컨소시엄...

    기준:
    - 숫자. 형태여야 함
    - 줄 앞 공백이 있어야 함
    """
    if not is_number_heading(line):
        return False

    leading_spaces = count_leading_spaces(line)

    return leading_spaces > 0


def print_heading_candidates(text: str) -> None:
    """
    아직 chunk를 만들지 않는다.
    먼저 각 줄이 어떤 역할로 분류되는지만 출력한다.
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

        elif is_main_number_heading(line):
            print(
                f"[MAIN_NUMBER_HEADING] line {line_number} "
                f"spaces={leading_spaces}: {line.strip()}"
            )

        elif is_number_list_item(line):
            print(
                f"[NUMBER_LIST_ITEM] line {line_number} "
                f"spaces={leading_spaces}: {line.strip()}"
            )


def main():
    txt_files = list(TEXT_DIR.glob("*.txt"))

    if not txt_files:
        print("docs/text 폴더에 txt 파일이 없습니다.")
        return

    file_path = txt_files[0]

    print(f"사용 파일: {file_path}")

    text = load_text(file_path)
    text = clean_basic_text(text)

    print_heading_candidates(text)


if __name__ == "__main__":
    main()