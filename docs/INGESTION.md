# 문서를 글자로 바꾸는 기능

## 이 기능이 하는 일

RAG가 문서에 답하려면 먼저 문서 안의 글자를 읽어야 합니다.

```text
파일 업로드 → 글자 추출 → 사람이 결과 확인 → 검색 색인에 저장
```

이번 단계는 앞의 세 칸, 즉 **업로드·추출·확인**까지 구현합니다. 업로드한
문서를 검색 색인에 자동 저장하는 기능은 다음 단계에서 연결합니다.

## 지원하는 파일

| 파일 | 읽는 방법 | 현재 제한 |
|---|---|---|
| 일반 PDF | PDF 안에 저장된 글자를 바로 추출 | 복잡한 표의 칸 구조는 흐트러질 수 있음 |
| 스캔 PDF | 글자가 거의 없는 페이지만 이미지로 바꿔 OCR | Tesseract 설치 필요 |
| DOCX | 문단과 표를 문서 순서대로 추출 | 문서 안 그림의 글자는 아직 OCR하지 않음 |
| PNG/JPG/TIFF/BMP | 이미지 전체를 OCR | Tesseract 설치 필요 |
| TXT/MD | UTF-8, CP949, EUC-KR 순서로 읽기 | 바이너리 파일은 지원하지 않음 |
| 옛날 Word `.doc` | 지원하지 않음 | Word에서 `.docx`로 다시 저장 |

파일 하나의 기본 크기 제한은 20MB이며 추출을 시작하기 전에 검사합니다.
암호로 잠긴 PDF는 암호를 해제한 사본을 올려야 합니다.

## 실행

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

브라우저에서 `1. 문서 넣기` 탭을 열고 파일을 올린 뒤 `텍스트 추출`을
누릅니다. 추출 방식, 글자 수, 경고, 결과 미리보기를 확인하고 전체 결과를
TXT로 받을 수 있습니다.

UI 없이 파일 하나만 추출할 수도 있습니다.

```powershell
python src\document_ingestion.py "C:\문서\공고문.pdf"
python src\document_ingestion.py "C:\문서\공고문.docx" -o "result.txt"
```

`docs/raw` 폴더의 PDF를 한꺼번에 `docs/text`로 옮기려면 다음 명령을
사용합니다.

```powershell
python src\extract_pdf.py
```

## 한국어 OCR 준비

`pytesseract`는 OCR 프로그램을 불러주는 Python 연결선입니다. 실제로 글자를
읽는 **Tesseract 5 실행 파일**과 `kor`, `eng` 언어 데이터는 따로
설치해야 합니다.

Windows에서는 다음을 확인합니다.

1. Tesseract 5를 설치합니다.
2. 설치 폴더의 `tessdata`에 `kor.traineddata`와 `eng.traineddata`가 있는지
   확인합니다.
3. `tesseract.exe`를 PATH에 넣거나 환경 변수로 정확한 경로를 알려줍니다.

```powershell
$env:TESSERACT_CMD = "C:\Program Files\Tesseract-OCR\tesseract.exe"
tesseract --list-langs
```

목록에 `kor`와 `eng`가 모두 보이면 한국어+영어 OCR 준비가 끝난 것입니다.
앱의 `추출 설정`에서도 준비 상태를 바로 확인할 수 있습니다.

## 코드에서 재사용하기

업로드 UI와 추출 코드는 분리되어 있습니다. 나중에 FastAPI, LangChain,
LangGraph에서도 같은 추출 함수를 그대로 호출할 수 있습니다.

```python
from src.document_ingestion import extract_document

result = extract_document("notice.pdf", uploaded_bytes)

print(result.text)
print(result.pages[0].method)  # native, ocr, docx, plain, empty
print(result.source_sha256)    # 같은 파일인지 확인하는 값
print(result.warnings)
```

PDF는 먼저 저장된 글자를 읽습니다. 글자가 거의 없는 페이지만 OCR하기 때문에
일반 PDF까지 매번 느린 OCR을 돌리지 않습니다.
OCR은 여러 줄로 이루어진 한국어 문서를 위에서 아래로 읽는 설정을 기본으로
사용합니다.

## 이번 단계의 완료 기준

- 일반 PDF의 저장된 글자가 추출된다.
- 스캔 PDF와 이미지는 OCR 경로를 사용한다.
- DOCX의 문단과 표가 추출된다.
- 결과에 파일명, 페이지, 추출 방식, 경고가 남는다.
- 화면에서 여러 파일을 올리고 결과를 미리 볼 수 있다.
- OCR이 설치되지 않아도 앱이 꺼지지 않고 해결 방법을 알려준다.

아직 하지 않는 일은 검색용 chunk 생성, ChromaDB 저장, 중복 문서 처리,
업로드 문서 삭제입니다. 이것들은 추출 결과를 눈으로 검증한 뒤 다음 단계에서
연결합니다.
