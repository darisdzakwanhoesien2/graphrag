from pathlib import Path

# Always resolve from project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE_PATH = PROJECT_ROOT / "data" / "ocr_output"


def get_report_folders():
    if not BASE_PATH.exists():
        print("OCR base path does not exist:", BASE_PATH)
        return []

    return [f.name for f in BASE_PATH.iterdir() if f.is_dir()]


def load_report(report_name):
    report_path = BASE_PATH / report_name
    pages_path = report_path / "pages"

    documents = []

    print("Looking inside:", pages_path)

    if not pages_path.exists():
        print("Pages folder not found:", pages_path)
        return documents

    page_files = sorted(
        pages_path.glob("page_*.md"),
        key=lambda x: int(x.stem.split("_")[1])
    )

    print("Found page files:", len(page_files))

    for page_file in page_files:
        with open(page_file, "r", encoding="utf-8") as f:
            documents.append({
                "doc_id": report_name,
                "page": int(page_file.stem.split("_")[1]),
                "text": f.read()
            })

    return documents