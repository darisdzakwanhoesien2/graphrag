import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE_PATH = PROJECT_ROOT / "data" / "abstract"


def get_abstract_folders():
    if not BASE_PATH.exists():
        return []
    return [f.name for f in BASE_PATH.iterdir() if f.is_dir()]


def load_abstract_folder(folder_name):

    folder_path = BASE_PATH / folder_name
    documents = []

    csv_files = list(folder_path.glob("*.csv"))

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)

        for _, row in df.iterrows():

            abstract_text = str(row.get("Abstract", ""))
            title = str(row.get("Title", ""))

            if abstract_text.strip() == "":
                continue

            documents.append({
                "doc_id": folder_name,
                "title": title,
                "doi": row.get("DOI", ""),
                "authors": row.get("Authors", ""),
                "journal": row.get("Journal", ""),
                "year": row.get("Year", ""),
                "text": f"{title}\n\n{abstract_text}"
            })

    return documents
