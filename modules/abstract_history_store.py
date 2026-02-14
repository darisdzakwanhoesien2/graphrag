import json
from pathlib import Path
from datetime import datetime

BASE_DIR = Path("data/processed/abstract")
BASE_DIR.mkdir(parents=True, exist_ok=True)

HISTORY_FILE = BASE_DIR / "query_history.json"


def load_history():
    if HISTORY_FILE.exists():
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    return []


def save_history(history):
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)


def append_history(entry):
    history = load_history()
    history.append(entry)
    save_history(history)
