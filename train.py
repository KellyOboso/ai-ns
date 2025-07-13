import os
import sqlite3
import yaml
from pathlib import Path

DB_PATH = "mental_health_chatbot.db"

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS bot_responses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_input TEXT,
                bot_response TEXT,
                source_file TEXT
            )
        """)

def clear_data():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM bot_responses")
        conn.commit()

def train():
    init_db()
    clear_data()
    data_dir = Path("data")
    if not data_dir.exists():
        print("❌ data/ folder not found.")
        return

    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        for file in data_dir.glob("*.yml"):
            with open(file, "r", encoding="utf-8") as f:
                try:
                    data = yaml.safe_load(f)

                    # ✅ NEW: If top-level is dict, use `responses` key.
                    if isinstance(data, dict) and "responses" in data:
                        responses = data["responses"]
                    elif isinstance(data, list):
                        responses = data
                    else:
                        print(f"❌ {file} does not contain a list or 'responses' key.")
                        continue

                    if not isinstance(responses, list):
                        print(f"❌ 'responses' in {file} is not a list.")
                        continue

                    for pair in responses:
                        if isinstance(pair, list) and len(pair) == 2:
                            input_text, response_text = pair
                            if isinstance(input_text, str) and isinstance(response_text, str):
                                cursor.execute(
                                    "INSERT INTO bot_responses (user_input, bot_response, source_file) VALUES (?, ?, ?)",
                                    (input_text.strip().lower(), response_text.strip(), str(file))
                                )
                            else:
                                print(f"❌ Skipped non-string pair in {file}: {pair}")
                        else:
                            print(f"❌ Bad format in {file}: {pair}")

                except Exception as e:
                    print(f"❌ Failed to parse {file}: {e}")

        conn.commit()
    print("✅ Training complete. All bot responses updated from data/")

if __name__ == "__main__":
    train()
