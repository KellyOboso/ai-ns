# setup.py
from app import init_db, seed_bot_responses

if __name__ == "__main__":
    init_db()
    seed_bot_responses()
    print("✅ Database initialized and seeded.")
