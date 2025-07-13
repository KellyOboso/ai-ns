import streamlit as st
import sqlite3
import bcrypt
from pathlib import Path
import logging
import re

logging.getLogger('watchdog').setLevel(logging.WARNING)

DB_PATH = Path("mental_health_chatbot.db")

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("MentalHealthBot")


@st.cache_resource
def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE,
                email TEXT UNIQUE,
                password TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            );
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER,
                sender TEXT,
                content TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id)
            );
            CREATE TABLE IF NOT EXISTS bot_responses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_input TEXT,
                bot_response TEXT,
                source_file TEXT
            );
        """)
    logger.info("Database initialized.")


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def check_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode(), hashed.encode())


def register_user(username, email, password):
    try:
        hashed_pw = hash_password(password)
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO users (username, email, password) VALUES (?, ?, ?)",
                (username.strip(), email.strip(), hashed_pw)
            )
            conn.commit()
        return True, "Registration successful!"
    except sqlite3.IntegrityError:
        return False, "Username or email already exists."
    except Exception as e:
        logger.error(f"Registration failed: {e}")
        return False, "Error during registration."


def login_user(username, password):
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, username, email, password FROM users WHERE username = ? OR email = ?",
                (username.strip(), username.strip())
            )
            row = cursor.fetchone()
            if row and check_password(password, row[3]):
                st.session_state.logged_in = True
                st.session_state.current_user = {"id": row[0], "username": row[1], "email": row[2]}
                return True, f"Welcome back, {row[1]}!"
            else:
                return False, "Invalid credentials."
    except Exception as e:
        logger.error(f"Login failed: {e}")
        return False, "Error during login."


def start_new_conversation():
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO conversations (user_id) VALUES (?)",
            (st.session_state.current_user['id'],)
        )
        conn.commit()
        return cursor.lastrowid


def save_message(conv_id, sender, content):
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO messages (conversation_id, sender, content) VALUES (?, ?, ?)",
            (conv_id, sender, content.strip())
        )
        conn.commit()


def get_conversation_history():
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, created_at FROM conversations WHERE user_id = ? ORDER BY created_at DESC",
            (st.session_state.current_user['id'],)
        )
        return cursor.fetchall()


def load_conversation(conv_id):
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT sender, content FROM messages WHERE conversation_id = ? "
            "AND conversation_id IN (SELECT id FROM conversations WHERE user_id = ?)",
            (conv_id, st.session_state.current_user['id'])
        )
        rows = cursor.fetchall()
        st.session_state.messages = [{"sender": row[0], "content": row[1]} for row in rows]


def delete_conversation(conv_id):
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "DELETE FROM messages WHERE conversation_id = ? AND conversation_id IN (SELECT id FROM conversations WHERE user_id = ?)",
            (conv_id, st.session_state.current_user['id'])
        )
        cursor.execute(
            "DELETE FROM conversations WHERE id = ? AND user_id = ?",
            (conv_id, st.session_state.current_user['id'])
        )
        conn.commit()


def normalize(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = text.replace("im", "i'm")
    text = text.replace("i am", "i'm")
    return text


def get_bot_response(user_input):
    text = normalize(user_input)
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT bot_response FROM bot_responses WHERE user_input = ?", (text,))
        row = cursor.fetchone()
        if row:
            return row[0]

        keywords = text.split()
        for word in keywords:
            cursor.execute("SELECT bot_response FROM bot_responses WHERE user_input LIKE ?", (f"%{word}%",))
            row = cursor.fetchone()
            if row:
                return row[0]

    return "I’m here to listen. Could you tell me more about what’s on your mind?"


def show_chat():
    st.title("Mental Health Chatbot")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    if 'conversation_id' not in st.session_state or st.session_state.conversation_id is None:
        st.session_state.conversation_id = start_new_conversation()
        greeting = "Hi there! How can I support you today?"
        st.session_state.messages.append({"sender": "bot", "content": greeting})
        save_message(st.session_state.conversation_id, "bot", greeting)

    for m in st.session_state.messages:
        st.write(f"**{'You' if m['sender']=='user' else 'Bot'}:** {m['content']}")

    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_input("Your message:")
        submitted = st.form_submit_button("Send")
        if submitted and user_input.strip():
            st.session_state.messages.append({"sender": "user", "content": user_input})
            save_message(st.session_state.conversation_id, "user", user_input)

            response = get_bot_response(user_input)
            st.session_state.messages.append({"sender": "bot", "content": response})
            save_message(st.session_state.conversation_id, "bot", response)

            st.rerun()


def show_login():
    st.title("Login")
    username = st.text_input("Username or Email")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        ok, msg = login_user(username, password)
        if ok:
            st.success(msg)
            st.session_state.show_login = False
            st.rerun()
        else:
            st.error(msg)


def show_register():
    st.title("Register")
    username = st.text_input("Username")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    confirm = st.text_input("Confirm Password", type="password")
    if st.button("Register"):
        if password != confirm:
            st.error("Passwords do not match.")
        else:
            ok, msg = register_user(username, email, password)
            if ok:
                st.success(msg)
                st.session_state.show_register = False
                st.session_state.show_login = True
                st.rerun()
            else:
                st.error(msg)


def main():
    init_db()

    for k in ["logged_in", "current_user", "show_login", "show_register"]:
        if k not in st.session_state:
            st.session_state[k] = False if k == "logged_in" else None

    with st.sidebar:
        top = st.container()
        mid = st.container()

        with top:
            st.markdown(
                f"### 👤 {st.session_state.current_user['username']}"
                if st.session_state.logged_in and st.session_state.current_user else "### Please log in"
            )
            if st.session_state.logged_in:
                if st.button("🚪 Logout"):
                    for k in ["messages", "logged_in", "current_user", "conversation_id"]:
                        st.session_state[k] = None if k != "logged_in" else False
                    st.rerun()
            else:
                if st.button("Login"):
                    st.session_state.show_login = True
                if st.button("Register"):
                    st.session_state.show_register = True

        with mid:
            st.markdown("### 📜 Chat History")
            chat_history = []
            if st.session_state.logged_in and st.session_state.current_user:
                chat_history = get_conversation_history()
                if chat_history:
                    history_container = st.container()
                    for cid, created_at in chat_history:
                        cols = history_container.columns([3, 1])
                        if cols[0].button(f"{created_at[:19]}", key=f"conv_{cid}"):
                            st.session_state.conversation_id = cid
                            load_conversation(cid)
                            st.rerun()
                        if cols[1].button("🗑️", key=f"del_{cid}"):
                            delete_conversation(cid)
                            st.session_state.messages = []
                            st.session_state.conversation_id = None
                            st.rerun()
                    if st.button("🗑️ Clear All History"):
                        with sqlite3.connect(DB_PATH) as conn:
                            cursor = conn.cursor()
                            cursor.execute(
                                "DELETE FROM messages WHERE conversation_id IN "
                                "(SELECT id FROM conversations WHERE user_id = ?)",
                                (st.session_state.current_user['id'],)
                            )
                            cursor.execute(
                                "DELETE FROM conversations WHERE user_id = ?",
                                (st.session_state.current_user['id'],)
                            )
                            conn.commit()
                        st.session_state.messages = []
                        st.session_state.conversation_id = None
                        st.rerun()

    if st.session_state.show_register:
        show_register()
    elif st.session_state.show_login:
        show_login()
    elif st.session_state.logged_in:
        show_chat()
    else:
        st.title("Welcome!")
        st.write("Login or register to start.")


if __name__ == "__main__":
    main()
