import streamlit as st
import random
import logging
import yaml
import joblib
import os
import sqlite3
import uuid 
import numpy as np 
from datetime import datetime, timedelta 
from dotenv import load_dotenv
from textblob import TextBlob 
import nltk 
import re 
import pandas as pd 

# --- Load Environment Variables ---
load_dotenv() 

# --- Configuration and Setup ---

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('MentalHealthBot')

# Define paths
DATA_DIR = 'data'
MODELS_DIR = 'models'
USERS_DB_PATH = 'users.db' 
TRAINING_DATA_DB_PATH = os.path.join(DATA_DIR, 'mental_health_chatbot.db') 

# Admin credentials from environment variables for security
ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", "admin@example.com") 
ADMIN_PASSWORD_HASH = os.getenv("ADMIN_PASSWORD_HASH", "") 

# Ensure data and models directories exist
os.makedirs(DATA_DIR, exist_ok=True) 
os.makedirs(MODELS_DIR, exist_ok=True) 

# Point NLTK to the custom data directory
NLTK_DATA_DIR = os.path.join(os.path.dirname(__file__), 'data', 'nltk_data')
if NLTK_DATA_DIR not in nltk.data.path:
    nltk.data.path.append(NLTK_DATA_DIR)
    logger.info(f"NLTK data path added: {NLTK_DATA_DIR}")


# YAML file paths
INTENTS_FILE = os.path.join(DATA_DIR, 'intents.yml')
BOT_PROFILE_FILE = os.path.join(DATA_DIR, 'bot_profile.yml')
COPING_STRATEGIES_FILE = os.path.join(DATA_DIR, 'coping_strategies.yml') 
CRISIS_SUPPORT_FILE = os.path.join(DATA_DIR, 'crisis_support.yml')
EMPATHY_AND_FEELINGS_FILE = os.path.join(DATA_DIR, 'empathy_and_feelings.yml')
GREETINGS_FILE = os.path.join(DATA_DIR, 'greetings.yml')
AFFIRMATIONS_FILE = os.path.join(DATA_DIR, 'affirmations.yml')
RESOURCES_AND_CRISIS_FILE = os.path.join(DATA_DIR, 'resources_and_crisis.yml')
USER_FEEDBACK_FILE = os.path.join(DATA_DIR, 'user_feedback.yml')
FALLBACKS_FILE = os.path.join(DATA_DIR, 'fallbacks.yml') 

# --- Global Data Loading (Cached) ---

# Use global variables to hold cached resources and models
GLOBAL_RESOURCES = {}
CLASSIFIER_MODEL = None
LABEL_ENCODER = None 

@st.cache_data(show_spinner="Loading AI knowledge base...")
def load_yaml_cached(filepath): 
    """Loads a YAML file from the specified path, designed for caching."""
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        logger.warning(f"YAML file not found: {filepath}. Returning None.")
        return None
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file {filepath}: {e}")
        return None

@st.cache_resource(show_spinner="Loading all bot resources...")
def load_all_resources_cached():
    """Loads all YAML resource files into GLOBAL_RESOURCES."""

    resources_dict = {} 

    # Load Intents - Store as a dictionary for easy lookup by tag
    intents_data_raw = load_yaml_cached(INTENTS_FILE)
    if intents_data_raw and 'intents' in intents_data_raw:
        resources_dict['intents'] = {item['tag']: item for item in intents_data_raw['intents']}
    else:
        resources_dict['intents'] = {}

    # Load Bot Profile
    bot_profile_data = load_yaml_cached(BOT_PROFILE_FILE)
    if bot_profile_data:
        resources_dict['bot_identity'] = bot_profile_data.get('bot_identity', {})
        resources_dict['response_architecture'] = bot_profile_data.get('response_architecture', {})
        resources_dict['bot_user_notices'] = bot_profile_data.get('user_notices', {}) 
    else:
        resources_dict['bot_identity'] = {}
        resources_dict['response_architecture'] = {}
        resources_dict['bot_user_notices'] = {}

    # Load Coping Strategies (full detail needed for selection logic)
    resources_dict['coping_strategies_detail'] = load_yaml_cached(COPING_STRATEGIES_FILE)

    # Load Crisis Support
    crisis_data = load_yaml_cached(CRISIS_SUPPORT_FILE)
    if crisis_data:
        resources_dict['crisis_protocols'] = crisis_data.get('response_protocols', {})
        resources_dict['kenya_emergency_services'] = crisis_data.get('kenya_emergency_services', {})
        resources_dict['safety_plan_template'] = crisis_data.get('safety_plan_template', {})
    else:
        resources_dict['crisis_protocols'] = {}
        resources_dict['kenya_emergency_services'] = {}
        resources_dict['safety_plan_template'] = {}

    # Load Empathy and Feelings
    empathy_data = load_yaml_cached(EMPATHY_AND_FEELINGS_FILE)
    if empathy_data:
        resources_dict['empathy_emotional_states'] = empathy_data.get('emotional_states', {})
        resources_dict['cultural_emotions'] = empathy_data.get('cultural_emotions', {}) 
    else:
        resources_dict['empathy_emotional_states'] = {}
        resources_dict['cultural_emotions'] = {}

    # Load Greetings and Emotional Openers
    greetings_data = load_yaml_cached(GREETINGS_FILE)
    all_greetings_patterns = []
    if greetings_data:
        cultural_greetings = greetings_data.get('cultural_greetings', {})
        for key, value in cultural_greetings.items():
            if isinstance(value, dict): 
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, list):
                        all_greetings_patterns.extend(sub_value)
            elif isinstance(value, list): 
                all_greetings_patterns.extend(value)
            elif isinstance(value, str): 
                all_greetings_patterns.append(value)

        time_sensitive_greetings = greetings_data.get('time_sensitive', {})
        all_greetings_patterns.extend(time_sensitive_greetings.get('morning', []))
        all_greetings_patterns.extend(time_sensitive_greetings.get('afternoon', []))
        all_greetings_patterns.extend(time_sensitive_greetings.get('evening', []))

        resources_dict['emotional_openers_responses'] = greetings_data.get('emotional_openers', {})
    else:
        resources_dict['emotional_openers_responses'] = {}

    resources_dict['greetings_patterns'] = all_greetings_patterns 

    # Load Affirmations
    affirmations_data = load_yaml_cached(AFFIRMATIONS_FILE)
    if affirmations_data:
        resources_dict['affirmations_core'] = affirmations_data.get('core_affirmations', {}) 
        resources_dict['affirmations_contextual'] = affirmations_data.get('contextual_affirmations', {})
    else:
        resources_dict['affirmations_core'] = {}
        resources_dict['affirmations_contextual'] = {}

    # Load General Resources and Crisis info (from resources_and_crisis.yml)
    resources_data = load_yaml_cached(RESOURCES_AND_CRISIS_FILE)
    if resources_data:
        resources_dict['therapy_resources'] = resources_data.get('therapy_resources', {})
        resources_dict['self_help_articles'] = resources_data.get('articles', {})
        resources_dict['inclusivity_resources_detail'] = resources_data.get('inclusivity_resources', {}) 
        resources_dict['mythbusters_content'] = resources_data.get('mythbusters', {})
        resources_dict['app_recommendations'] = resources_data.get('app_recommendations', {})
    else:
        resources_dict['therapy_resources'] = {}
        resources_dict['self_help_articles'] = {}
        resources_dict['inclusivity_resources_detail'] = {}
        resources_dict['mythbusters_content'] = {}
        resources_dict['app_recommendations'] = {}

    # Load User Feedback configuration
    user_feedback_data = load_yaml_cached(USER_FEEDBACK_FILE)
    resources_dict['user_feedback_config'] = user_feedback_data.get('feedback_channels', {}) if user_feedback_data else {}

    # Load Fallbacks
    fallbacks_data = load_yaml_cached(FALLBACKS_FILE)
    if fallbacks_data:
        resources_dict['fallbacks_default'] = fallbacks_data.get('default_fallbacks', []) 
        resources_dict['fallbacks_escalation'] = fallbacks_data.get('escalation_fallbacks', []) 
    else:
        resources_dict['fallbacks_default'] = []
        resources_dict['fallbacks_escalation'] = []

    logger.info("All YAML resources loaded into resources_dict.")
    return resources_dict 


@st.cache_resource(show_spinner="Loading AI model...")
def load_classifier_model_cached():
    """Loads the pre-trained intent classifier model and label encoder."""
    classifier_model = None
    label_encoder = None

    model_path = os.path.join(MODELS_DIR, 'intent_classifier.joblib')
    label_encoder_path = os.path.join(MODELS_DIR, 'label_encoder.joblib') 

    try:
        classifier_model = joblib.load(model_path) 
        logger.info(f"Classifier model loaded successfully from {model_path}.")
    except FileNotFoundError:
        logger.error(f"Model file not found at {model_path}. Please run train.py first.")
    except Exception as e:
        logger.error(f"Error loading classifier model: {e}")

    try:
        label_encoder = joblib.load(label_encoder_path)
        logger.info(f"Label Encoder loaded successfully from {label_encoder_path}.")
    except FileNotFoundError:
        logger.error(f"Label Encoder file not found at {label_encoder_path}. Please run train.py first.")
    except Exception as e:
        logger.error(f"Error loading Label Encoder: {e}")

    return classifier_model, label_encoder

# --- Database Functions (for users.db) ---

@st.cache_resource(show_spinner="Initializing user database...")
def init_user_db_cached():
    """Initialize database for users, conversations, and messages."""
    with sqlite3.connect(USERS_DB_PATH) as conn: 
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                username TEXT UNIQUE,
                email TEXT UNIQUE,
                password TEXT,
                is_admin INTEGER DEFAULT 0,
                plan_id INTEGER DEFAULT 1,
                plan_expiration_date TEXT, 
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                end_time TIMESTAMP,
                user_id TEXT,
                FOREIGN KEY (user_id) REFERENCES users (id)
            );
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                message_id TEXT PRIMARY KEY,
                conversation_id TEXT,
                sender TEXT NOT NULL,
                text TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                intent TEXT,
                confidence REAL,
                sentiment REAL,
                response_type TEXT,
                FOREIGN KEY (conversation_id) REFERENCES conversations (id)
            );
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pricing (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plan_name TEXT UNIQUE,
                max_conversations INTEGER,
                price INTEGER,
                currency TEXT DEFAULT 'Ksh'
            );
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS stories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT, 
                title TEXT,
                content TEXT,
                likes INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            );
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS contacts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                email TEXT,
                message TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        cursor.executescript("""
            INSERT OR IGNORE INTO pricing (id, plan_name, max_conversations, price, currency) VALUES 
            (1, 'Free', 5, 0, 'Ksh');
            INSERT OR IGNORE INTO pricing (id, plan_name, max_conversations, price, currency) VALUES 
            (2, 'Premium', 20, 1000, 'Ksh');
            INSERT OR IGNORE INTO pricing (id, plan_name, max_conversations, price, currency) VALUES 
            (3, 'Unlimited', NULL, 2000, 'Ksh');
        """)

        cursor.execute("SELECT id FROM users WHERE id = ?", (ADMIN_EMAIL,)) 
        if not cursor.fetchone():
            cursor.execute(
                "INSERT OR IGNORE INTO users (id, username, email, password, is_admin, plan_id, plan_expiration_date) VALUES (?, ?, ?, ?, 1, 3, ?)", 
                (ADMIN_EMAIL, "Admin", ADMIN_EMAIL, ADMIN_PASSWORD_HASH, None) 
            )
            logger.info("Default admin user created.")

        conn.commit()
    logger.info("User database initialized successfully.")


def log_message(conversation_id, sender, text, intent=None, confidence=None, sentiment=None, response_type=None):
    """Logs a message to the user database. Requires a conversation_id."""
    if not conversation_id:
        logger.warning(f"Attempted to save message without a valid conversation ID: {text[:20]}...")
        return

    message_id = str(uuid.uuid4())
    timestamp = datetime.now().isoformat()
    try:
        with sqlite3.connect(USERS_DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO messages (message_id, conversation_id, sender, text, timestamp, intent, confidence, sentiment, response_type)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (message_id, conversation_id, sender, text, timestamp, intent, confidence, sentiment, response_type)
            )
            conn.commit()
    except sqlite3.Error as e:
        logger.error(f"Database error logging message: {e}")

# --- Authentication Functions (using users.db) ---
import bcrypt 

def hash_pw(password): 
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

def check_pw(password, hashed): 
    return bcrypt.checkpw(password.encode(), hashed.encode())

def register_user(username, email, password, plan_id):
    user_id = email.strip().lower()
    try:
        with sqlite3.connect(USERS_DB_PATH) as conn:
            c = conn.cursor()
            c.execute(
                "INSERT INTO users (id, username, email, password, plan_id) VALUES (?, ?, ?, ?, ?)",
                (user_id, username.strip(), email.strip(), hash_pw(password), plan_id)
            )
            conn.commit()
        return True, "Registered successfully!"
    except sqlite3.IntegrityError as e:
        if "UNIQUE constraint failed" in str(e):
            return False, "Username or email already exists."
        return False, f"Database error during registration: {e}"
    except Exception as e:
        logger.error(f"Registration error: {e}")
        return False, "An error occurred during registration."

def login_user(user_or_email, password):
    user_or_email_clean = user_or_email.strip().lower()

    if user_or_email_clean == ADMIN_EMAIL.lower():
        if check_pw(password, ADMIN_PASSWORD_HASH):
            st.session_state.update({
                "logged_in": True,
                "current_user": {"id": ADMIN_EMAIL, "username": "Admin", "email": ADMIN_EMAIL}, 
                "is_admin": True
            })
            return True, "Admin login successful."
        return False, "Wrong admin password."

    try:
        with sqlite3.connect(USERS_DB_PATH) as conn:
            c = conn.cursor()
            c.execute(
                "SELECT id, username, email, password, is_admin FROM users WHERE username = ? OR email = ?",
                (user_or_email_clean, user_or_email_clean)
            )
            row = c.fetchone()

            if row and check_pw(password, row[3]):
                st.session_state.update({
                    "logged_in": True,
                    "current_user": {"id": row[0], "username": row[1], "email": row[2]},
                    "is_admin": bool(row[4])
                })
                return True, f"Welcome back, {row[1]}!"
            return False, "Invalid credentials."
    except Exception as e:
        logger.error(f"Login error: {e}")
        return False, "An error occurred during login."

def has_exceeded_usage(user_id):
    # Admin users have unlimited conversations
    if st.session_state.is_admin:
        return False

    with sqlite3.connect(USERS_DB_PATH) as conn:
        c = conn.cursor()
        c.execute("""
            SELECT p.max_conversations, u.plan_expiration_date
            FROM users u 
            JOIN pricing p ON u.plan_id = p.id 
            WHERE u.id = ?
        """, (user_id,))
        result = c.fetchone()

        if not result:
            logger.warning(f"User {user_id} not found for usage check.")
            return True 

        max_convos = result[0]
        plan_expiration_date_str = result[1]

        if plan_expiration_date_str:
            expiration_date = datetime.fromisoformat(plan_expiration_date_str)
            if datetime.now() > expiration_date:
                logger.info(f"User {user_id} plan has expired.")
                return True 

        if max_convos is None: 
            return False

        c.execute("SELECT COUNT(*) FROM conversations WHERE user_id = ?", (user_id,))
        current_convos = c.fetchone()[0]
        return current_convos >= max_convos

def start_conversation_in_db(user_id):
    # If anonymous, create a temp user ID and ensure it exists in DB
    if not st.session_state.get('logged_in', False): 
        anon_session_id_prefix = f"anon_user_{str(uuid.uuid4())[:8]}" 
        user_id = anon_session_id_prefix 
        try:
            with sqlite3.connect(USERS_DB_PATH) as conn:
                c = conn.cursor()
                c.execute("INSERT OR IGNORE INTO users (id, username, email, is_admin, plan_id) VALUES (?, ?, ?, 0, 1)", 
                          (user_id, "Anonymous", f"{user_id}@example.com")) 
                conn.commit()
            st.session_state.current_user = {'id': user_id, 'username': 'Anonymous', 'email': f'{user_id}@example.com'}
        except sqlite3.Error as e:
            logger.error(f"Error creating anonymous user in DB: {e}")
            return None 

    conversation_id = str(uuid.uuid4())
    try:
        with sqlite3.connect(USERS_DB_PATH) as conn:
            c = conn.cursor()
            c.execute(
                "INSERT INTO conversations (id, user_id) VALUES (?, ?)",
                (conversation_id, user_id)
            )
            conn.commit()
        logger.info(f"Conversation {conversation_id} started for user {user_id}")
        return conversation_id
    except sqlite3.Error as e:
        logger.error(f"Error starting conversation in DB: {e}")
        return None

# --- AI Logic Functions ---

def analyze_sentiment(text):
    """
    Analyzes sentiment of text using TextBlob with custom Kenyan keyword biasing.
    """
    try:
        analysis = TextBlob(text)
        polarity = analysis.sentiment.polarity

        kenyan_positive_words = {"poa", "safi", "fitii", "mzima", "nzuri", "fresh", "radhi", "fiti", "nzuri", "barikiwa", "shukran", "good", "great", "excellent", "happy", "joy", "positive", "wonderful", "amazing"}
        kenyan_negative_words = {"pole", "sad", "huzuni", "mbaya", "stress", "pressure", "kufa", "shida", "majuto", "not okay", "depressed", "afraid", "low", "worthless", "hopeless", "terrible", "awful", "bad", "maisha ni ngumu", "nimechoka", "broken", "unhappy", "miserable", "pain", "hurt", "grieving", "loss", "dumped", "failed"}

        text_lower = text.lower()

        if any(word in text_lower for word in kenyan_negative_words):
            polarity = max(-1.0, polarity - 0.2) 
        elif any(word in text_lower for word in kenyan_positive_words):
            polarity = min(1.0, polarity + 0.2) 

        return max(-1.0, min(1.0, float(polarity))) 

    except Exception as e:
        logger.error(f"Error in analyze_sentiment with TextBlob: {e}")
        text_lower = text.lower()
        score = 0
        if any(word in text_lower for word in ["sad", "depressed", "not okay", "bad", "low"]):
            score -= 1
        if any(word in text_lower for word in ["happy", "good", "great", "okay"]):
            score += 1
        return score / (len(text_lower.split()) + 0.01) 

def get_random_coping_strategy_detail(strategy_type=None):
    """Retrieves a random coping strategy detail from the loaded YAML, optionally by type."""
    coping_detail = GLOBAL_RESOURCES.get('coping_strategies_detail', {})

    all_strategies_flat = []

    def add_strategies_from_nested(category_data, list_key=None, source_label=None):
        if not category_data: return
        if list_key and isinstance(category_data, dict) and list_key in category_data and isinstance(category_data[list_key], list):
            for item in category_data[list_key]:
                if isinstance(item, dict):
                    item_copy = item.copy() 
                    item_copy['source_type'] = source_label 
                    all_strategies_flat.append(item_copy)
        elif isinstance(category_data, list): 
            for item in category_data:
                if isinstance(item, dict):
                    item_copy = item.copy() 
                    item_copy['source_type'] = source_label
                    all_strategies_flat.append(item_copy)
        elif isinstance(category_data, dict) and source_label: 
             item_copy = category_data.copy() 
             item_copy['source_type'] = source_label
             all_strategies_flat.append(item_copy)


    # Core Strategies
    if 'core_strategies' in coping_detail:
        for strat_type, strat_content in coping_detail['core_strategies'].items():
            add_strategies_from_nested(strat_content, 'techniques', strat_type) 

    # Cultural Strategies
    if 'cultural_strategies' in coping_detail:
        for cat_name, cat_data in coping_detail['cultural_strategies'].items():
            add_strategies_from_nested(cat_data, 'methods', cat_name) 

    # Situation-Specific (by_scenario)
    if 'by_scenario' in coping_detail and isinstance(coping_detail['by_scenario'], list):
        for scenario in coping_detail['by_scenario']:
            add_strategies_from_nested(scenario, 'strategies', 'by_scenario_strategy') 

    # Other top-level lists/dicts
    add_strategies_from_nested(coping_detail.get('movement_based'), None, 'movement_based')
    add_strategies_from_nested(coping_detail.get('creative_outlets'), None, 'creative_outlets')

    # For digital_tools, need to go deeper
    if 'digital_tools' in coping_detail and coping_detail['digital_tools'] and 'app_recommendations' in coping_detail['digital_tools']:
        for app_type, app_list in coping_detail['digital_tools']['app_recommendations'].items():
            if isinstance(app_list, list):
                for app_item in app_list:
                    if isinstance(app_item, dict):
                        app_copy = app_item.copy()
                        app_copy['source_type'] = f'digital_app_{app_type}'
                        all_strategies_flat.append(app_copy)

    # For immediate_support_tips
    if 'immediate_support_tips' in coping_detail and coping_detail['immediate_support_tips'] and 'tips' in coping_detail['immediate_support_tips']:
        add_strategies_from_nested(coping_detail['immediate_support_tips'], 'tips', 'immediate_support_tips')

    # For safety_planning_preventive
    if 'safety_planning_preventive' in coping_detail and coping_detail['safety_planning_preventive'] and 'steps' in coping_detail['safety_planning_preventive']:
        add_strategies_from_nested(coping_detail['safety_planning_preventive'], 'steps', 'safety_planning_preventive')


    # Filter by strategy_type if specified
    if strategy_type:
        filtered_strategies = [s for s in all_strategies_flat if 
                                s.get('source_type') == strategy_type or 
                                (s.get('name') and s.get('name').lower().replace(' ', '_') == strategy_type.lower()) or
                                (s.get('tip') and s.get('tip').lower().replace(' ', '_') == strategy_type.lower()) 
                               ]
        if filtered_strategies:
            return random.choice(filtered_strategies)

    if all_strategies_flat:
        return random.choice(all_strategies_flat)
    return None

def get_random_affirmation_detail(affirmation_type=None):
    """Retrieves a random affirmation detail from the loaded YAML, optionally by type."""
    core_affirmations = GLOBAL_RESOURCES.get('affirmations_core', {})
    contextual_affirmations = GLOBAL_RESOURCES.get('affirmations_contextual', {})

    target_list = []

    def add_affirmations_from_category(category_dict, source_type):
        if category_dict and source_type in category_dict and 'list' in category_dict[source_type]:
            for item in category_dict[source_type]['list']:
                item_copy = item.copy() 
                item_copy['type'] = source_type
                target_list.append(item_copy)

    if affirmation_type:
        add_affirmations_from_category(core_affirmations, affirmation_type)
        add_affirmations_from_category(contextual_affirmations, affirmation_type)
    else: 
        for category_tag in core_affirmations.keys():
            add_affirmations_from_category(core_affirmations, category_tag)
        for category_tag in contextual_affirmations.keys():
            add_affirmations_from_category(contextual_affirmations, category_tag)

    if target_list:
        return random.choice(target_list)
    return None


def find_response(user_input):
    """
    Determines the appropriate chatbot response based on user input,
    prioritizing crisis, then specific emotional states (keyword/intent),
    then classified general intent, and finally sentiment.
    """
    user_input_lower = user_input.lower().strip()
    logger.info(f"User Input: '{user_input}'") 

    predicted_intent = "unclassified" 
    confidence = 0.0

    # --- 1. Intent Classification (Always run first to get intent & confidence) ---
    if CLASSIFIER_MODEL and LABEL_ENCODER: 
        try:
            intent_probabilities = CLASSIFIER_MODEL.predict_proba([user_input_lower])[0]
            predicted_intent_index = np.argmax(intent_probabilities)
            predicted_intent = LABEL_ENCODER.inverse_transform([predicted_intent_index])[0] 
            confidence = intent_probabilities[predicted_intent_index]
            logger.info(f"Classifier Predicted: {predicted_intent} with confidence {confidence:.2f}")
        except Exception as e:
            logger.error(f"Error during classifier prediction: {e}")
            predicted_intent = "classifier_error_fallback" 
            confidence = 0.0
    else:
        logger.warning("Classifier model or Label Encoder not loaded. Relying on keyword matching and sentiment.")

    # --- 2. Crisis Intent Handling (Highest Priority - Very Low Threshold for Safety) ---
    critical_crisis_intents = [
        'suicidal_ideation_explicit',
        'suicidal_attempt_active',
        'self_harm_threat',
        'abuse_gbv'
    ]

    if predicted_intent in critical_crisis_intents and confidence > 0.25: # Safety threshold
        intent_data = GLOBAL_RESOURCES.get('intents', {}).get(predicted_intent) 

        if intent_data:
            responses = intent_data.get('responses', [])
            follow_up_questions = intent_data.get('follow_up', [])

            if responses:
                bot_response = f"🚨 {random.choice(responses)}"
                if follow_up_questions:
                    bot_response += f" {random.choice(follow_up_questions)}"

                st.session_state.last_bot_message_type = predicted_intent
                st.session_state.expected_next_action = None 
                logger.info(f"Responded based on HIGH PRIORITY crisis intent: {predicted_intent}")
                log_message(st.session_state.conversation_id, "bot", bot_response, predicted_intent, confidence, None, "crisis_response")
                return bot_response
        logger.warning(f"No responses found for HIGH PRIORITY crisis intent: {predicted_intent}")

    # --- 3. Contextual Affirmative Response Handling ---
    is_affirmative_keyword_present = ("yes" in user_input_lower or "sure" in user_input_lower or "okay" in user_input_lower or "yup" in user_input_lower or "ndiyo" in user_input_lower or "na'am" in user_input_lower or "sawa" in user_input_lower)

    if (predicted_intent == 'affirmative_response' and confidence > 0.5) or \
    (is_affirmative_keyword_present and st.session_state.get('expected_next_action')): 

        response_given = False

        if st.session_state.get('expected_next_action') == 'ask_coping_strategy_choice':
            selected_strategy = get_random_coping_strategy_detail() 
            if selected_strategy:
                bot_response = f"Excellent! Let's try **{selected_strategy.get('name', 'a coping strategy')}**."
                if selected_strategy.get('how_to_guide') and isinstance(selected_strategy['how_to_guide'], list):
                    bot_response += f" Here's how: {random.choice(selected_strategy['how_to_guide'])}"
                elif selected_strategy.get('description'):
                    bot_response += f" {selected_strategy['description']}"

                if selected_strategy.get('follow_up_questions'): 
                    bot_response += f" {random.choice(selected_strategy['follow_up_questions'])}"
                else:
                    bot_response += " How does that sound? Did it help you feel a little better?"

                st.session_state.last_bot_message_type = 'delivered_coping_strategy'
                response_given = True
                logger.info("Responded: Contextual Affirmative for Coping Strategy.")
                log_message(st.session_state.conversation_id, "bot", bot_response, predicted_intent, confidence, None, "affirmative_coping")
                return bot_response
            else:
                logger.warning("No specific coping strategy found for affirmative_response context.")

        elif st.session_state.expected_next_action == 'ask_affirmation_choice':
            selected_affirmation = get_random_affirmation_detail()
            if selected_affirmation:
                bot_response = f"Wonderful! Here’s a positive affirmation for you: '{selected_affirmation.get('text', 'No affirmation found.')}'"
                if selected_affirmation.get('meaning'):
                    bot_response += f" Meaning: {selected_affirmation['meaning']}."

                affirmation_category_tag = selected_affirmation.get('type') 
                category_details = GLOBAL_RESOURCES.get('affirmations_core', {}).get(affirmation_category_tag) or \
                                   GLOBAL_RESOURCES.get('affirmations_contextual', {}).get(affirmation_category_tag)
                if category_details and category_details.get('reflection_questions'):
                     bot_response += f" {random.choice(category_details['reflection_questions'])}"
                else:
                     bot_response += " How does that resonate with you today?"

                st.session_state.last_bot_message_type = 'delivered_affirmation'
                response_given = True
                logger.info("Responded: Contextual Affirmative for Affirmation.")
                log_message(st.session_state.conversation_id, "bot", bot_response, predicted_intent, confidence, None, "affirmative_affirmation")
                return bot_response
            else:
                logger.warning("No affirmation found for affirmative_response context.")

        elif st.session_state.expected_next_action == 'ask_resource_choice':
            therapy_resources_kenya = GLOBAL_RESOURCES.get('therapy_resources', {}).get('kenya', [])
            if therapy_resources_kenya:
                selected_resource = random.choice(therapy_resources_kenya)
                bot_response = f"Okay, here's a professional resource: **{selected_resource.get('name')}**. They offer {selected_resource.get('services', 'various services')}. Contact: {selected_resource.get('contact', 'N/A')}."
                if selected_resource.get('follow_up_questions'):
                    bot_response += f" {random.choice(selected_resource['follow_up_questions'])}"
                else:
                    bot_response += " Would you like to know more about this resource?"

                st.session_state.last_bot_message_type = 'delivered_resource'
                response_given = True
                logger.info("Responded: Contextual Affirmative for Resources.")
                log_message(st.session_state.conversation_id, "bot", bot_response, predicted_intent, confidence, None, "affirmative_resource")
                return bot_response
            else:
                logger.warning("No resources found for affirmative_response context.")

        if response_given: 
            st.session_state.expected_next_action = None
        else: 
            intent_data = GLOBAL_RESOURCES.get('intents', {}).get(predicted_intent) 
            if intent_data:
                responses = intent_data.get('responses', [])
                follow_up_questions = intent_data.get('follow_up', [])
                bot_response = random.choice(responses)
                if follow_up_questions:
                    bot_response += f" {random.choice(follow_up_questions)}"

                st.session_state.last_bot_message_type = predicted_intent 
                st.session_state.expected_next_action = None 
                logger.info(f"Responded based on general affirmative intent (no specific context): {predicted_intent}")
                log_message(st.session_state.conversation_id, "bot", bot_response, predicted_intent, confidence, None, "affirmative_general")
                return bot_response

    # --- 4. Direct Emotional Keyword Match (High Priority Fallback for Empathy) ---
    emotional_keywords_map = {
        "not okay": "sadness", "upset": "sadness", "down": "sadness", "terrible": "sadness", "awful": "sadness", "bad": "sadness",
        "sad": "sadness", "depressed": "sadness", "hopeless": "sadness", "crying": "sadness", "nothing matters": "sadness",
        "heartbroken": "sadness", "maisha ni ngumu": "sadness", "nimechoka": "sadness", "low": "sadness", "dumped": "sadness", "failed": "sadness", "broken": "sadness", "miserable": "sadness", "unhappy": "sadness",
        "lonely": "loneliness", "alone": "loneliness", "isolated": "loneliness", "no one cares": "loneliness",
        "anxious": "anxiety", "wasiwasi": "anxiety", "panicking": "anxiety", "nervous": "anxiety", "overwhelmed": "anxiety",
        "stressed": "stress", "pressure": "stress", "burnt out": "stress", "exhausted": "stress",
        "angry": "anger", "furious": "anger", "hasira": "anger", "frustrated": "anger", "lash out": "anger",
        "pain": "express_pain_distress", "hurting": "express_pain_distress", "suffering": "express_pain_distress"
    }

    matched_emotional_state_by_keyword = None
    for keyword, intent_tag in emotional_keywords_map.items():
        if keyword in user_input_lower:
            matched_emotional_state_by_keyword = intent_tag
            break 

    if matched_emotional_state_by_keyword:
        intent_data = GLOBAL_RESOURCES.get('intents', {}).get(matched_emotional_state_by_keyword)
        if intent_data:
            responses = intent_data.get('responses', [])
            follow_up_questions = intent_data.get('follow_up', [])

            if responses:
                bot_response = f"💙 {random.choice(responses)}"
                if follow_up_questions:
                    bot_response += f" {random.choice(follow_up_questions)}"

                st.session_state.last_bot_message_type = matched_emotional_state_by_keyword
                st.session_state.expected_next_action = None 
                logger.info(f"Matched: Direct Emotional Keyword Fallback for {matched_emotional_state_by_keyword}")
                log_message(st.session_state.conversation_id, "bot", bot_response, predicted_intent, confidence, None, "emotional_keyword_response")
                return bot_response
        logger.warning(f"No specific intent responses found for direct emotional keyword: {matched_emotional_state_by_keyword}")

    # --- 5. Respond based on Classified Intent (Specific Emotional Intent Threshold) ---
    emotional_intents_classifier = ['sadness', 'anxiety', 'stress', 'anger', 'loneliness', 'express_pain_distress', 'express_stress_anticipation']

    if predicted_intent in emotional_intents_classifier and confidence > 0.35: # Emotional classifier threshold
        intent_data = GLOBAL_RESOURCES.get('intents', {}).get(predicted_intent)

        if intent_data:
            responses = intent_data.get('responses', [])
            follow_up_questions = intent_data.get('follow_up', [])

            if responses:
                bot_response = random.choice(responses)
                if follow_up_questions:
                    bot_response += f" {random.choice(follow_up_questions)}"

                st.session_state.last_bot_message_type = predicted_intent 
                st.session_state.expected_next_action = None 
                logger.info(f"Responded based on classified emotional intent (threshold): {predicted_intent}")
                log_message(st.session_state.conversation_id, "bot", bot_response, predicted_intent, confidence, None, "intent_response_emotional_classifier")
                return bot_response
        logger.warning(f"No responses found for classified emotional intent (threshold): {predicted_intent}")

    # --- 6. Respond based on Classified Intent (Standard Threshold for Other Intents) ---
    if predicted_intent and confidence > 0.5: # Standard threshold for general intents
        intent_data = GLOBAL_RESOURCES.get('intents', {}).get(predicted_intent)

        if intent_data:
            responses = intent_data.get('responses', [])
            follow_up_questions = intent_data.get('follow_up', [])

            if responses:
                bot_response = random.choice(responses)

                # Special handling for intents that offer options, setting expected_next_action
                if predicted_intent == 'seek_coping_strategies':
                    st.session_state.expected_next_action = 'ask_coping_strategy_choice'
                elif predicted_intent == 'seek_affirmation' or predicted_intent == 'seek_contextual_affirmation':
                    st.session_state.expected_next_action = 'ask_affirmation_choice'
                elif predicted_intent == 'therapy_info' or predicted_intent == 'general_resources':
                    st.session_state.expected_next_action = 'ask_resource_choice'
                else:
                    st.session_state.expected_next_action = None 

                if follow_up_questions:
                    bot_response += f" {random.choice(follow_up_questions)}"

                st.session_state.last_bot_message_type = predicted_intent 
                logger.info(f"Responded based on classified general intent: {predicted_intent}")
                log_message(st.session_state.conversation_id, "bot", bot_response, predicted_intent, confidence, None, "intent_response_general")
                return bot_response
        logger.warning(f"No responses found for classified general intent: {predicted_intent}")

    # --- 7. Sentiment analysis fallback (If no specific intent/keyword matched) ---
    sentiment = analyze_sentiment(user_input)
    logger.info(f"Sentiment analysis result: {sentiment}") 

    if sentiment < -0.3: 
        sad_openers = GLOBAL_RESOURCES.get('emotional_openers_responses', {}).get('sad', [])
        if sad_openers:
            bot_response = f"💙 {random.choice(sad_openers)}"
            logger.info("Matched: Negative Sentiment Fallback (Emotional Opener)")
            st.session_state.last_bot_message_type = 'sentiment_negative'
            st.session_state.expected_next_action = None 
            log_message(st.session_state.conversation_id, "bot", bot_response, predicted_intent, confidence, sentiment, "sentiment_negative")
            return bot_response
        else:
            bot_response = "💙 I hear you might be struggling. Would you like to talk about it more?"
            logger.info("Matched: Negative Sentiment Fallback (Generic)")
            st.session_state.last_bot_message_type = 'sentiment_negative_generic'
            st.session_state.expected_next_action = None 
            log_message(st.session_state.conversation_id, "bot", bot_response, predicted_intent, confidence, sentiment, "sentiment_negative_generic")
            return bot_response
    elif sentiment > 0.3: 
        happy_openers = GLOBAL_RESOURCES.get('emotional_openers_responses', {}).get('happy', [])
        if happy_openers:
            bot_response = f"😊 {random.choice(happy_openers)}"
            logger.info("Matched: Positive Sentiment Fallback (Emotional Opener)")
            st.session_state.last_bot_message_type = 'sentiment_positive'
            st.session_state.expected_next_action = None 
            log_message(st.session_state.conversation_id, "bot", bot_response, predicted_intent, confidence, sentiment, "sentiment_positive")
            return bot_response
        else:
            bot_response = "😊 That sounds positive! I'm glad to hear it. How can I continue to support you?"
            logger.info("Matched: Positive Sentiment Fallback (Generic)")
            st.session_state.last_bot_message_type = 'sentiment_positive_generic'
            st.session_state.expected_next_action = None 
            log_message(st.session_state.conversation_id, "bot", bot_response, predicted_intent, confidence, sentiment, "sentiment_positive_generic")
            return bot_response

    # --- 8. Final fallback if nothing else matches ---
    final_fallbacks = GLOBAL_RESOURCES.get('fallbacks_default', []) 
    if not final_fallbacks: 
        final_fallbacks = [
            "💙 I'm here to listen. Please tell me more.",
            "💙 How can I support you today?",
            "💙 Niko hapa kwa ajili yako. (I'm here for you.)"
        ]

    bot_response = random.choice(final_fallbacks)
    logger.info("Matched: Final Generic Fallback (from fallbacks.yml or hardcoded)")
    st.session_state.last_bot_message_type = 'generic_fallback'
    st.session_state.expected_next_action = None 
    log_message(st.session_state.conversation_id, "bot", bot_response, predicted_intent, confidence, sentiment, "generic_fallback")
    return bot_response

# --- Streamlit UI ---

def show_landing_page():
    """Displays the initial landing page for Kelly AI."""
    st.title("👋 Karibu to Kelly AI - Your Mental Health Companion 🇰🇪")
    st.markdown(f"""
    At **Kelly AI**, we understand that navigating mental well-being in Kenya comes with unique joys and challenges. 
    We are a **culturally-sensitive AI chatbot** designed to offer a safe and supportive space for you.

    **What I can do for you:**
    * **Listen Actively:** Share your thoughts and feelings without judgment.
    * **Provide Coping Strategies:** Discover practical techniques for managing stress, anxiety, sadness, and more, including culturally relevant approaches.
    * **Offer Affirmations:** Receive positive, uplifting messages grounded in Kenyan values.
    * **Connect to Resources:** Get information on local Kenyan mental health services, hotlines, and community support groups.
    * **Support in Crisis:** Access immediate guidance and and emergency contacts during difficult moments.
    * **Understand Your Context:** I'm built to recognize and respond to Kenyan idioms, proverbs, and unique life experiences.

    Whether you need a listening ear, a quick calming exercise, or a referral to a professional, Kelly AI is here, 24/7.
    """)

    st.markdown("---")
    st.markdown("Click 'Continue' to start your conversation or explore our features.")

    if st.button("Continue to Kelly AI Chat", key="landing_continue_button"):
        st.session_state.page = "chat_page" 
        st.rerun()

def show_chat():
    """Renders the Streamlit chat interface."""
    st.title("💬 Kelly AI - Mental Bot")

    if not st.session_state.logged_in:
        st.info(
            "Welcome! You are currently chatting as an anonymous user. "
            "Some features like 'History' and 'Stories' are available only to logged-in users. "
            "Please **Login** or **Register** in the sidebar to access full features."
        )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What's on your mind?"):
        if has_exceeded_usage(st.session_state.current_user.get('id', 'anonymous_session_init')): 
            st.warning("You've exceeded your conversation limit. Please upgrade your plan or log in.")
            return 

        st.session_state.messages.append({"role": "user", "content": prompt})
        log_message(st.session_state.conversation_id, "user", prompt) 
        with st.chat_message("user"):
            st.markdown(prompt)

        bot_response = find_response(prompt)

        st.session_state.messages.append({"role": "assistant", "content": bot_response})
        with st.chat_message("assistant"):
            st.markdown(bot_response)

        st.rerun() 

# --- Authentication UI (Updated to use USERS_DB_PATH) ---

def show_login():
    st.title("🔑 Login")
    user_or_email = st.text_input("Username or Email", key="login_user_email")
    password = st.text_input("Password", type="password", key="login_password")

    if st.button("Login", key="login_button"):
        success, message = login_user(user_or_email, password)
        if success:
            st.success(message)
            if 'current_user' in st.session_state and st.session_state.current_user:
                st.session_state.conversation_id = start_conversation_in_db(st.session_state.current_user['id'])
                st.session_state.messages = [] 
                st.session_state.last_bot_message_type = None 
                st.session_state.expected_next_action = None 
            st.session_state.page = "chat_page" 
            st.rerun()
        else:
            st.error(message)

def show_register():
    st.title("📝 Register")
    username = st.text_input("Username", key="reg_username")
    email = st.text_input("Email", key="reg_email")
    password = st.text_input("Password", type="password", key="reg_password")
    confirm = st.text_input("Confirm Password", type="password", key="reg_confirm_password")

    with sqlite3.connect(USERS_DB_PATH) as conn: 
        c = conn.cursor()
        c.execute("SELECT id, plan_name FROM pricing") 
        plans = c.fetchall()

    plan_options = [p[1] for p in plans if p[1] != 'Unlimited'] 
    selected_plan = st.selectbox("Choose a Plan", plan_options, key="reg_plan_select")

    if st.button("Register", key="register_button"):
        if password != confirm:
            st.error("Passwords don't match")
        elif not (8 <= len(password) <= 20): 
            st.error("Password must be between 8 and 20 characters.")
        elif not re.search(r"[a-z]", password) or \
             not re.search(r"[A-Z]", password) or \
             not re.search(r"[0-9]", password) or \
             not re.search(r"[!@#$%^&*()_+\-=\[\]{};':\"\\|,.<>\/?]", password): 
            st.error("Password must include uppercase, lowercase, numbers, and special characters (!@#$%^&*()_+-=[]{};':\"|,.<>/?).")
        else:
            plan_id = next(p[0] for p in plans if p[1] == selected_plan)
            success, message = register_user(username, email, password, plan_id)
            if success:
                st.success(message)
                st.session_state.page = "login"
                st.rerun()
            else:
                st.error(message)

def show_about():
    st.title("ℹ️ About Kelly AI")
    bot_identity = GLOBAL_RESOURCES.get('bot_identity', {})
    st.markdown(f"""
    **{bot_identity.get('full_title', 'Kelly AI Mental Health Companion')}** is designed to be your supportive partner in navigating mental well-being, with a special focus on the Kenyan context.

    **Our Mission:** {bot_identity.get('mission_statement', 'To provide accessible, culturally sensitive, and compassionate mental health support to Kenyans, empowering individuals to navigate their well-being journeys with resilience and hope.')}

    **Key Capabilities:**
    * **Listen Actively:** Share your thoughts and feelings without judgment.
    * **Provide Coping Strategies:** Discover practical techniques for managing stress, anxiety, sadness, and more, including culturally relevant approaches.
    * **Offer Affirmations:** Receive positive, uplifting messages grounded in Kenyan values.
    * **Connect to Resources:** Get information on local Kenyan mental health services, hotlines, and community support groups.
    * **Support in Crisis:** Access immediate guidance and emergency contacts during difficult moments.
    * **Understand Your Context:** I'm built to recognize and respond to Kenyan idioms, proverbs, and unique life experiences.

    **Core Values:**
    * {', '.join(bot_identity.get('core_values', ['Empathy', 'Confidentiality', 'Cultural Relevance', 'Accessibility', 'Empowerment']))}

    **Developed in:** {bot_identity.get('kenyan_identity', {}).get('development_location', 'Nairobi, Kenya')} by {bot_identity.get('creator', 'a dedicated team')}.
    """)
    st.markdown("---")
    st.subheader("Transparency & Limitations")
    user_notices = GLOBAL_RESOURCES.get('bot_user_notices', {}) 
    if user_notices.get('limitations'):
        st.markdown("**Limitations:**")
        for limit in user_notices['limitations']:
            st.markdown(f"- {limit}")
    else:
        st.markdown("- No limitations data found in bot_profile.yml")

    st.markdown("---")
    st.subheader("Your Rights")
    if user_notices.get('rights'):
        for right in user_notices['rights']:
            st.markdown(f"- {right}")
    else:
        st.markdown("- No user rights data found in bot_profile.yml")


def show_contact():
    st.title("📬 Contact Us")
    st.markdown("""
    Have questions, feedback, or need to reach us for support? You can contact the Kelly AI team.
    """)
    with st.form("contact_form", key="contact_form_key"):
        name = st.text_input("Your Name", key="contact_name")
        email = st.text_input("Your Email", key="contact_email")
        message = st.text_area("Your Message", key="contact_message")

        if st.form_submit_button("Send Message", key="contact_submit"):
            if name and email and message:
                with sqlite3.connect(USERS_DB_PATH) as conn: 
                    c = conn.cursor()
                    c.execute(
                        "INSERT INTO contacts (name, email, message) VALUES (?, ?, ?)",
                        (name, email, message)
                    )
                    conn.commit()
                st.success("Thank you! Your message has been sent successfully. We'll get back to you soon.")
                st.session_state["contact_name"] = ""
                st.session_state["contact_email"] = ""
                st.session_state["contact_message"] = ""
                st.rerun() 
            else:
                st.error("Please fill in all fields.")

def show_pricing():
    st.title("💰 Pricing Plans")
    st.markdown("Choose the plan that best suits your needs for accessing Kelly AI's full features.")
    with sqlite3.connect(USERS_DB_PATH) as conn: 
        c = conn.cursor()
        c.execute("SELECT * FROM pricing")
        plans = c.fetchall()

    for plan in plans:
        st.subheader(plan[1])
        st.markdown(f"**Price:** {plan[3]} {plan[4]}")
        st.markdown(f"**Max conversations:** {'Unlimited' if plan[2] is None else plan[2]}")
        if plan[1] == 'Free':
            st.markdown("Includes basic chat features and limited conversation history.")
        elif plan[1] == 'Premium':
            st.markdown("Access to extended conversation history, priority support, and all community features.")
        elif plan[1] == 'Unlimited':
            st.markdown("Everything in Premium, plus unlimited conversations and advanced analytics access (for admin/devs).")
        st.markdown("---")

def show_history():
    if not st.session_state.get("logged_in"):
        st.warning("Please **Login** or **Register** to view your conversation history.")
        return

    st.title("🕒 Your Conversation History")
    user_id = st.session_state.current_user.get('id')
    if user_id is None: 
        st.error("User ID not found. Please log in again.")
        return

    with sqlite3.connect(USERS_DB_PATH) as conn: 
        c = conn.cursor()
        c.execute("""
            SELECT id, start_time, end_time
            FROM conversations
            WHERE user_id = ?
            ORDER BY start_time DESC
        """, (user_id,))
        conversations = c.fetchall()

    if not conversations:
        st.info("You don't have any past conversations yet. Start chatting!")
        return

    st.markdown("Browse your past conversations with Kelly AI:")
    for conv_id, start_time, end_time in conversations:
        end_time_display = end_time if end_time else "Ongoing"
        with st.expander(f"Conversation {conv_id[:8]}... - Started: {start_time} (Ended: {end_time_display})"): 
            st.subheader(f"Conversation ID: `{conv_id}`")
            with sqlite3.connect(USERS_DB_PATH) as conn: 
                c = conn.cursor()
                c.execute("""
                    SELECT sender, text, timestamp, intent, confidence, sentiment, response_type
                    FROM messages 
                    WHERE conversation_id = ?
                    ORDER BY timestamp
                """, (conv_id,))
                messages = c.fetchall()

            for msg_sender, msg_text, msg_timestamp, msg_intent, msg_confidence, msg_sentiment, msg_response_type in messages:
                sender_label = "You" if msg_sender == "user" else "Kelly AI"
                with st.chat_message(msg_sender): 
                    st.markdown(f"**{sender_label} ({msg_timestamp.split('T')[1][:8]})**: {msg_text}")
                    intent_display = msg_intent if msg_intent is not None else "N/A"
                    confidence_display = f"{msg_confidence:.2f}" if msg_confidence is not None else "N/A"
                    sentiment_display = f"{msg_sentiment:.2f}" if msg_sentiment is not None else "N/A"
                    response_type_display = msg_response_type if msg_response_type is not None else "N/A"
                    st.caption(f"Intent: {intent_display} (Conf: {confidence_display}) | Sent: {sentiment_display} | Type: {response_type_display}")

def show_stories():
    if not st.session_state.get("logged_in"):
        st.warning("Please **Login** or **Register** to view and share community stories.")
        return

    st.title("📖 Community Stories")
    st.markdown("A space for shared experiences and inspiration within our community. Share your journey or read others'.")

    user_id = st.session_state.current_user.get('id')
    if user_id is None: 
        st.error("User ID not found. Please log in again.")
        return

    with st.expander("Share Your Story", expanded=False): 
        story_title = st.text_input("Title of Your Story", key="story_title_input")
        story_content = st.text_area("Write Your Story Here (max 2000 characters)", key="story_content_input", max_chars=2000)

        if st.button("Post My Story", key="post_story_button"):
            if story_title and story_content:
                if len(story_content) > 2000:
                    st.error("Story is too long. Please keep it under 2000 characters.")
                else:
                    with sqlite3.connect(USERS_DB_PATH) as conn: 
                        c = conn.cursor()
                        c.execute(
                            "INSERT INTO stories (user_id, title, content) VALUES (?, ?, ?)",
                            (user_id, story_title, story_content)
                        )
                        conn.commit()
                    st.success("Your story has been posted successfully! Thank you for sharing.")
                    st.session_state["story_title_input"] = ""
                    st.session_state["story_content_input"] = ""
                    st.rerun()
            else:
                st.error("Please provide both a title and content for your story.")

    st.markdown("---")
    st.subheader("Recent Stories from the Community:")
    with sqlite3.connect(USERS_DB_PATH) as conn: 
        c = conn.cursor()
        c.execute("""
            SELECT s.id, u.username, s.title, s.content, s.likes, s.created_at
            FROM stories s
            JOIN users u ON s.user_id = u.id
            ORDER BY s.created_at DESC
        """)
        stories = c.fetchall()

    if not stories:
        st.info("No community stories yet. Be the first to share your journey!")
        return

    for story_id, author_username, story_title, story_content, story_likes, created_at in stories:
        st.markdown(f"#### {story_title}")
        st.markdown(f"By **{author_username}** on _{created_at.split(' ')[0]}_") 
        st.write(story_content)
        if st.session_state.get("logged_in"):
            if st.button(f"❤️ {story_likes} Likes", key=f"like_{story_id}"):
                with sqlite3.connect(USERS_DB_PATH) as conn:
                    c = conn.cursor()
                    c.execute("UPDATE stories SET likes = likes + 1 WHERE id = ?", (story_id,))
                    conn.commit()
                st.success(f"You liked '{story_title}'!")
                st.rerun() 

        st.markdown("---")

def show_admin():
    if not st.session_state.get("is_admin"):
        st.warning("Admin access required")
        return

    st.title("👑 Admin Dashboard")
    st.markdown("Manage users, view conversation statistics, and review feedback.")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Users", "Conversations & Training Data", "User Feedback", "Manage Plans", "Add Training Data"]) 

    with tab1:
        st.subheader("User Management")
        with sqlite3.connect(USERS_DB_PATH) as conn: 
            c = conn.cursor()
            c.execute("SELECT id, username, email, created_at, is_admin, plan_id, plan_expiration_date FROM users")
            users = c.fetchall()

        user_data = [{"ID": u[0][:8] + '...', "Username": u[1], "Email": u[2], "Joined": u[3].split(' ')[0], "Admin": bool(u[4]), "Plan ID": u[5], "Plan Expires": u[6] if u[6] else "N/A"} for u in users] 
        st.dataframe(user_data) 

        st.markdown("---")
        st.subheader("Add New Admin User (Manual)")
        st.markdown("For adding initial admin or if existing admin has issues. Requires hashed password.")
        new_admin_username = st.text_input("New Admin Username", key="new_admin_username")
        new_admin_email = st.text_input("New Admin Email", key="new_admin_email")
        new_admin_password_hash = st.text_input("New Admin Hashed Password", key="new_admin_password_hash")

        if st.button("Create New Admin", key="create_new_admin_button"):
            if new_admin_username and new_admin_email and new_admin_password_hash:
                try:
                    with sqlite3.connect(USERS_DB_PATH) as conn:
                        c = conn.cursor()
                        admin_id = str(uuid.uuid4())
                        c.execute(
                            "INSERT OR IGNORE INTO users (id, username, email, password, is_admin, plan_id, plan_expiration_date) VALUES (?, ?, ?, ?, 1, 3, ?)", 
                            (admin_id, new_admin_username, new_admin_email, new_admin_password_hash, None) 
                        )
                        conn.commit()
                    st.success(f"Admin user '{new_admin_username}' created successfully!")
                    st.rerun()
                except sqlite3.IntegrityError:
                    st.error("Username or email already exists for new admin.")
                except Exception as e:
                    st.error(f"Error creating admin: {e}")
            else:
                st.error("Please fill all fields to create new admin.")


    with tab2:
        st.subheader("Conversation Statistics (User DB)")
        with sqlite3.connect(USERS_DB_PATH) as conn: 
            c = conn.cursor()
            c.execute("SELECT COUNT(*) FROM conversations")
            total_convos_user = c.fetchone()[0]
            c.execute("SELECT COUNT(*) FROM messages")
            total_messages_user = c.fetchone()[0]

        st.metric("Total User Conversations", total_convos_user)
        st.metric("Total User Messages", total_messages_user)

        st.markdown("---")
        st.subheader("Training Data Log (Data Folder DB)")
        with sqlite3.connect(TRAINING_DATA_DB_PATH) as conn: 
            c = conn.cursor()
            c.execute("SELECT COUNT(*) FROM bot_responses")
            total_training_entries = c.fetchone()[0]
        st.metric("Total Training Data Entries", total_training_entries)

        st.markdown("---")
        st.subheader("View Raw Training Data Log")
        if pd is not None:
            if st.button("Load Training Log (Max 100 entries)", key="load_training_log_button"):
                try:
                    with sqlite3.connect(TRAINING_DATA_DB_PATH) as conn:
                        df = pd.read_sql_query("SELECT * FROM bot_responses ORDER BY created_at DESC LIMIT 100", conn)
                        st.dataframe(df)
                except Exception as e:
                    st.error(f"Error loading training data log: {e}")
            else:
                st.info("Click 'Load Training Log' to view recent entries.")


    with tab3:
        st.subheader("User Feedback Submissions")
        with sqlite3.connect(USERS_DB_PATH) as conn: 
            c = conn.cursor()
            c.execute("SELECT name, email, message, created_at FROM contacts ORDER BY created_at DESC")
            feedback = c.fetchall()

        if not feedback:
            st.info("No user feedback submitted yet.")
        else:
            for item in feedback:
                st.markdown(f"**From**: {item[0]} (<{item[1]}>)")
                st.markdown(f"**Message**: {item[2]}")
                st.markdown(f"**Date**: _{item[3].split('T')[0]}_")
                st.markdown("---")

    with tab4: # Plan Management
        st.subheader("Manage User Plans")
        st.markdown("Select a user to update their subscription plan and set an expiration date.")

        with sqlite3.connect(USERS_DB_PATH) as conn:
            c = conn.cursor()
            c.execute("SELECT id, username, email, plan_id, plan_expiration_date FROM users WHERE is_admin = 0") # Exclude admins
            non_admin_users = c.fetchall()
            c.execute("SELECT id, plan_name FROM pricing")
            all_plans = c.fetchall()

        user_options = {f"{u[1]} ({u[2]})": u[0] for u in non_admin_users} 
        selected_user_display = st.selectbox("Select User", ["-- Select User --"] + list(user_options.keys()), key="select_user_for_plan")

        if selected_user_display != "-- Select User --":
            selected_user_id = user_options[selected_user_display]

            current_user_details = next((u for u in non_admin_users if u[0] == selected_user_id), None)
            if current_user_details:
                current_plan_id = current_user_details[3]
                current_expiration_date = current_user_details[4]

                st.write(f"Current Plan for **{selected_user_display}**: {plan_id_to_name.get(current_plan_id, 'N/A')}")
                st.write(f"Current Expiration: {current_expiration_date if current_expiration_date else 'Never / N/A'}")

                new_plan_name = st.selectbox("Assign New Plan", ["-- Select Plan --"] + [p[1] for p in all_plans], key="assign_new_plan") 
                plan_name_to_id = {p[1]: p[0] for p in all_plans} 

                if new_plan_name != "-- Select Plan --":
                    new_plan_id = plan_name_to_id[new_plan_name]

                    st.markdown("Set Plan Duration (for temporary upliftment, e.g., 'Premium for a month')")
                    duration_type = st.radio("Duration Type", ["No Expiration", "Days", "Months"], key="duration_type")

                    plan_duration_value = None
                    if duration_type == "Days":
                        plan_duration_value = st.number_input("Number of Days", min_value=1, value=30, key="plan_duration_days")
                    elif duration_type == "Months":
                        plan_duration_value = st.number_input("Number of Months", min_value=1, value=1, key="plan_duration_months")

                    calculated_expiration_date = None
                    if duration_type == "Days" and plan_duration_value:
                        calculated_expiration_date = datetime.now() + timedelta(days=plan_duration_value)
                    elif duration_type == "Months" and plan_duration_value:
                        calculated_expiration_date = datetime.now() + timedelta(days=plan_duration_value * 30) 

                    expiration_date_str = calculated_expiration_date.isoformat() if calculated_expiration_date else None

                    if st.button(f"Update Plan for {selected_user_display}", key="update_plan_button"):
                        try:
                            with sqlite3.connect(USERS_DB_PATH) as conn:
                                c = conn.cursor()
                                c.execute(
                                    "UPDATE users SET plan_id = ?, plan_expiration_date = ? WHERE id = ?",
                                    (new_plan_id, expiration_date_str, selected_user_id)
                                )
                                conn.commit()
                            st.success(f"Plan for {selected_user_display} updated to {new_plan_name} expiring on {expiration_date_str if expiration_date_str else 'N/A'}.")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error updating plan: {e}")
            else:
                st.info("Select a user to manage their plan.")

    with tab5: # New tab for adding training data
        st.subheader("Add New Training Data (Patterns)")
        st.markdown("""
        Add new user input patterns and link them to existing intents. 
        This is crucial for improving Kelly AI's understanding.
        **Remember to run `python3 train.py` from your terminal and then restart this Streamlit app after adding data for changes to take effect!**
        """)

        new_pattern = st.text_area("New User Input Pattern", key="new_pattern_input")

        intent_tags = sorted([tag for tag in GLOBAL_RESOURCES.get('intents', {}).keys()]) 
        selected_intent_tag = st.selectbox("Assign to Intent Category", ["-- Select Intent --"] + intent_tags, key="new_pattern_intent_select")

        new_example_response = st.text_area("Example Bot Response (Optional, for logging)", key="new_response_input")

        urgency_levels = ["low", "medium", "high", "critical"]
        selected_urgency = st.selectbox("Urgency Level (for logging)", urgency_levels, key="new_pattern_urgency")

        if st.button("Add Training Pattern", key="add_training_pattern_button"):
            if new_pattern and selected_intent_tag != "-- Select Intent --":
                try:
                    current_intents_yaml = load_yaml_cached(INTENTS_FILE)
                    if not current_intents_yaml:
                        current_intents_yaml = {'version': 1.0, 'type': 'intent_classification', 'last_updated': datetime.now().isoformat().split('T')[0], 'intents': []}

                    found_intent_obj = None
                    for intent_obj in current_intents_yaml.get('intents', []):
                        if intent_obj.get('tag') == selected_intent_tag:
                            found_intent_obj = intent_obj
                            break

                    if found_intent_obj:
                        if 'patterns' not in found_intent_obj:
                            found_intent_obj['patterns'] = []
                        found_intent_obj['patterns'].append(new_pattern)
                        if new_example_response and 'responses' not in found_intent_obj:
                            found_intent_obj['responses'] = [] 
                        if new_example_response and new_example_response not in found_intent_obj.get('responses', []):
                             found_intent_obj['responses'].append(new_example_response) 
                    else:
                        current_intents_yaml['intents'].append({
                            'tag': selected_intent_tag,
                            'patterns': [new_pattern],
                            'responses': [new_example_response if new_example_response else "Default response for new intent."],
                            'follow_up': [],
                            'metadata': {'urgency': selected_urgency} 
                        })

                    current_intents_yaml['last_updated'] = datetime.now().isoformat().split('T')[0]
                    current_intents_yaml['version'] = round(float(current_intents_yaml.get('version', 1.0)) + 0.01, 2) 

                    with open(INTENTS_FILE, 'w', encoding='utf-8') as f:
                        yaml.dump(current_intents_yaml, f, default_flow_style=False, sort_keys=False)

                    st.success(f"Pattern '{new_pattern}' added to '{selected_intent_tag}' in intents.yml!")
                    st.info("Remember: **Run `python3 train.py` and then restart this Streamlit app** for changes to take effect.")

                    st.session_state.new_pattern_input = ""
                    st.session_state.new_response_input = ""
                    st.session_state.new_pattern_intent_select = "-- Select Intent --" 
                    st.rerun()
                except Exception as e:
                    st.error(f"Error adding training pattern: {e}")
            else:
                st.error("Please enter a pattern and select an intent.")


# --- Main App Execution Flow ---

def main():
    """Main function to run the Streamlit app."""
    # Set page config here to avoid "already run" errors on rerun
    st.set_page_config(page_title="Kelly AI Mental Bot", layout="wide", initial_sidebar_state="collapsed") 

    # Initialize session state variables once per Streamlit session
    if 'page' not in st.session_state:
        st.session_state.page = "landing" 
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'current_user' not in st.session_state:
        st.session_state.current_user = {} 
    if 'is_admin' not in st.session_state:
        st.session_state.is_admin = False

    if 'expected_next_action' not in st.session_state:
        st.session_state.expected_next_action = None 

    if 'conversation_id' not in st.session_state: 
        st.session_state.conversation_id = None 
    if 'messages' not in st.session_state:
        st.session_state.messages = [] 
    if 'last_bot_message_type' not in st.session_state:
        st.session_state.last_bot_message_type = None 
    if 'initial_greeting_sent' not in st.session_state:
        st.session_state.initial_greeting_sent = False


    loaded_resources = load_all_resources_cached() 
    global GLOBAL_RESOURCES 
    GLOBAL_RESOURCES = loaded_resources 

    global CLASSIFIER_MODEL, LABEL_ENCODER 
    CLASSIFIER_MODEL, LABEL_ENCODER = load_classifier_model_cached() 
    init_user_db_cached() 

    if st.session_state.page == "chat_page" and st.session_state.conversation_id is None:
        user_id_for_conv = st.session_state.current_user.get('id') if st.session_state.get('current_user') else None
        st.session_state.conversation_id = start_conversation_in_db(user_id_for_conv) 
        st.session_state.messages = [] 
        st.session_state.last_bot_message_type = None 
        st.session_state.expected_next_action = None 

        if not st.session_state.initial_greeting_sent:
            greeting_responses = GLOBAL_RESOURCES.get('emotional_openers_responses', {}).get('general', ["Hello! How can I help you today?"])
            initial_message = random.choice(greeting_responses)
            st.session_state.messages.append({"role": "assistant", "content": initial_message})
            log_message(st.session_state.conversation_id, "bot", initial_message, "greeting", 1.0, 0.5, "initial_greeting")
            st.session_state.initial_greeting_sent = True 

    with st.sidebar:
        st.title("📚 Kelly AI")

        if st.session_state.logged_in:
            st.write(f"👋 Hello **{st.session_state.current_user.get('username', 'User')}**!")
            try:
                with sqlite3.connect(USERS_DB_PATH) as conn:
                    c = conn.cursor()
                    user_id = st.session_state.current_user['id']
                    c.execute("""
                        SELECT p.plan_name, 
                            (SELECT COUNT(*) FROM conversations WHERE user_id = ?) as convos,
                            p.max_conversations,
                            u.plan_expiration_date
                        FROM users u
                        JOIN pricing p ON u.plan_id = p.id 
                        WHERE u.id = ?
                    """, (user_id, user_id))
                    plan_info = c.fetchone()

                    if plan_info:
                        plan_name, convos, max_convos, plan_expiration_date = plan_info
                        st.markdown(f"Plan: **{plan_name}**")
                        st.markdown(f"Conversations: **{convos}**" + ("" if max_convos is None else f"/{max_convos}"))
                        if plan_expiration_date:
                            st.markdown(f"Expires: _{plan_expiration_date.split('T')[0]}_")
            except sqlite3.Error as e:
                logger.error(f"Error fetching plan info for sidebar: {e}")
                st.write("Plan info: N/A")
        else:
            st.write("👤 Status: **Anonymous User**")

        st.markdown("---")

        if st.session_state.page == "landing": 
            pass 
        else: 
            st.button("💬 Chat", key="nav_chat", on_click=lambda: setattr(st.session_state, 'page', "chat_page"))
            if st.session_state.logged_in:
                st.button("🕒 History", key="nav_history", on_click=lambda: setattr(st.session_state, 'page', "history"))
                st.button("📖 Stories", key="nav_stories", on_click=lambda: setattr(st.session_state, 'page', "stories"))
                st.button("💰 Pricing", key="nav_pricing", on_click=lambda: setattr(st.session_state, 'page', "pricing"))
            st.button("📬 Contact", key="nav_contact", on_click=lambda: setattr(st.session_state, 'page', "contact"))
            st.button("ℹ️ About", key="nav_about", on_click=lambda: setattr(st.session_state, 'page', "about"))
            if st.session_state.is_admin:
                st.button("👑 Admin", key="nav_admin", on_click=lambda: setattr(st.session_state, 'page', "admin"))

            st.markdown("---") 
            if not st.session_state.logged_in:
                st.button("🔑 Login", key="nav_login", on_click=lambda: setattr(st.session_state, 'page', "login"))
                st.button("📝 Register", key="nav_register", on_click=lambda: setattr(st.session_state, 'page', "register"))
            else:
                if st.button("🚪 Logout", key="nav_logout"):
                    for key in list(st.session_state.keys()):
                        del st.session_state[key]
                    st.session_state.page = "landing" 
                    st.session_state.initial_greeting_sent = False 
                    st.rerun()

    if st.session_state.page == "landing":
        show_landing_page()
    elif st.session_state.page == "chat_page": 
        show_chat()
    elif st.session_state.page == "login":
        show_login()
    elif st.session_state.page == "register":
        show_register()
    elif st.session_state.page == "about":
        show_about()
    elif st.session_state.page == "contact":
        show_contact()
    elif st.session_state.page == "pricing":
        show_pricing()
    elif st.session_state.page == "history":
        show_history()
    elif st.session_state.page == "stories":
        show_stories()
    elif st.session_state.page == "admin":
        show_admin()

if __name__ == "__main__":
    main()
