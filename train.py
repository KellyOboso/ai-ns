import yaml
import os
import logging
import sqlite3
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np 
import random 

# --- Configuration and Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = 'data'
MODELS_DIR = 'models'
# train.py will continue to use this DB for logging training data attributes (bot_responses table)
TRAINING_DB_PATH = os.path.join(DATA_DIR, 'mental_health_chatbot.db') 
INTENTS_FILE = os.path.join(DATA_DIR, 'intents.yml')
TRAINING_DATA_FILE = os.path.join(DATA_DIR, 'training_data.yml')

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# --- Database Functions ---
def init_db():
    """Initialize database for training data log."""
    os.makedirs(os.path.dirname(TRAINING_DB_PATH), exist_ok=True) # Ensure data directory exists
    with sqlite3.connect(TRAINING_DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS bot_responses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                input_text TEXT,
                normalized_text TEXT,
                response_text TEXT,
                intent_category TEXT,
                sentiment REAL,
                entities TEXT,
                urgency_level TEXT CHECK(urgency_level IN ('low', 'medium', 'high', 'critical')),
                language_mix TEXT,
                source_file TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
    logger.info("Database initialized with Kenyan context schema for training data log.")

def log_training_data_entry(entry):
    """Logs a single processed training data entry to the database."""
    try:
        with sqlite3.connect(TRAINING_DB_PATH) as conn: # Use TRAINING_DB_PATH
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO bot_responses (input_text, normalized_text, response_text, intent_category, sentiment, entities, urgency_level, language_mix, source_file)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (entry.get('input_text'), entry.get('normalized_text'), entry.get('output'),
                 entry.get('intent'), entry.get('sentiment'), str(entry.get('entities')),
                 entry.get('urgency_level'), entry.get('language_mix'), entry.get('source_file'))
            )
            conn.commit()
    except sqlite3.Error as e:
        logger.error(f"Database error logging training data entry: {e}")

def clear_training_data_log():
    """Clears existing training data log in the bot_responses table."""
    try:
        with sqlite3.connect(TRAINING_DB_PATH) as conn: # Use TRAINING_DB_PATH
            cursor = conn.cursor()
            cursor.execute("DELETE FROM bot_responses;")
            conn.commit()
        logger.info("Clearing existing training data log in bot_responses table...")
    except sqlite3.Error as e:
        logger.error(f"Database error clearing training data log: {e}")

# --- Data Loading and Preprocessing ---
def load_yaml(filepath):
    """Loads a YAML file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        logger.error(f"Error: {filepath} not found.")
        return None
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file {filepath}: {e}")
        return None

def preprocess_text(text):
    """Basic text preprocessing."""
    return text.lower().strip()

def load_and_prepare_data():
    """Loads data from intents.yml and training_data.yml and prepares it for training."""
    intents_data = load_yaml(INTENTS_FILE)
    training_samples_data = load_yaml(TRAINING_DATA_FILE)

    if not intents_data or 'intents' not in intents_data:
        logger.error("Intents data not loaded or malformed.")
        return [], [], []

    X = [] # Features (user inputs)
    y = [] # Labels (intents)
    processed_training_data_log = [] # Data to log to DB

    # Process data from intents.yml
    for intent in intents_data['intents']:
        tag = intent['tag']
        patterns = intent.get('patterns', [])
        responses = intent.get('responses', []) # Get responses for logging
        
        # Determine urgency level for logging, default to 'low'
        urgency_level = intent.get('metadata', {}).get('risk_level', 'low')
        
        for pattern in patterns:
            X.append(preprocess_text(pattern))
            y.append(tag)
            
            # Log data for bot_responses table
            processed_training_data_log.append({
                'input_text': pattern,
                'normalized_text': preprocess_text(pattern),
                'output': random.choice(responses) if responses else "No response defined.",
                'intent': tag,
                'sentiment': 0.0, # Placeholder, can be improved with a real sentiment model
                'entities': {}, # Placeholder
                'urgency_level': urgency_level,
                'language_mix': 'mixed_swahili_sheng_english', # Example
                'source_file': 'intents.yml'
            })

    # Process data from training_data.yml (conversation_samples and special_cases)
    if training_samples_data:
        conversation_samples = training_samples_data.get('conversation_samples', [])
        special_cases = training_samples_data.get('special_cases', {})
        localized_content = training_samples_data.get('localized_content', [])

        for sample in conversation_samples:
            input_text = sample.get('input')
            output_text = sample.get('output')
            metadata = sample.get('metadata', {})
            
            if input_text and output_text:
                X.append(preprocess_text(input_text))
                # Use the 'intent' key from the sample if available, otherwise fallback to 'tags' or 'unknown_intent'
                sample_intent = sample.get('intent')
                if sample_intent:
                    y.append(sample_intent)
                elif metadata.get('tags'):
                    y.append(metadata.get('tags')[0]) # Use first tag as intent
                else:
                    y.append("unknown_intent")
                
                processed_training_data_log.append({
                    'input_text': input_text,
                    'normalized_text': preprocess_text(input_text),
                    'output': output_text,
                    'intent': sample_intent if sample_intent else (metadata.get('tags', ["unknown_intent"])[0] if metadata.get('tags') else "unknown_intent"),
                    'sentiment': 0.0, # Placeholder
                    'entities': {}, # Placeholder
                    'urgency_level': metadata.get('crisis_level', 'low') if 'crisis_level' in metadata else 'low',
                    'language_mix': 'mixed_swahili_sheng_english',
                    'source_file': 'training_data.yml'
                })

        for case_type, cases in special_cases.items():
            for case in cases:
                input_text = case.get('input')
                output_text = case.get('output')
                metadata = case.get('metadata', {})
                
                if input_text and output_text:
                    X.append(preprocess_text(input_text))
                    # Use the 'intent' key from the case if available, otherwise fallback to case_type
                    case_intent = case.get('intent')
                    if case_intent:
                        y.append(case_intent)
                    else:
                        y.append(case_type) # Use case_type as intent (e.g., 'trauma', 'psychosis')
                    
                    processed_training_data_log.append({
                        'input_text': input_text,
                        'normalized_text': preprocess_text(input_text),
                        'output': output_text,
                        'intent': case_intent if case_intent else case_type,
                        'sentiment': 0.0, # Placeholder
                        'entities': {}, # Placeholder
                        'urgency_level': metadata.get('crisis_level', 'low') if 'crisis_level' in metadata else 'low',
                        'language_mix': 'mixed_swahili_sheng_english',
                        'source_file': 'training_data.yml'
                    })

        for content in localized_content:
            input_text = content.get('input')
            output_text = content.get('output')
            metadata = content.get('metadata', {})

            if input_text and output_text:
                X.append(preprocess_text(input_text))
                # Use the 'intent' key from the content if available, otherwise fallback to 'localized_content'
                content_intent = content.get('intent')
                if content_intent:
                    y.append(content_intent)
                else:
                    y.append(metadata.get('tag', 'localized_content')) # Use a generic tag or specific if available
                
                processed_training_data_log.append({
                    'input_text': input_text,
                    'normalized_text': preprocess_text(input_text),
                    'output': output_text,
                    'intent': content_intent if content_intent else metadata.get('tag', 'localized_content'),
                    'sentiment': 0.0, # Placeholder
                    'entities': {}, # Placeholder
                    'urgency_level': 'low', # Default for localized content
                    'language_mix': 'mixed_swahili_sheng_english',
                    'source_file': 'training_data.yml'
                })


    return X, y, processed_training_data_log

# --- Model Training ---
def train_model(X, y):
    """Trains the intent classification model."""
    if not X or not y:
        logger.warning("No data to train the model.")
        return None, None

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Split data
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
    except ValueError as e:
        logger.warning(f"Could not stratify split due to insufficient samples in some classes: {e}. Attempting non-stratified split.")
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)


    # Create a pipeline: TF-IDF Vectorizer -> SVM Classifier
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('svm', SVC(kernel='linear', probability=True, random_state=42)) # probability=True is needed for predict_proba
    ])

    logger.info(f"Training classifier with {len(X_train)} samples across {len(label_encoder.classes_)} intents.")
    pipeline.fit(X_train, y_train)

    # Evaluate the model
    y_pred = pipeline.predict(X_test)
    
    # Get unique labels present in the test set
    unique_y_test = np.unique(y_test)
    
    # Filter labels that are actually present in the label_encoder's classes
    valid_labels_in_test_set = [label for label in unique_y_test if label < len(label_encoder.classes_)]
    
    if len(valid_labels_in_test_set) > 0: # Check if the list of valid labels is not empty
        target_names = [label_encoder.inverse_transform([i])[0] for i in valid_labels_in_test_set]
        report = classification_report(y_test, y_pred, labels=valid_labels_in_test_set, target_names=target_names, zero_division=0)
        logger.info(f"\nClassification Report:\n{report}")
    else:
        logger.warning("No valid labels in test set for classification report. Skipping report generation.")
        logger.info(f"y_test was: {y_test}")
        logger.info(f"unique_y_test was: {unique_y_test}")
        logger.info(f"label_encoder.classes_ length: {len(label_encoder.classes_)}")
    
    return pipeline, label_encoder 

# --- Main Execution ---
def main():
    init_db()
    
    X, y, processed_training_data_log = load_and_prepare_data()
    
    if not X or not y:
        logger.error("Failed to load or prepare data. Exiting training.")
        return

    # Train model
    pipeline, label_encoder = train_model(X, y) 

    if pipeline and label_encoder: 
        model_path = os.path.join(MODELS_DIR, 'intent_classifier.joblib')
        label_encoder_path = os.path.join(MODELS_DIR, 'label_encoder.joblib') 

        joblib.dump(pipeline, model_path) 
        joblib.dump(label_encoder, label_encoder_path) 
        logger.info(f"Model trained and saved with Kenyan language features to {model_path}")
        logger.info(f"Label Encoder saved to {label_encoder_path}")
    else:
        logger.error("Model training failed. Skipping save.")

    # Log all processed training data attributes to the database
    logger.info("⏳ Processing and storing all training data attributes in database...")
    clear_training_data_log() 
    for entry in processed_training_data_log:
        log_training_data_entry(entry)
    logger.info(f"Stored {len(processed_training_data_log)} processed training data entries in database for analysis.")
    logger.info("✅ Training complete.")

if __name__ == "__main__":
    main()
