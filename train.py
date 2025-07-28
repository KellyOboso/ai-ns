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
from textblob import TextBlob # Added for sentiment analysis consistency in train.py
import nltk # Added for TextBlob dependency

# --- Configuration and Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = 'data'
MODELS_DIR = 'models'
TRAINING_DB_PATH = os.path.join(DATA_DIR, 'mental_health_chatbot.db') 
INTENTS_FILE = os.path.join(DATA_DIR, 'intents.yml')
TRAINING_DATA_FILE = os.path.join(DATA_DIR, 'training_data.yml')
# Added paths for other YAMLs used in loading for logging purposes
COPING_STRATEGIES_FILE = os.path.join(DATA_DIR, 'coping_strategies.yml') 
CRISIS_SUPPORT_FILE = os.path.join(DATA_DIR, 'crisis_support.yml')
EMPATHY_AND_FEELINGS_FILE = os.path.join(DATA_DIR, 'empathy_and_feelings.yml')
GREETINGS_FILE = os.path.join(DATA_DIR, 'greetings.yml')
AFFIRMATIONS_FILE = os.path.join(DATA_DIR, 'affirmations.yml')
RESOURCES_AND_CRISIS_FILE = os.path.join(DATA_DIR, 'resources_and_crisis.yml')

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# --- NLTK Downloads (for TextBlob in train.py) ---
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords', quiet=True)

# Kenyan-specific stopwords (consistent with app.py)
kenyan_stopwords = {'kwani', 'sasa', 'hapo', 'bado', 'sana', 'tu', 'kwa', 'mambo', 'niaje', 'sema', 'msee', 'fiti', 'nzuri', 'barikiwa', 'shukran', 'pole', 'sad', 'huzuni', 'mbaya', 'stress', 'pressure', 'kufa', 'shida', 'majuto', 'not okay', 'depressed', 'afraid', 'low', 'worthless', 'hopeless', 'terrible', 'awful', 'bad', 'maisha ni ngumu', 'nimechoka', 'broken'}
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english')).union(kenyan_stopwords)


# --- Database Functions ---
def init_db():
    """Initialize database for training data log."""
    os.makedirs(os.path.dirname(TRAINING_DB_PATH), exist_ok=True) 
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
        with sqlite3.connect(TRAINING_DB_PATH) as conn: 
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
        with sqlite3.connect(TRAINING_DB_PATH) as conn: 
            cursor = conn.cursor()
            cursor.execute("DELETE FROM bot_responses;")
            conn.commit()
        logger.info("Clearing existing training data log in bot_responses table...")
    except sqlite3.Error as e:
        logger.error(f"Database error clearing training data log: {e}")

# --- Text Processing Functions (Consistent with app.py's sentiment) ---

def analyze_sentiment_train(text):
    """Sentiment analysis using TextBlob, consistent with app.py."""
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
        logger.error(f"Error in analyze_sentiment_train with TextBlob: {e}")
        return 0.0

def normalize_text(text):
    """Enhanced text normalization for Kenyan English, Swahili, and Sheng."""
    # Ensure this is consistent with how the app's classifier preprocesses text if any
    # (Currently, app.py uses simple lower().strip() before passing to model)
    text = text.lower()
    text = re.sub(r'[^\w\s\'\-]', '', text)  
    tokens = nltk.word_tokenize(text) # Ensure nltk.word_tokenize is used
    tokens = [word for word in tokens if word not in stop_words and len(word) > 1] # len > 1 to keep short words
    return ' '.join(tokens)

def detect_language_mix(text):
    """Detect language mix in Kenyan context."""
    swahili_words = {'sasa', 'mambo', 'sema', 'niaje', 'poa', 'habari', 'asubuhi', 'jioni', 'mchana', 'kwaheri', 'ndugu', 'dada', 'karibu', 'asante', 'pole', 'nimechoka', 'shida', 'jina', 'maisha', 'ndiyo', 'na' 'am', 'sawa'}
    sheng_words = {'msee', 'fiti', 'nare', 'mbogi', 'radhi', 'buda', 'supa', 'manze'}
    
    has_swahili = any(word in text.lower() for word in swahili_words)
    has_sheng = any(word in text.lower() for word in sheng_words)
    has_english = any(word in text.lower() for word in {'how', 'what', 'when', 'why', 'feel', 'help', 'is', 'am', 'i', 'you', 'my', 'me', 'not', 'okay', 'the'})
    
    lang_flags = {
        'swahili': has_swahili,
        'sheng': has_sheng,
        'english': has_english
    }

    if lang_flags['swahili'] and lang_flags['sheng'] and lang_flags['english']:
        return 'swahili_sheng_english_mix'
    elif lang_flags['swahili'] and lang_flags['sheng']:
        return 'swahili_sheng_mix'
    elif lang_flags['swahili'] and lang_flags['english']:
        return 'swahili_english_mix'
    elif lang_flags['sheng'] and lang_flags['english']:
        return 'sheng_english_mix'
    elif lang_flags['sheng']:
        return 'sheng'
    elif lang_flags['swahili']:
        return 'swahili'
    return 'english' 

def extract_kenyan_entities(text):
    """Entity extraction for Kenyan context."""
    tokens = nltk.word_tokenize(text)
    tagged = nltk.pos_tag(tokens)
    
    entities = []
    kenyan_locations = {'nairobi', 'mombasa', 'kisumu', 'nakuru', 'eldoret', 'thika', 'kenya', 'umoja', 'kibera', 'donholm'}
    kenyan_roles = {'mganga', 'mama mboga', 'chama', 'daktari', 'ndugu', 'mzee'} 

    for word, pos in tagged:
        word_lower = word.lower()
        if pos.startswith(('NN', 'JJ')) or word_lower in kenyan_locations or word_lower in kenyan_roles:
            entities.append(word)
    
    return ', '.join(list(set(entities))[:5]) 

# --- Data Loading and Preparation for Training ---
def load_and_prepare_data():
    """Loads data from intents.yml and training_data.yml and prepares it for training."""
    intents_data = load_yaml(INTENTS_FILE)
    training_samples_data = load_yaml(TRAINING_DATA_FILE)

    if not intents_data or 'intents' not in intents_data:
        logger.error("Intents data not loaded or malformed.")
        return [], [], []

    X = [] 
    y = [] 
    processed_training_data_log = [] 

    # Process data from intents.yml
    for intent in intents_data['intents']:
        tag = intent['tag']
        patterns = intent.get('patterns', [])
        responses = intent.get('responses', []) 
        
        urgency_level = intent.get('metadata', {}).get('risk_level', 'low')
        
        for pattern in patterns:
            X.append(normalize_text(pattern)) # Use normalize_text for classifier
            y.append(tag)
            
            # Log data for bot_responses table
            processed_training_data_log.append({
                'input_text': pattern,
                'normalized_text': normalize_text(pattern),
                'output': random.choice(responses) if responses else "No response defined.",
                'intent': tag,
                'sentiment': analyze_sentiment_train(pattern), # Use sentiment from TextBlob
                'entities': extract_kenyan_entities(pattern), # Extract entities
                'urgency_level': urgency_level,
                'language_mix': detect_language_mix(pattern), # Detect language mix
                'source_file': 'intents.yml'
            })

    # Process data from training_data.yml (conversation_samples and special_cases)
    if training_samples_data:
        conversation_samples = training_samples_data.get('conversation_samples', [])
        special_cases = training_samples_data.get('special_cases', {})
        localized_content = training_samples_data.get('localized_content', [])
        crisis_interventions = training_samples_data.get('crisis_interventions', [])
        training_cautions = training_samples_data.get('training_cautions', [])


        for sample in conversation_samples:
            input_text = sample.get('input')
            output_text = sample.get('output')
            metadata = sample.get('metadata', {})
            
            if input_text and output_text:
                X.append(normalize_text(input_text)) # Use normalize_text
                sample_intent = sample.get('intent')
                if sample_intent:
                    y.append(sample_intent)
                elif metadata.get('tags'):
                    y.append(metadata.get('tags')[0]) 
                else:
                    y.append("unknown_intent")
                
                processed_training_data_log.append({
                    'input_text': input_text,
                    'normalized_text': normalize_text(input_text),
                    'output': output_text,
                    'intent': sample_intent if sample_intent else (metadata.get('tags', ["unknown_intent"])[0] if metadata.get('tags') else "unknown_intent"),
                    'sentiment': analyze_sentiment_train(input_text), 
                    'entities': extract_kenyan_entities(input_text), 
                    'urgency_level': metadata.get('urgency', 'low') if 'urgency' in metadata else 'low', # Use 'urgency' from metadata
                    'language_mix': detect_language_mix(input_text),
                    'source_file': 'training_data.yml'
                })

        for case_type, cases in special_cases.items(): # e.g., 'trauma', 'psychosis'
            for case in cases:
                input_text = case.get('input')
                output_text = case.get('output')
                metadata = case.get('metadata', {})
                
                if input_text and output_text:
                    X.append(normalize_text(input_text))
                    case_intent = case.get('intent')
                    if case_intent:
                        y.append(case_intent)
                    else:
                        y.append(case_type) 
                    
                    processed_training_data_log.append({
                        'input_text': input_text,
                        'normalized_text': normalize_text(input_text),
                        'output': output_text,
                        'intent': case_intent if case_intent else case_type,
                        'sentiment': analyze_sentiment_train(input_text), 
                        'entities': extract_kenyan_entities(input_text), 
                        'urgency_level': metadata.get('urgency', 'low') if 'urgency' in metadata else 'low', # Use 'urgency' from metadata
                        'language_mix': detect_language_mix(input_text),
                        'source_file': 'training_data.yml'
                    })

        for content in localized_content:
            input_text = content.get('input')
            output_text = content.get('output')
            metadata = content.get('metadata', {})

            if input_text and output_text:
                X.append(normalize_text(input_text))
                content_intent = content.get('intent')
                if content_intent:
                    y.append(content_intent)
                else:
                    y.append(metadata.get('tag', 'localized_content')) 
                
                processed_training_data_log.append({
                    'input_text': input_text,
                    'normalized_text': normalize_text(input_text),
                    'output': output_text,
                    'intent': content_intent if content_intent else metadata.get('tag', 'localized_content'),
                    'sentiment': analyze_sentiment_train(input_text), 
                    'entities': extract_kenyan_entities(input_text), 
                    'urgency_level': metadata.get('urgency', 'low') if 'urgency' in metadata else 'low', 
                    'language_mix': detect_language_mix(input_text),
                    'source_file': 'training_data.yml'
                })
        
        # Add crisis_interventions from training_data.yml
        for intervention in crisis_interventions:
            input_text = intervention.get('input')
            output_text = intervention.get('output')
            metadata = intervention.get('metadata', {})
            if input_text and output_text:
                X.append(normalize_text(input_text))
                y.append(intervention.get('intent', 'crisis_intervention_unknown')) # Use intent or generic tag
                processed_training_data_log.append({
                    'input_text': input_text,
                    'normalized_text': normalize_text(input_text),
                    'output': output_text,
                    'intent': intervention.get('intent', 'crisis_intervention_unknown'),
                    'sentiment': analyze_sentiment_train(input_text),
                    'entities': extract_kenyan_entities(input_text),
                    'urgency_level': metadata.get('urgency', 'critical'),
                    'language_mix': detect_language_mix(input_text),
                    'source_file': 'training_data.yml'
                })
        
        # Add training_cautions (negative examples) from training_data.yml
        for caution in training_cautions:
            input_text = caution.get('input')
            preferred_response = caution.get('preferred_response')
            if input_text and preferred_response:
                X.append(normalize_text(input_text))
                y.append('user_input_caution') # Give a specific intent for these
                processed_training_data_log.append({
                    'input_text': input_text,
                    'normalized_text': normalize_text(input_text),
                    'output': preferred_response,
                    'intent': 'user_input_caution',
                    'sentiment': analyze_sentiment_train(input_text),
                    'entities': extract_kenyan_entities(input_text),
                    'urgency_level': 'low', # These are training notes, not crisis
                    'language_mix': detect_language_mix(input_text),
                    'source_file': 'training_data.yml'
                })

    return X, y, processed_training_data_log

# --- Model Training ---
def train_model(X, y):
    """Trains the intent classification model."""
    if not X or not y:
        logger.warning("No data to train the model.")
        return None, None

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
    except ValueError as e:
        logger.warning(f"Could not stratify split due to insufficient samples in some classes: {e}. Attempting non-stratified split.")
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('svm', SVC(kernel='linear', probability=True, random_state=42)) 
    ])

    logger.info(f"Training classifier with {len(X_train)} samples across {len(label_encoder.classes_)} intents.")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    
    unique_y_test = np.unique(y_test)
    valid_labels_in_test_set = [label for label in unique_y_test if label < len(label_encoder.classes_)]
    
    if len(valid_labels_in_test_set) > 0: 
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

    logger.info("⏳ Processing and storing all training data attributes in database...")
    clear_training_data_log() 
    for entry in processed_training_data_log:
        log_training_data_entry(entry)
    logger.info(f"Stored {len(processed_training_data_log)} processed training data entries in database for analysis.")
    logger.info("✅ Training complete.")

if __name__ == "__main__":
    main()
