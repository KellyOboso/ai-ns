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
import re # For regex in normalization and entity extraction

# --- Configuration and Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = 'data'
MODELS_DIR = 'models'
TRAINING_DB_PATH = os.path.join(DATA_DIR, 'mental_health_chatbot.db') 
INTENTS_FILE = os.path.join(DATA_DIR, 'intents.yml')
TRAINING_DATA_FILE = os.path.join(DATA_DIR, 'training_data.yml')

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# --- NLTK Downloads (for TextBlob in train.py) ---
# Ensure these are downloaded when train.py runs, to prepare environment
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords', quiet=True)
try:
    nltk.data.find('taggers/averaged_perceptron_tagger') # Needed for pos_tag
except nltk.downloader.DownloadError:
    nltk.download('averaged_perceptron_tagger', quiet=True)


# Kenyan-specific stopwords (consistent with app.py)
kenyan_stopwords = {'kwani', 'sasa', 'hapo', 'bado', 'sana', 'tu', 'kwa', 'mambo', 'niaje', 'sema', 'msee', 'fiti', 'nzuri', 'barikiwa', 'shukran', 'pole', 'sad', 'huzuni', 'mbaya', 'stress', 'pressure', 'kufa', 'shida', 'majuto', 'not okay', 'depressed', 'afraid', 'low', 'worthless', 'hopeless', 'terrible', 'awful', 'bad', 'maisha ni ngumu', 'nimechoka', 'broken'}
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
    tokens = nltk.word_tokenize(text) 
    # Use global stop_words defined at top
    tokens = [word for word in tokens if word not in stop_words and len(word) > 1] 
    return ' '.join(tokens)

def detect_language_mix(text):
    """Detect language mix in Kenyan context."""
    swahili_words = {'sasa', 'mambo', 'sema', 'niaje', 'poa', 'habari', 'asubuhi', 'jioni', 'mchana', 'kwaheri', 'ndugu', 'dada', 'karibu', 'asante', 'pole', 'nimechoka', 'shida', 'jina', 'maisha', 'ndiyo', 'na' 'am', 'sawa', 'umeamkaje', 'jioni'}
    sheng_words = {'msee', 'fiti', 'nare', 'mbogi', 'radhi', 'buda', 'supa', 'manze', 'vipi'}
    
    has_swahili = any(word in text.lower() for word in swahili_words)
    has_sheng = any(word in text.lower() for word in sheng_words)
    has_english = any(word in text.lower() for word in {'how', 'what', 'when', 'why', 'feel', 'help', 'is', 'am', 'i', 'you', 'my', 'me', 'not', 'okay', 'the', 'a', 'to', 'for', 'about', 'can', 'do', 'i\'m'})
    
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
    kenyan_roles = {'mganga', 'mama mboga', 'chama', 'daktari', 'ndugu', 'mzee', 'baba', 'mama', 'sista', 'bro', 'teacher', 'pastor', 'sheikh'} 

    for word, pos in tagged:
        word_lower = word.lower()
        if pos.startswith(('NN', 'JJ')) or word_lower in kenyan_locations or word_lower in kenyan_roles:
            entities.append(word)
    
    return ', '.join(list(set(entities))[:5]) 

# --- Data Loading and Preparation for Training ---
def load_and_prepare_data():
    """Loads data from all YAML files and prepares it for training."""
    
    # Paths for all YAMLs that contribute training data
    intents_file_path = INTENTS_FILE
    training_data_file_path = TRAINING_DATA_FILE
    coping_strategies_file_path = COPING_STRATEGIES_FILE
    empathy_and_feelings_file_path = EMPATHY_AND_FEELINGS_FILE
    crisis_support_file_path = CRISIS_SUPPORT_FILE
    affirmations_file_path = AFFIRMATIONS_FILE
    resources_and_crisis_file_path = RESOURCES_AND_CRISIS_FILE # For mythbusters, etc.
    greetings_file_path = GREETINGS_FILE # For greeting patterns


    # Load all YAML files
    intents_data = load_yaml(intents_file_path)
    training_samples_data = load_yaml(training_data_file_path)
    coping_strategies_data = load_yaml(coping_strategies_file_path)
    empathy_data = load_yaml(empathy_and_feelings_file_path)
    crisis_data = load_yaml(crisis_support_file_path)
    affirmations_data = load_yaml(affirmations_file_path)
    resources_data = load_yaml(resources_and_crisis_file_path)
    greetings_data = load_yaml(greetings_file_path)


    X = [] 
    y = [] 
    processed_training_data_log = [] 

    # Helper function to add data to X, y, and log
    def add_training_data(input_text, intent_tag, response_output, source_file, metadata=None):
        if not input_text or not intent_tag:
            return
        
        X.append(normalize_text(input_text))
        y.append(intent_tag)
        
        sentiment_val = analyze_sentiment_train(input_text)
        entities_val = extract_kenyan_entities(input_text)
        language_mix_val = detect_language_mix(input_text)
        urgency_val = metadata.get('risk_level', 'low') if metadata else 'low'
        
        processed_training_data_log.append({
            'input_text': input_text,
            'normalized_text': normalize_text(input_text),
            'output': response_output,
            'intent': intent_tag,
            'sentiment': sentiment_val,
            'entities': entities_val,
            'urgency_level': urgency_val,
            'language_mix': language_mix_val,
            'source_file': os.path.basename(source_file)
        })

    # --- 1. Process data from intents.yml (Primary source of intents) ---
    if intents_data and 'intents' in intents_data:
        for intent_entry in intents_data['intents']:
            tag = intent_entry.get('tag')
            patterns = intent_entry.get('patterns', [])
            responses = intent_entry.get('responses', [])
            example_response = random.choice(responses) if responses else "No response defined."
            metadata = intent_entry.get('metadata', {})

            for pattern in patterns:
                add_training_data(pattern, tag, example_response, intents_file_path, metadata)

    # --- 2. Process data from training_data.yml (Conversation Samples, Special Cases, Localized) ---
    if training_samples_data:
        # conversation_samples
        for sample in training_samples_data.get('conversation_samples', []):
            add_training_data(sample.get('input'), sample.get('intent'), sample.get('output'), training_data_file_path, sample.get('metadata', {}))
        
        # special_cases
        for case_type, cases in training_samples_data.get('special_cases', {}).items():
            for case in cases:
                add_training_data(case.get('input'), case.get('intent', case_type), case.get('output'), training_data_file_path, case.get('metadata', {}))
        
        # localized_content
        for content_item in training_samples_data.get('localized_content', []):
            add_training_data(content_item.get('input'), content_item.get('intent'), content_item.get('output'), training_data_file_path, content_item.get('metadata', {}))
        
        # crisis_interventions (from training_data.yml)
        for intervention in training_samples_data.get('crisis_interventions', []):
            add_training_data(intervention.get('input'), intervention.get('intent'), intervention.get('output'), training_data_file_path, intervention.get('metadata', {}))
        
        # training_cautions (negative examples for logging, not classification input)
        for caution in training_samples_data.get('training_cautions', []):
            add_training_data(caution.get('input'), 'user_input_caution', caution.get('preferred_response', ''), training_data_file_path)


    # --- 3. Process additional patterns/responses from other YAMLs (for logging completeness) ---
    # These often contain responses that aren't direct intent patterns but are good for logging.

    # Coping Strategies - interactive_responses
    if coping_strategies_data and 'interactive_responses' in coping_strategies_data:
        for item in coping_strategies_data['interactive_responses']:
            if 'trigger' in item and 'response' in item:
                # Use a specific intent for these, or create one like 'coping_response_trigger'
                add_training_data(item['trigger'], 'seek_coping_strategies', item['response'], coping_strategies_file_path, item.get('metadata', {}))

    # Crisis Support - response_protocols (scripts for actual bot output, not new intent patterns)
    if crisis_data and 'response_protocols' in crisis_data:
        for protocol_name, protocol_data in crisis_data['response_protocols'].items():
            if 'triggers' in protocol_data and 'script' in protocol_data:
                example_script = random.choice(protocol_data['script']) if protocol_data['script'] else "No script defined."
                for trigger in protocol_data['triggers']:
                    add_training_data(trigger, f'crisis_{protocol_name}', example_script, crisis_support_file_path, protocol_data.get('metadata', {}))

    # Empathy and Feelings - emotional_states
    if empathy_data and 'emotional_states' in empathy_data:
        for state_name, state_data in empathy_data['emotional_states'].items():
            if 'triggers' in state_data and 'responses' in state_data:
                example_response = random.choice(state_data['responses']) if state_data['responses'] else "No response defined."
                for trigger in state_data['triggers']:
                    add_training_data(trigger, f'empathy_{state_name}', example_response, empathy_and_feelings_file_path, state_data.get('metadata', {}))
            if 'cultural_emotions' in empathy_data:
                 for state_name, state_data in empathy_data['cultural_emotions'].items():
                     if 'triggers' in state_data and 'responses' in state_data:
                         example_response = random.choice(state_data['responses']) if state_data['responses'] else "No response defined."
                         for trigger in state_data['triggers']:
                             add_training_data(trigger, f'cultural_empathy_{state_name}', example_response, empathy_and_feelings_file_path, state_data.get('metadata', {}))


    # Affirmations - core_affirmations and contextual_affirmations (patterns are implicit, use main intent)
    if affirmations_data:
        if 'core_affirmations' in affirmations_data:
            for cat_name, cat_data in affirmations_data['core_affirmations'].items():
                if 'list' in cat_data:
                    for item in cat_data['list']:
                        # Affirmations are responses, but their request triggers an intent
                        # We are assuming 'seek_affirmation' intent covers these.
                        # Adding examples like "Give me an affirmation on {cat_name}" as patterns.
                        if cat_name == 'self_worth':
                            add_training_data(f"Give me an affirmation on self-worth", 'seek_affirmation', item.get('text'), affirmations_file_path)
                        elif cat_name == 'resilience':
                            add_training_data(f"Give me an affirmation on resilience", 'seek_affirmation', item.get('text'), affirmations_file_path)
                        elif cat_name == 'community':
                            add_training_data(f"Give me an affirmation on community", 'seek_affirmation', item.get('text'), affirmations_file_path)

        if 'contextual_affirmations' in affirmations_data:
            for cat_name, cat_data in affirmations_data['contextual_affirmations'].items():
                if 'list' in cat_data:
                    for item in cat_data['list']:
                        # These are handled by seek_contextual_affirmation intent
                        # No need to add patterns here as intents.yml already handles.
                        pass


    # Resources and Crisis - mythbusters (myth is input, fact is response)
    if resources_data and 'mythbusters' in resources_data:
        if 'myths' in resources_data['mythbusters'] and isinstance(resources_data['mythbusters']['myths'], list):
            for item in resources_data['mythbusters']['myths']:
                if 'myth' in item and 'fact' in item:
                    add_training_data(item['myth'], 'address_stigma/myth', item['fact'], resources_and_crisis_file_path)
    
    # Greetings - specific time-based/cultural greetings (for logging)
    if greetings_data:
        if 'cultural_greetings' in greetings_data:
            for lang_cat, entries in greetings_data['cultural_greetings'].items():
                if isinstance(entries, dict):
                    for sub_cat, patterns in entries.items():
                        for pattern in patterns:
                            add_training_data(pattern, 'greeting', f"Hello ({lang_cat}, {sub_cat})", greetings_file_path)
                elif isinstance(entries, list): # Sheng direct list
                    for pattern in entries:
                        add_training_data(pattern, 'greeting', f"Hello ({lang_cat})", greetings_file_path)
        if 'time_sensitive' in greetings_data:
            for time_cat, patterns in greetings_data['time_sensitive'].items():
                for pattern in patterns:
                    add_training_data(pattern, 'greeting', f"Good {time_cat}", greetings_file_path)


    if not X:
        logger.error("No training data loaded. Check YAML files.")
        return [], [], []

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
