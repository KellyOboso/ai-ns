```python
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
from textblob import TextBlob 
import nltk 
import re 

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
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords', quiet=True)
try:
    nltk.data.find('taggers/averaged_perceptron_tagger') 
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
    text = text.lower()
    text = re.sub(r'[^\w\s\'\-]', '', text)  
    tokens = nltk.word_tokenize(text) 
    stop_words_local = set(stopwords.words('english')).union(kenyan_stopwords) # Ensure stop_words is local or re-defined if needed
    tokens = [word for word in tokens if word not in stop_words_local and len(word) > 1] 
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
    
    intents_file_path = INTENTS_FILE
    training_data_file_path = TRAINING_DATA_FILE
    coping_strategies_file_path = os.path.join(DATA_DIR, 'coping_strategies.yml') 
    empathy_and_feelings_file_path = os.path.join(DATA_DIR, 'empathy_and_feelings.yml')
    crisis_support_file_path = os.path.join(DATA_DIR, 'crisis_support.yml')
    affirmations_file_path = os.path.join(DATA_DIR, 'affirmations.yml')
    resources_and_crisis_file_path = os.path.join(DATA_DIR, 'resources_and_crisis.yml') 
    greetings_file_path = os.path.join(DATA_DIR, 'greetings.yml') 


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

    def add_training_data(input_text, intent_tag, response_output, source_file, metadata=None):
        if not input_text or not intent_tag:
            return
        
        X.append(normalize_text(input_text))
        y.append(intent_tag)
        
        sentiment_val = analyze_sentiment_train(input_text)
        entities_val = extract_kenyan_entities(input_text)
        language_mix_val = detect_language_mix(input_text)
        urgency_val = metadata.get('risk_level', 'low') if metadata and 'risk_level' in metadata else (metadata.get('urgency', 'low') if metadata and 'urgency' in metadata else 'low')
        
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
            input_text = sample.get('input')
            output_text = sample.get('output')
            metadata = sample.get('metadata', {})
            sample_intent = sample.get('intent') or (metadata.get('tags', ["unknown_intent"])[0] if metadata.get('tags') else "unknown_intent")
            add_training_data(input_text, sample_intent, output_text, training_data_file_path, metadata)
        
        # special_cases
        for case_type, cases in training_samples_data.get('special_cases', {}).items():
            for case in cases:
                input_text = case.get('input')
                output_text = case.get('output')
                metadata = case.get('metadata', {})
                case_intent = case.get('intent') or case_type
                add_training_data(input_text, case_intent, output_text, training_data_file_path, metadata)
        
        # localized_content
        for content_item in training_samples_data.get('localized_content', []):
            input_text = content_item.get('input')
            output_text = content_item.get('output')
            metadata = content_item.get('metadata', {})
            content_intent = content_item.get('intent') or (metadata.get('tag') or 'localized_content')
            add_training_data(input_text, content_intent, output_text, training_data_file_path, metadata)
        
        # crisis_interventions (from training_data.yml)
        for intervention in training_samples_data.get('crisis_interventions', []):
            input_text = intervention.get('input')
            output_text = intervention.get('output')
            metadata = intervention.get('metadata', {})
            add_training_data(input_text, intervention.get('intent', 'crisis_intervention_unknown'), output_text, training_data_file_path, metadata)
        
        # training_cautions (negative examples for logging, not classification input)
        for caution in training_samples_data.get('training_cautions', []):
            add_training_data(caution.get('input'), 'user_input_caution', caution.get('preferred_response', ''), training_data_file_path)


    # --- 3. Process additional patterns/responses from other YAMLs (for logging completeness) ---
    # These sections also contain valuable patterns and responses for logging.

    # Coping Strategies - all sub-sections with names/tips/etc.
    if coping_strategies_data:
        coping_detail_map = coping_strategies_data # Use directly
        
        for category in ['core_strategies', 'cultural_strategies', 'by_scenario']:
            if category in coping_detail_map:
                for sub_cat_name, sub_cat_content in coping_detail_map[category].items():
                    if 'techniques' in sub_cat_content and isinstance(sub_cat_content['techniques'], list):
                        for tech in sub_cat_content['techniques']:
                            add_training_data(tech.get('name'), f'coping_tech_{sub_cat_name}', tech.get('how_to_guide', [''])[0], coping_strategies_file_path, tech.get('metadata', {}))
                    if 'methods' in sub_cat_content and isinstance(sub_cat_content['methods'], list):
                        for method in sub_cat_content['methods']:
                            add_training_data(method.get('name'), f'coping_method_{sub_cat_name}', method.get('how_to', ''), coping_strategies_file_path, method.get('metadata', {}))
                    if 'strategies' in sub_cat_content and isinstance(sub_cat_content['strategies'], list): # For by_scenario
                         for strategy in sub_cat_content['strategies']:
                             add_training_data(strategy.get('tip'), f'coping_scenario_{sub_cat_name}', strategy.get('how_to', ''), coping_strategies_file_path, strategy.get('metadata', {}))

        for top_level_list in ['movement_based', 'creative_outlets', 'immediate_support_tips', 'safety_planning_preventive']:
            if top_level_list in coping_detail_map and isinstance(coping_detail_map[top_level_list], list):
                for item in coping_detail_map[top_level_list]:
                    add_training_data(item.get('name') or item.get('tip'), f'coping_general_{top_level_list}', item.get('description') or item.get('how_to') or item.get('benefits', [])[0] if isinstance(item.get('benefits'),list) else item.get('benefits'), coping_strategies_file_path, item.get('metadata', {}))
            elif top_level_list in coping_detail_map and isinstance(coping_detail_map[top_level_list], dict) and 'tips' in coping_detail_map[top_level_list] and isinstance(coping_detail_map[top_level_list]['tips'], list):
                 for item in coping_detail_map[top_level_list]['tips']:
                     add_training_data(item.get('name') or item.get('tip'), f'coping_general_{top_level_list}', item.get('how_to'), coping_strategies_file_path, item.get('metadata', {}))

    # Empathy and Feelings - cultural_emotions (triggers and responses)
    if empathy_data and 'cultural_emotions' in empathy_data:
        for state_name, state_data in empathy_data['cultural_emotions'].items():
            if 'triggers' in state_data and 'responses' in state_data:
                example_response = random.choice(state_data['responses']) if state_data['responses'] else "No response defined."
                for trigger in state_data['triggers']:
                    add_training_data(trigger, f'cultural_emotion_{state_name}', example_response, empathy_and_feelings_file_path, state_data.get('metadata', {}))

    # Affirmations - core_affirmations and contextual_affirmations
    if affirmations_data:
        for cat_type in ['core_affirmations', 'contextual_affirmations']:
            if cat_type in affirmations_data:
                for cat_name, cat_data in affirmations_data[cat_type].items():
                    if 'list' in cat_data and isinstance(cat_data['list'], list):
                        for item in cat_data['list']:
                            # Using the affirmation text itself as input for logging
                            add_training_data(item.get('text'), f'affirmation_{cat_name}', item.get('meaning'), affirmations_file_path)

    # Resources and Crisis - mythbusters
    if resources_data and 'mythbusters' in resources_data and 'myths' in resources_data['mythbusters'] and isinstance(resources_data['mythbusters']['myths'], list):
        for item in resources_data['mythbusters']['myths']:
            if 'myth' in item and 'fact' in item:
                add_training_data(item['myth'], 'mythbusters_question', item['fact'], resources_and_crisis_file_path)

    # Crisis Support - kenya_emergency_services, targeted_resources (for logging info)
    if crisis_data:
        for category in ['kenya_emergency_services', 'targeted_resources']:
            if category in crisis_data and isinstance(crisis_data[category], dict):
                for sub_category, entries in crisis_data[category].items():
                    if isinstance(entries, list):
                        for entry in entries:
                            if 'name' in entry and 'contact' in entry:
                                # Log names/contacts as patterns for information retrieval
                                add_training_data(entry['name'], f'{category}_info', f"Contact: {entry['contact']}", crisis_support_file_path)
                                add_training_data(f"Contact for {entry['name']}", f'{category}_info', f"Contact: {entry['contact']}", crisis_support_file_path)

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
