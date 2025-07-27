# 💬 Kelly AI - Your Mental Health Companion 🇰🇪

## Project Overview

Kelly AI is a culturally-sensitive mental health chatbot designed specifically to support individuals in Kenya. Built with Streamlit and a custom machine learning model, Kelly AI provides accessible, empathetic, and culturally relevant mental well-being assistance, understanding the unique contexts and nuances of Kenyan life.

**Mission:** To empower Kenyans on their mental health journeys by offering a safe, confidential, and intelligent digital companion that understands and respects their cultural background.

## ✨ Features

* **Culturally Sensitive Conversations:** Understands and responds to Swahili, Sheng, common Kenyan idioms, and cultural contexts.
* **Emotional Support:** Provides empathetic listening and guidance for various emotional states (sadness, anxiety, stress, anger, loneliness, pain).
* **Crisis Intervention:** Immediate protocols for high-risk situations with direct contacts to Kenyan emergency services and helplines.
* **Coping Strategies:** Offers a diverse range of coping mechanisms, including traditional Kenyan methods, breathing exercises, grounding techniques, and creative outlets.
* **Positive Affirmations:** Delivers culturally-grounded and situation-specific affirmations to uplift and encourage users, with reflection prompts.
* **Resource Connection:** Connects users to local professional therapists, support groups, self-help articles, and relevant mobile applications in Kenya.
* **User Authentication:** Secure user registration and login system with bcrypt for password hashing.
* **Conversation History:** Logged-in users can review their past conversations.
* **Community Stories:** A platform for users to share and read anonymous mental health journeys and experiences.
* **Admin Dashboard:** Comprehensive admin interface for managing users, monitoring conversation statistics, reviewing feedback, and **adding new training data**.
* **Scalable Knowledge Base:** All bot responses and conversational flows are managed through easily editable YAML files.

## 🚀 Getting Started

Follow these steps to set up and run Kelly AI on your local machine.

### Prerequisites

* Python 3.8+
* `pip` (Python package installer)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url> # Replace with your actual repository URL
    cd Final-project
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate # On Windows: .venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up environment variables:**
    Create a file named `.env` in the root directory of your project (same level as `app.py`) and add your admin credentials:
    ```
    # .env
    ADMIN_EMAIL="kellyoboso@gmail.com"
    ADMIN_PASSWORD_HASH="PASTE_YOUR_BCRYPT_HASH_HERE"
    ```
    **To generate the `ADMIN_PASSWORD_HASH`:**
    Open your terminal and run:
    ```bash
    python -c "import bcrypt; print(bcrypt.hashpw(b'YourStrongAdminPasswordHere', bcrypt.gensalt()).decode())"
    ```
    Replace `'YourStrongAdminPasswordHere'` with your desired strong password. Copy the output and paste it into your `.env` file.

### Database Setup & Model Training

This step initializes the databases and trains the AI model. **You must run this at least once, and every time you modify the YAML data files.**

1.  **Delete old database files (if they exist from previous runs):**
    ```bash
    rm users.db data/mental_health_chatbot.db
    ```
    (On Windows, use `del users.db data\mental_health_chatbot.db` or delete them manually.)

2.  **Run the training script:**
    ```bash
    python3 train.py
    ```
    This will:
    * Create `users.db` (for user accounts, conversations, stories).
    * Create `data/mental_health_chatbot.db` (for logging training data attributes).
    * Train the NLU model based on your YAML data and save it as `models/intent_classifier.joblib` and `models/label_encoder.joblib`.

### Running the Application

1.  **Start the Streamlit app:**
    ```bash
    streamlit run app.py
    ```
2.  Your browser should automatically open to `http://localhost:8501`.

## 🤝 Usage & Interaction

Upon launching, you'll see a landing page describing Kelly AI. Click "Continue" to proceed to the chat.

* **Anonymous Chat:** You can start chatting immediately as an anonymous user.
* **Login/Register:** Use the sidebar to log in or register for an account to access features like conversation history and community stories.
    * **Admin Login:** Use the `ADMIN_EMAIL` and the password you set in your `.env` file (`YourStrongAdminPasswordHere`) to access the Admin Dashboard.

**Examples of what to ask Kelly AI:**

* **Greetings:** "Hi", "Habari yako?", "How are you?"
* **Emotional Support:** "I'm sad", "I feel depressed", "Nina huzuni", "My anxiety is through the roof", "I'm feeling stressed", "I'm so angry", "I feel lonely", "Everything just hurts."
* **Coping:** "What are some coping mechanisms?", "How can I calm down?", "Give me a grounding exercise."
* **Affirmations:** "Give me a positive affirmation", "Affirmation for job seekers", "Something positive for farmers."
* **Crisis:** "I want to die", "I'm going to hurt myself", "I'm being abused", "Call emergency."
* **Resources:** "How do I find a therapist?", "Any mental health apps?", "Tell me about traditional healing."
* **Kenyan Context:** "I'm worried about the elections", "My crops failed", "Tell me about chama groups."
* **Bot Info:** "Who are you?", "What can you do?", "Thank you."
* **Conversation Control:** "Yes", "Okay", "Start over."

## ⚙️ Admin Dashboard Features

Access the Admin Dashboard by logging in with your `ADMIN_EMAIL`.

* **Users:** View and manage user accounts.
* **Conversations & Training Data:** Monitor conversation statistics and the training data log.
* **User Feedback:** Review feedback submitted by users.
* **Manage Plans:** **(New!)** Update user subscription plans (e.g., to Premium for a month) and set expiration dates.
* **Add Training Data:** **(New - Developer Feature!)** Manually add new user input patterns and assign them to intents directly through the UI.
    * **Important:** After adding data via the Admin Dashboard, you **must manually run `python3 train.py` in your terminal** and then **restart `streamlit run app.py`** for the new data to be incorporated into the AI model. This step is for development/data curation only.

## 📁 Project Structure
.
├── .venv/                   # Python virtual environment
├── data/                    # YAML data files for bot knowledge base
│   ├── affirmations.yml
│   ├── bot_profile.yml
│   ├── coping_strategies.yml
│   ├── crisis_support.yml
│   ├── cultural_responses.yml
│   ├── empathy_and_feelings.yml
│   ├── fallbacks.yml
│   ├── greetings.yml
│   ├── intents.yml
│   ├── mental_health_chatbot.db  # Database for logging training data
│   ├── resources_and_crisis.yml
│   └── user_feedback.yml
├── models/                  # Trained AI models
│   ├── intent_classifier.joblib
│   └── label_encoder.joblib
├── .env                     # Environment variables (e.g., admin credentials)
├── app.py                   # Main Streamlit application
├── requirements.txt         # Python dependencies
├── train.py                 # Script for training the NLU model
└── users.db                 # Database for user accounts, conversations, stories, contacts


## 📈 Future Improvements (Ideas)

* **Automated Model Retraining & Deployment:** Implement a CI/CD pipeline or background service to automate retraining and model updates.
* **Advanced NLU:** Integrate with external NLU services like Dialogflow or Rasa for more sophisticated conversational understanding.
* **Voice Interface:** Add speech-to-text and text-to-speech capabilities.
* **Personalized Experience:** Store user preferences (with consent) to tailor responses and resource recommendations.
* **Sentiment Trend Analysis:** Visualize user sentiment over time in the Admin Dashboard.
* **AI-Driven Follow-ups:** More intelligent and adaptive follow-up questions based on conversation history.

## ❤️ Credits

Developed by Kelly Oboso.
