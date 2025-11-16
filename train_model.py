import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK data 
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class MentalHealthClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1,2))
        self.model = LogisticRegression(random_state=42, max_iter=1000)
        self.classes_ = None
        self.accuracy = 0
        
    def clean_text(self, text):
        if not isinstance(text, str) or not text.strip():
            return ""
        
        text = text.lower().strip()
        
        # Remove URLs, emails, phone numbers
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'\S*@\S*\s?', '', text)
        text = re.sub(r'\d{10,}', '', text)
        
        # Keep some meaningful punctuation (! and ?) for mental health context
        text = re.sub(r'[^a-zA-Z\s!?]', '', text) 
        
        # Handle repeated characters
        text = re.sub(r'(.)\1+', r'\1', text)
        
        return ' '.join(text.split())
    
    def create_training_data(self):
    
        training_data = []
        
        crisis_queries = [
            "i want to end my life", "suicidal thoughts", "thinking about suicide",
            "i want to die", "life is not worth living", "no reason to live",
            "ending it all", "i feel hopeless", "extreme emotional pain",
            "overwhelming despair", "cant take it anymore", "self harm","i cant go on like this anymore", "everything feels pointless",
            "the pain is too much to bear", "i want to disappear forever",
            "no one would care if i was gone", "i'm a burden to everyone",
            "the world would be better without me", "i have nothing left to live for",
            "this suffering needs to end", "i'm planning to kill myself",
            "thinking of ways to end it all", "i've made plans to commit suicide",
            "i have a suicide plan", "ready to end my suffering",
            "can't cope with this pain anymore", "life is unbearable",
            "i see no way out of this", "completely hopeless about everything",
            "the darkness is consuming me", "i feel trapped with no escape",
            "ending everything seems like the only option", "suicide is my only way out",
            "i've decided to kill myself", "preparing to end my life",
            "this will all be over soon", "final goodbye to everyone",
            "no hope left in my life", "completely broken beyond repair"
            "harm myself", "not safe with myself", "my existence is a burden", "I don't see the point in living anymore"
        ]
        
        gender_queries = [
            "i think i might be asexual", "struggling with my gender identity",
            "coming out as gay", "family rejection for being lgbtq",
            "unable to relate to peers about sexuality", "gender dysphoria",
            "sexual orientation confusion", "lgbtq discrimination", "questioning my sexual orientation", "not sure if i'm straight",
            "attracted to the same gender", "coming out to my parents",
            "family doesn't accept my identity", "lgbtq bullying at school",
            "workplace discrimination for being gay", "gender identity crisis",
            "want to change my gender", "hormone therapy questions",
            "sexual identity exploration", "bisexual but confused",
            "pansexual and need support", "non-binary gender expression",
            "they/them pronouns struggle", "misgendered at work",
            "transitioning socially", "medical transition concerns",
            "lgbtq relationship issues", "same-sex marriage counseling",
            "conversion therapy trauma", "religious family rejection",
            "internalized homophobia", "dating as a transgender person",
            "lgbtq mental health support", "queer community finding",
            "asexual spectrum questions", "aromantic and need guidance",
            "intersex identity support", "gender fluidity confusion",
            "cross-dressing and mental health", "lgbtq youth homelessness",
            "elderly lgbtq isolation", "rural area lgbtq loneliness",
            "cultural lgbtq conflicts", "traditional family pressure",
            "lgbtq career challenges", "workplace coming out anxiety",
            "religious lgbtq reconciliation", "spiritual lgbtq acceptance",
            "lgbtq parenting questions", "rainbow family support",
            "gender affirming care access", "surgery consultation support",
            "name change process stress", "legal gender marker change",
            "lgbtq immigration concerns", "refugee lgbtq safety",
            "caste and sexuality intersection", "regional lgbtq challenges"
            "transgender transition", "pronouns and identity", "confused about my gender", "confused about sexuality"
        ]
        
        academic_queries = [
            "jee preparation stress", "neet exam pressure", "academic burnout",
            "college entrance exam anxiety", "study pressure overwhelming",
            "cant handle academic stress", "exam performance anxiety",
            "competitive exam pressure", "parental expectations stress"
        ]
        
        relationship_queries = [
            "my husband yells at me", "marital conflict", "relationship problems",
            "partner anger issues", "emotional abuse in marriage",
            "communication issues with spouse", "marital counseling needed",
            "domestic conflict resolution"
        ]

        additional_crisis = [
        "i can't go on like this", "want to disappear forever", 
        "thinking of self harm", "life is too painful",
        "no one would miss me", "planning suicide", "feel completely alone"
        ]
    
        additional_therapy = [
            "anxiety management techniques", "coping with depression",
            "stress management counseling", "relationship counseling",
            "career guidance therapy", "family therapy needed",
            "dealing with panic attacks", "social anxiety help"
        ]

        general_queries = [
        "mental health resources", "find a therapist", 
        "psychological help", "mental wellness", "emotional support"
        ]
        
        # Add to training data with appropriate labels
        for query in crisis_queries:
            training_data.append({"text": query, "label": "crisis"})
        for query in gender_queries:
            training_data.append({"text": query, "label": "therapy"})
        for query in academic_queries:
            training_data.append({"text": query, "label": "therapy"})
        for query in relationship_queries:
            training_data.append({"text": query, "label": "therapy"})
        for query in additional_crisis:
            training_data.append({"text": query, "label": "crisis"})
        for query in additional_therapy:
            training_data.append({"text": query, "label": "therapy"})
        for query in general_queries:
            training_data.append({"text": query, "label": "therapy"})
        
        return pd.DataFrame(training_data)
    
    def train(self):
        print("Creating training data...")
        df = self.create_training_data()
        
        # Clean the text
        df['cleaned_text'] = df['text'].apply(self.clean_text)
        
        # Prepare features and labels
        X = df['cleaned_text']
        y = df['label']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Vectorize the text
        print("Vectorizing text data...")
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)
        
        # Train the model
        print("Training Logistic Regression model...")
        self.model.fit(X_train_vec, y_train)
        
        # Evaluate the model
        y_pred = self.model.predict(X_test_vec)
        self.accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Model trained with accuracy: {self.accuracy:.2f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        self.classes_ = self.model.classes_
        
        return self.accuracy
    
    def predict(self, text, confidence_threshold=0.6):
        cleaned_text = self.clean_text(text)
        text_vec = self.vectorizer.transform([cleaned_text])
        probabilities = self.model.predict_proba(text_vec)[0]
        confidence = max(probabilities)
        prediction = self.model.predict(text_vec)[0]
        
        # Return "uncertain" if confidence is low
        if confidence < confidence_threshold:
            return "uncertain", confidence, dict(zip(self.classes_, probabilities))
        
        return prediction, confidence, dict(zip(self.classes_, probabilities))
    
    def save_model(self, filename='mental_health_model.pkl'):
        with open(filename, 'wb') as f:
            pickle.dump({
                'vectorizer': self.vectorizer,
                'model': self.model,
                'classes': self.classes_,
                'accuracy': self.accuracy
            }, f)
        print(f"Model saved as {filename}")
    
    def load_model(self, filename='mental_health_model.pkl'):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            self.vectorizer = data['vectorizer']
            self.model = data['model']
            self.classes_ = data['classes']
            self.accuracy = data['accuracy']
        print(f"Model loaded from {filename}")

    def validate_model(self, custom_queries=None):

        #Validate model with known test cases
        test_cases = {
            "crisis": [
                "i want to kill myself",
                "ending my life seems like the only option",
                "can't take this pain anymore"
            ],
            "therapy": [
                "need help with anxiety",
                "looking for counseling services",
                "relationship problems affecting my mental health"
            ]
        }
        
        print("Model Validation Results:")
        for true_label, queries in test_cases.items():
            print(f"\nTesting {true_label} cases:")
            for query in queries:
                pred, conf, probs = self.predict(query)
                status = "✓" if pred == true_label else "✗"
                print(f"  {status} '{query}' -> {pred} (conf: {conf:.2f})")

def main():
    """Train, validate and save the model"""
    classifier = MentalHealthClassifier()
    
    # Train the model
    classifier.train()
    
    # Validate with test cases
    classifier.validate_model()
    
    # Interactive testing
    print("\nInteractive testing (type 'quit' to exit):")
    while True:
        user_input = input("\nEnter a query to classify: ").strip()
        if user_input.lower() in ['quit', 'exit', '']:
            break
        
        prediction, confidence, probabilities = classifier.predict(user_input)
        print(f"Prediction: {prediction}")
        print(f"Confidence: {confidence:.2f}")
        print(f"Probabilities: {probabilities}")
        
        # Add safety warning for crisis detection
        if prediction == "crisis" and confidence > 0.7:
            print("\n🚨 CRISIS DETECTED - Consider immediate help:")
            print("National Suicide Prevention Lifeline: 1-800-273-8255")
            print("Crisis Text Line: Text HOME to 741741")
    
    # Save the model
    classifier.save_model()
    
    return classifier

if __name__ == "__main__":
    classifier = main()