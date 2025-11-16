import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK data (run first time only)
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
        """Clean and preprocess text data"""
        if isinstance(text, str):
            # Convert to lowercase
            text = text.lower()
            # Remove URLs
            text = re.sub(r'http\S+', '', text)
            # Remove special characters and digits
            text = re.sub(r'[^a-zA-Z\s]', '', text)
            # Remove extra whitespace
            text = ' '.join(text.split())
            return text
        return ""
    
    def create_training_data(self):
    
        training_data = []
        
        # Enhanced crisis queries
        crisis_queries = [
            "i want to end my life", "suicidal thoughts", "thinking about suicide",
            "i want to die", "life is not worth living", "no reason to live",
            "ending it all", "i feel hopeless", "extreme emotional pain",
            "overwhelming despair", "cant take it anymore", "self harm",
            "harm myself", "not safe with myself"
        ]
        
        # Enhanced gender/sexuality queries
        gender_queries = [
            "i think i might be asexual", "struggling with my gender identity",
            "coming out as gay", "family rejection for being lgbtq",
            "unable to relate to peers about sexuality", "gender dysphoria",
            "sexual orientation confusion", "lgbtq discrimination",
            "transgender transition", "pronouns and identity"
        ]
        
        # Enhanced academic stress queries
        academic_queries = [
            "jee preparation stress", "neet exam pressure", "academic burnout",
            "college entrance exam anxiety", "study pressure overwhelming",
            "cant handle academic stress", "exam performance anxiety",
            "competitive exam pressure", "parental expectations stress"
        ]
        
        # Enhanced relationship queries
        relationship_queries = [
            "my husband yells at me", "marital conflict", "relationship problems",
            "partner anger issues", "emotional abuse in marriage",
            "communication issues with spouse", "marital counseling needed",
            "domestic conflict resolution"
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
        
        return pd.DataFrame(training_data)
    
    def train(self):
        """Train the classification model"""
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
    
    def predict(self, text):
        """Predict category for new text"""
        cleaned_text = self.clean_text(text)
        text_vec = self.vectorizer.transform([cleaned_text])
        prediction = self.model.predict(text_vec)[0]
        probabilities = self.model.predict_proba(text_vec)[0]
        
        # Get confidence score
        confidence = max(probabilities)
        
        return prediction, confidence, dict(zip(self.classes_, probabilities))
    
    def save_model(self, filename='mental_health_model.pkl'):
        """Save the trained model"""
        with open(filename, 'wb') as f:
            pickle.dump({
                'vectorizer': self.vectorizer,
                'model': self.model,
                'classes': self.classes_,
                'accuracy': self.accuracy
            }, f)
        print(f"Model saved as {filename}")
    
    def load_model(self, filename='mental_health_model.pkl'):
        """Load a trained model"""
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            self.vectorizer = data['vectorizer']
            self.model = data['model']
            self.classes_ = data['classes']
            self.accuracy = data['accuracy']
        print(f"Model loaded from {filename}")

def main():
    """Train and save the model"""
    classifier = MentalHealthClassifier()
    
    # Train the model
    accuracy = classifier.train()
    
    # Test with some examples
    test_queries = [
        "I've been feeling really sad and hopeless lately",
        "I need help with anxiety and panic attacks",
        "Looking for affordable therapy options",
        "I'm having suicidal thoughts and need immediate help",
        "Where can I find mental health resources?"
    ]
    
    print("\nTesting the model:")
    for query in test_queries:
        prediction, confidence, probs = classifier.predict(query)
        print(f"Query: '{query}'")
        print(f"Prediction: {prediction} (confidence: {confidence:.2f})")
        print(f"Probabilities: {probs}")
        print("-" * 50)
    
    # Save the model
    classifier.save_model()
    
    return classifier

if __name__ == "__main__":
    classifier = main()