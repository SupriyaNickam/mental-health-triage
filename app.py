import streamlit as st
import pandas as pd
import random

if 'selected_languages' not in st.session_state:
    st.session_state.selected_languages = []

if 'selected_locations' not in st.session_state:
    st.session_state.selected_locations = []

if 'filters_initialized' not in st.session_state:
    st.session_state.filters_initialized = False

# Load css
def load_css():
    try:
        with open('style.css') as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.error("CSS file not found. Using default styling.")

load_css()

# Set up the page
st.set_page_config(page_title="Mental Health Triage System", page_icon="💬", layout="wide")
st.title("Mental Health Triage System (India)")
st.write("Describe what you're going through, and our AI will match you with the most relevant mental health resources.")

# Initialize session state for model
if 'model_initialized' not in st.session_state:
    st.session_state.model_initialized = False
    st.session_state.classifier = None

# Load ML model 
if not st.session_state.model_initialized:
    with st.spinner("🤖 Loading AI model (one-time setup)..."):
        try:
            # Try multiple possible model locations
            model_paths = [
                'mental_health_model.pkl',
                './mental_health_model.pkl',
                'train_model.py'  # Fallback to training if no model file
            ]
            
            model_loaded = False
            for model_path in model_paths:
                try:
                    if model_path.endswith('.pkl'):
                        classifier = MentalHealthClassifier()
                        classifier.load_model(model_path)
                        st.session_state.classifier = classifier
                        model_loaded = True
                       # st.success("AI model loaded successfully!")
                        break
                    elif model_path.endswith('.py'):
                        # Train model if no pre-trained model exists
                        from train_model import MentalHealthClassifier
                        classifier = MentalHealthClassifier()
                        classifier.train()
                        classifier.save_model()
                        st.session_state.classifier = classifier
                        model_loaded = True
                        #st.success("AI model trained and loaded!")
                        break
                except Exception as e:
                    continue
            
            if not model_loaded:
                st.session_state.classifier = None
                st.warning("⚠️ Using rule-based system only. ML model unavailable.")
                
        except Exception as e:
            st.session_state.classifier = None
            st.warning(f"⚠️ Using rule-based system only: {str(e)}")
        
        st.session_state.model_initialized = True

#load csv file
@st.cache_data
def load_resources():
    try:
        df = pd.read_csv('indian_mental_health_resources.csv.csv')
        return df
    except FileNotFoundError:
        st.error("Resources database not found.")
        return pd.DataFrame()

# Load data to dataframe
resources_df = load_resources()

# Categorization with mental health keywords
def categorize_query(text):
     # Use ML model if available
    if st.session_state.classifier is not None:
        try:
            prediction, confidence, _ = st.session_state.classifier.predict(text)
            st.sidebar.info(f"🤖 AI Confidence: {confidence:.1%}")
            
            # Only use ML prediction if confidence is high enough
            if confidence > 0.6:
                return prediction
            # Fall back to rule-based if low confidence
        except Exception as e:
            st.sidebar.warning("ML model temporarily unavailable")

    text_lower = text.lower()
    
    crisis_keywords = ['suicide', 'suicidal', 'end my life', 'kill myself', 'want to die', 'hopeless', 'no reason to live']
    anxiety_keywords = ['panic attack', 'severe anxiety', 'anxiety attack', 'cant breathe', 'heart racing']
    general_anxiety_keywords = ['anxiety', 'worry', 'stress', 'nervous', 'overwhelmed']    
    depression_keywords = ['depression', 'depressed', 'sad', 'hopeless', 'empty', 'no energy', 'cant get out of bed', 'loss of interest']
    therapy_keywords = ['therapy', 'counsel', 'therapist', 'psychologist', 'counseling', 'session', 'professional help']
    relationship_keywords = ['relationship', 'marriage', 'family', 'partner', 'breakup', 'divorce']
    work_keywords = ['work', 'job', 'career', 'office', 'professional', 'workplace', 'corporate']
    trauma_keywords = ['trauma', 'abuse', 'violent', 'assault', 'ptsd']
    
    if any(word in text_lower for word in crisis_keywords):
        return 'crisis'
    elif any(word in text_lower for word in trauma_keywords):
        return 'therapy'  # Trauma specialists are in therapy category
    elif any(word in text_lower for word in anxiety_keywords):
        return 'crisis'
    elif any(word in text_lower for word in depression_keywords):
        return 'therapy'
    elif any(word in text_lower for word in relationship_keywords):
        return 'therapy'
    elif any(word in text_lower for word in work_keywords):
        return 'therapy'
    elif any(word in text_lower for word in therapy_keywords):
        return 'therapy'
    elif any(word in text_lower for word in general_anxiety_keywords):  # General anxiety
        return 'therapy'
    else:
        return 'therapy'

#Calm tip while loading analysis results
def show_calm_screen():
    with st.spinner("🤖 AI is carefully analyzing your needs..."):
        tips = [
            "🌱 **Breathe**: Inhale for 4, hold for 4, exhale for 6",
            "🕊️ **Ground yourself**: Name 3 things you can see, 2 you can touch, 1 you can hear",
            "💧 **Hydrate**: Take a sip of water while you wait",
            "📝 **Remember**: It's brave to seek help",
            "🌿 **Pause**: Notice your feet on the ground and the air on your skin"
        ]
        st.success(f"{random.choice(tips)}")

def calculate_relevance_score(resource, user_query):
    #Calculate relevance score with proper specialty matching
    score = 0
    query_lower = user_query.lower()
    specialty = str(resource['specialty']).lower()
    
    # Priority to crisis resources for urgent needs
    if resource['category'] == 'crisis':
        score += 100
    
    # Priority to free resources
    if resource['cost'] == 'Free':
        score += 50
    elif resource['cost'] == 'Flexible':
        score += 25
    
    # Priority to 24/7 availability
    if '24/7' in str(resource['availability']):
        score += 40
    
    # Priority to pan-India coverage
    if 'Pan-India' in str(resource['location']):
        score += 30
    elif 'Online' in str(resource['location']):
        score += 25
    
    # Proper specialty matching
    specialty_words = specialty.split()
    matched_specialty_words = [word for word in specialty_words if word in query_lower]
    
    if matched_specialty_words:
        # Give points based on how many specialty words match
        score += len(matched_specialty_words) * 15
        # Bonus for exact specialty match
        if len(matched_specialty_words) >= 2:
            score += 20
    
    #Check if query contains common mental health terms that match specialty
    mental_health_terms = {
        'anxiety': ['anxiety', 'panic', 'worry', 'nervous'],
        'depression': ['depression', 'sad', 'hopeless', 'empty'],
        'trauma': ['trauma', 'abuse', 'ptsd', 'violent'],
        'relationship': ['relationship', 'marriage', 'couple', 'partner'],
        'lgbtq': ['lgbtq', 'gay', 'lesbian', 'transgender', 'gender'],
        'academic': ['academic', 'exam', 'study', 'college', 'career']
    }
    
    for term, keywords in mental_health_terms.items():
        if term in specialty and any(keyword in query_lower for keyword in keywords):
            score += 25
    
    # Languages (bonus for English/Hindi which are most common)
    languages = str(resource['languages']).lower()
    if 'english' in languages or 'hindi' in languages:
        score += 15
    
    return score

# User input
user_input = st.text_area(
    "**How are you feeling or what kind of support are you looking for?**", 
    placeholder="e.g., I've been feeling really anxious about work and having trouble sleeping...",
    height=100
)

# Initialize session state
if 'filters_initialized' not in st.session_state:
    st.session_state.filters_initialized = False
    st.session_state.selected_languages = []
    st.session_state.selected_locations = []

if st.button("Analyze & Find Resources", type="primary"):
    if user_input:
        # Show calming message
        show_calm_screen()
        
        # Determine category
        predicted_category = categorize_query(user_input)
        
        # Store in session state
        st.session_state.predicted_category = predicted_category
        st.session_state.user_input = user_input
        st.session_state.filters_initialized = True

        # CLEAR ALL FILTERS
        st.session_state.selected_languages = []
        st.session_state.selected_locations = []
        st.session_state.online_only = False

# Display results and filters if there is user input
if st.session_state.get('filters_initialized', False):
    predicted_category = st.session_state.predicted_category
    user_input = st.session_state.user_input
    
    # Display analysis results with filters on the right
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Analysis Results")
        st.write(f"**You described:** '{user_input}'")
        st.success(f"**Detected Need:** `{predicted_category.title()}`")

    with col2:
        st.subheader("Refine Results")
        
        # Online/Remote switch
        online_only = st.checkbox(
            "Show only online/remote services", 
            value=False,
            help="Filter to show only services available online"
        )
        
        # Get available languages and locations from ALL data
        all_languages = set()
        all_locations = set()
        
        for _, row in resources_df.iterrows():
            # Languages
            if pd.notna(row['languages']):
                languages = [l.strip() for l in str(row['languages']).replace(',', ';').split(';')]
                all_languages.update(languages)
            # Locations
            if pd.notna(row['location']):
                locations = [l.strip() for l in str(row['location']).replace(',', ';').split(';')]
                all_locations.update(locations)

        # Language filter - disabled for crisis
        if all_languages:
            if predicted_category == 'crisis':
                st.multiselect(
                    "Filter by language:",
                    options=sorted(all_languages),
                    default=[],
                    help="All crisis helplines handle multiple languages",
                    disabled=True
                )
                st.caption("Crisis helplines handle multiple languages")
            else:
                st.session_state.selected_languages = st.multiselect(
                    "Filter by language:",
                    options=sorted(all_languages),
                    default=st.session_state.selected_languages,
                    help="Select one or more languages"
                )
        
        # Location filter - disabled for crisis and online only
        if all_locations:
            if predicted_category == 'crisis' or online_only:
                if predicted_category == 'crisis':
                    st.session_state.selected_locations = []  # Clear location selections
                    st.multiselect(
                        "Filter by location:",
                        options=sorted([loc for loc in all_locations if loc]),
                        default=[],
                        help="Crisis helplines are available nationwide",
                        disabled=True
                    )
                    st.caption("🇮🇳 Crisis helplines available across India")
                else:  # online_only
                    st.session_state.selected_locations = []  # Clear location selections
                    st.multiselect(
                        "Filter by location:",
                        options=sorted([loc for loc in all_locations if loc]),
                        default=[],
                        help="Location filter disabled when 'online only' is selected",
                        disabled=True
                    )
                    st.caption("Location filter disabled for online services")
            else:
                # Normal location filter
                st.session_state.selected_locations = st.multiselect(
                    "Filter by location:",
                    options=sorted([loc for loc in all_locations if loc]),
                    default=st.session_state.selected_locations,
                    help="Select one or more locations"
                )
        
        # Store online filter in session state
        st.session_state.online_only = online_only

    # Filter and display TOP resources
    st.subheader("Top Recommended Resources:")
    
    filtered_df = resources_df[resources_df['category'] == predicted_category]
    
    if not filtered_df.empty:
        # Apply filters
        if st.session_state.selected_languages:
            language_mask = filtered_df['languages'].apply(
                lambda x: any(lang in str(x) for lang in st.session_state.selected_languages) 
                if pd.notna(x) else False
            )
            filtered_df = filtered_df[language_mask]
        
        if st.session_state.selected_locations:
            location_mask = filtered_df['location'].apply(
                lambda x: any(loc in str(x) for loc in st.session_state.selected_locations)
                if pd.notna(x) else False
            )
            filtered_df = filtered_df[location_mask]
        
        # Apply online filter
        if st.session_state.get('online_only', False):
            online_mask = filtered_df['location'].apply(
                lambda x: any(online_keyword in str(x).lower() for online_keyword in ['online', 'remote', 'virtual', 'telehealth', 'pan-india'])
                if pd.notna(x) else False
            )
            filtered_df = filtered_df[online_mask]
        
        # Calculate relevance scores
        filtered_df = filtered_df.copy()
        filtered_df['relevance_score'] = filtered_df.apply(
            lambda row: calculate_relevance_score(row, user_input), 
            axis=1
        )
        
        # Get top 5 most relevant resources
        top_resources = filtered_df.nlargest(5, 'relevance_score')
        
        if not top_resources.empty:
            st.write(f"*Showing {len(top_resources)} most relevant resources out of {len(filtered_df)} filtered*")
            
            for idx, (_, resource) in enumerate(top_resources.iterrows(), 1):
                with st.expander(f"#{idx} {resource['name']}", expanded=idx==1):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.write(f"**Category:** {resource['category'].title()}")
                        st.write(f"**Specialty:** {resource['specialty']}")
                        st.write(f"**Location:** {resource['location']}")
                        st.write(f"**Languages:** {resource['languages']}")
                        
            
                    with col2:
                        if resource['cost'] == "Free":
                            cost_color = "🟢" 
                        else:
                            cost_color = ""
                        st.write(f"**Cost:** {cost_color} {resource['cost']}")
                        
                        if "24/7" in str(resource['availability']):
                            st.write("**Availability:** 🟢 Available Now")
                        else:
                            st.write(f"**Availability:** {resource['availability']}")
                        
                        contact_text = resource['contact']
                        if 'http' in str(contact_text):
                            st.write(f"**Website:** [{contact_text}]({contact_text})")
                        else:
                            st.write(f"**Contact:** `{contact_text}`")
                    
                    
                    if pd.notna(resource['notes']) and str(resource['notes']).strip():
                        st.info(f"💡 **Note:** {resource['notes']}")
        else:
            st.warning("No resources match your current filters. Try adjusting your filter criteria.")
            
    else:
        st.warning("No specific resources found for your query. Here are general crisis resources:")
        crisis_fallback = resources_df[resources_df['category'] == 'crisis'].head(3)
        for _, resource in crisis_fallback.iterrows():
            st.write(f"**{resource['name']}**: `{resource['contact']}` - {resource['notes']}")

# Quick self-care tips in sidebar
with st.sidebar:
    st.subheader("🌿 Quick Self-Care")
    if st.button("Show me a calming exercise"):
        exercises = [
            "**Box Breathing**: Breathe in 4s, hold 4s, out 4s, hold 4s. Repeat 4x",
            "**5-4-3-2-1 Grounding**: Notice 5 things you can see, 4 you can feel, 3 you can hear, 2 you can smell, 1 you can taste",
            "**Progressive Relaxation**: Tense and release each muscle group from toes to head",
            "**Mindful Pause**: Close eyes and focus only on your breath for 60 seconds",
            "**Self-Compassion Break**: Place hand on heart and say 'I'm doing the best I can right now'"
        ]
        st.info(random.choice(exercises))
    
    
    st.divider()
        
    try: 
        st.image("sidebar_image.png", width=220, caption="  ")
    except Exception as e:
        st.error(f"Cannot load image: {e}")

# Footer
st.subheader("🚨 Immediate Crisis Support (24/7)")
crisis_resources = resources_df[
    (resources_df['category'] == 'crisis') & 
    (resources_df['availability'].str.contains('24/7', na=False))
].head(2)

for _, crisis in crisis_resources.iterrows():
    st.write(f"**{crisis['name']}**: `{crisis['contact']}` - {crisis['notes']}")

st.divider()
st.caption("""
⚠️ **Disclaimer:** This tool is for resource matching only and **not a substitute for professional medical advice**. 
Always contact emergency services or 24/7 crisis helplines in case of immediate danger.
""")
st.caption("""
**Note: No personal information/data will be stored.**
""")