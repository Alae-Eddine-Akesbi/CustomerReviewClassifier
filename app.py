import streamlit as st
import joblib
import os
import torch
from transformers import AutoTokenizer, BertForSequenceClassification, pipeline, XLNetForSequenceClassification, XLNetTokenizer

# Add custom page configuration
st.set_page_config(
    page_title="Sentiment Analysis App 📊", 
    page_icon="🔍", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for improved styling
st.markdown("""
    <style>
    .big-font {
        font-size:20px !important;
        color: #2C3E50;
    }
    .stButton>button {
        background-color: #3498DB;
        color: white;
        border-radius: 10px;
    }
    .stTextArea>div>div>textarea {
        border: 2px solid #3498DB;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# Function to load the model and dependencies
def load_model(model_name):
    try:
        if "SVM" in model_name:
            # Load SVM model
            model = joblib.load('models/svm/svm_model.pkl')
            vectorizer = joblib.load('models/svm/tfidf_vectorizer.pkl')
            return model, vectorizer, None
        elif "Regression" in model_name:
            # Existing Logistic Regression model loading
            model = joblib.load('models/logistic_regression/logreg_model.pkl')
            vectorizer = joblib.load('models/logistic_regression/tfidf_vectorizer.pkl')
            label_encoder = joblib.load('models/logistic_regression/label_encoder.pkl')
            return model, vectorizer, label_encoder
        elif "BERT" in model_name:
            try:
                # Path to the BERT model
                path = "models/bert_lora"
                
                if not os.path.exists(path):
                    raise ValueError(f"Directory {path} does not exist")
                
                # Load model and tokenizer
                model = BertForSequenceClassification.from_pretrained(path, num_labels=3)
                tokenizer = AutoTokenizer.from_pretrained(path)
                
                model.eval()
                
                return model, tokenizer, None
            
            except Exception as e:
                st.error(f"Error loading BERT model: {str(e)}")
                st.write("Contents of models/bert_lora directory:")
                st.write(os.listdir("models/bert_lora"))
                return None, None, None
        elif "XLNET" in model_name:
            try:
                # Path to the XLNet model
                path = "models/xlnet"
                
                if not os.path.exists(path):
                    raise ValueError(f"Directory {path} does not exist")
                
                # Load model and tokenizer
                model = XLNetForSequenceClassification.from_pretrained(path, num_labels=3)
                tokenizer = XLNetTokenizer.from_pretrained(path)
                
                model.eval()
                
                return model, tokenizer, None
            
            except Exception as e:
                st.error(f"Error loading XLNet model: {str(e)}")
                st.write("Contents of models/xlnet directory:")
                st.write(os.listdir("models/xlnet"))
                return None, None, None
        else:
            st.error(f"Unsupported model: {model_name}")
            return None, None, None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

# Function to perform prediction
def predict_sentiment(text, model, vectorizer, label_encoder):
    if not text.strip():
        return "Error: The text field is empty."
    try:
        # Specific handling for BERT and XLNet models
        if isinstance(model, (BertForSequenceClassification, XLNetForSequenceClassification)):
            inputs = vectorizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            
            with torch.no_grad():
                outputs = model(**inputs)
                predictions = torch.softmax(outputs.logits, dim=1)
                predicted_class = torch.argmax(predictions, dim=1).item()
            
            sentiment_labels = ['Negative', 'Neutral', 'Positive']
            predicted_label = sentiment_labels[predicted_class]
            
            sentiment_map = {
                "Positive": ("🌟 Positive Comment! 🎉 The comment is very enthusiastic and encouraging 🚀", "success"),
                "Negative": ("😔 Negative Sentiment! 🚫 The comment expresses frustration or dissatisfaction 💥", "error"), 
                "Neutral": ("😐 Neutral Comment 🤷‍♀️ No strong emotion is expressed.", "warning")
            }
            
            message, color_type = sentiment_map[predicted_label]
            
            if color_type == "success":
                st.success(message)
            elif color_type == "error":
                st.error(message)
            elif color_type == "warning":
                st.warning(message)
            
            return message
        
        # Existing code for SVM and Logistic Regression
        text_tfidf = vectorizer.transform([text])
        predicted_rating = model.predict(text_tfidf)
        
        if label_encoder:
            predicted_rating_label = label_encoder.inverse_transform(predicted_rating)
        else:
            # For SVM, let's try to convert directly
            if predicted_rating[0] in [0, 1, 2]:
                predicted_rating_label = ['Negative', 'Neutral', 'Positive'][predicted_rating[0]]
            else:
                predicted_rating_label = [str(predicted_rating[0])]
        
        sentiment_map = {
            "positive": ("🌟 Positive Comment! 🎉 The comment is very enthusiastic and encouraging 🚀", "success"),
            "negative": ("😔 Negative Sentiment! 🚫 The comment expresses frustration or dissatisfaction 💥", "error"), 
            "neutral": ("😐 Neutral Comment 🤷‍♀️ No strong emotion is expressed.", "warning")
        }
        
        label = str(predicted_rating_label[0]).lower()
        
        if label in sentiment_map:
            message, color_type = sentiment_map[label]
            
            if color_type == "success":
                st.success(message)
            elif color_type == "error":
                st.error(message)
            elif color_type == "warning":
                st.warning(message)
            
            return message
        else:
            st.info(f"Sentiment detected: {label}")
            return f"Sentiment detected: {label}"
    except Exception as e:
        return f"Error during prediction: {e}"

# Streamlit Interface
st.title("🌟 Sentiment Analysis with Different Models")

# Sidebar
st.sidebar.header("🛠️ Settings")

# Model selection
model_options = ["Logistic Regression 📈", "SVM 🤖", "BERT 🧠", "XLNET 🔮"]
selected_model = st.sidebar.selectbox("Select a model:", model_options)

# Load model button
if st.sidebar.button("🔬 Load Model"):
    model, vectorizer, label_encoder = load_model(selected_model)
    if model and vectorizer and label_encoder:
        st.sidebar.success(f"Model {selected_model} loaded successfully! ✅")
        st.session_state.model = model
        st.session_state.vectorizer = vectorizer
        st.session_state.label_encoder = label_encoder
    elif model and vectorizer:
        st.sidebar.success(f"Model {selected_model} loaded successfully! ✅")
        st.session_state.model = model
        st.session_state.vectorizer = vectorizer
        st.session_state.label_encoder = None
    else:
        st.sidebar.error("❌ Failed to load model.")

# Main area
st.header("🔍 Sentiment Analysis 📊")
st.markdown('<p class="big-font">This application predicts whether a comment is <b>positive</b>, <b>negative</b>, or <b>neutral</b> based on the selected model.</p>', unsafe_allow_html=True)

# Initial message
st.info("👋 Welcome! Start by loading a model from the sidebar, then enter a comment to analyze its sentiment. 🕵️‍♀️")
# Input field for the comment
user_input = st.text_area("📝 Enter a comment:", "")

# Predict button
if st.button("🚀 Predict Sentiment"):
    if "model" in st.session_state and "vectorizer" in st.session_state:
        result = predict_sentiment(user_input, st.session_state.model, st.session_state.vectorizer, st.session_state.label_encoder)
    else:
        st.warning("⚠️ Please load a model from the sidebar first.")
