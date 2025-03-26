import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re
from deep_translator import GoogleTranslator
import os
import gdown

# Function to download the model from Google Drive if not present
def download_model():
    model_dir = "./indic_bert_toxicity_classifier_corrected"
    if not os.path.exists(model_dir):
        # Replace 'YOUR_FOLDER_ID' with the actual folder ID from Google Drive
        url = "https://drive.google.com/drive/folders/1P8OpPg9PNHk56y7mx42WRDflKhLy0d0C?usp=sharing"
        gdown.download_folder(url, output=model_dir, quiet=False)
    return model_dir

# Load the trained model and tokenizer
model_dir = download_model()
model = AutoModelForSequenceClassification.from_pretrained(model_dir)
tokenizer = AutoTokenizer.from_pretrained(model_dir)

# Initialize the translator
translator = GoogleTranslator(source='en', target='te')

# Function to clean the text
def clean_text(text):
    text = re.sub(r'[^\u0C00-\u0C7F\s.,!?]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Function to transliterate English to Telugu
def transliterate_to_telugu(text):
    telugu_text = translator.translate(text)
    return telugu_text

# Function to predict toxicity
def predict_toxicity(english_text):
    telugu_text = transliterate_to_telugu(english_text)
    cleaned_text = clean_text(telugu_text)
    inputs = tokenizer(cleaned_text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1).item()
    prob = torch.softmax(outputs.logits, dim=1)[0]
    confidence = max(prob).item() * 100
    label = "Toxic" if prediction == 0 else "Non-Toxic"
    return f"Transliterated Telugu Text: {cleaned_text}\nPrediction: {label}\nConfidence: {confidence:.2f}%"

# Create the Gradio interface
interface = gr.Interface(
    fn=predict_toxicity,
    inputs=gr.Textbox(
        label="Enter Telugu Text (in English Transliterated Form)",
        placeholder="e.g., 'neeku antha scene ledu le anni jaripoy unnayi'",
        lines=2,
    ),
    outputs=gr.Textbox(
        label="Prediction",
        lines=5,
        max_lines=10,
    ),
    title="Telugu Text Toxicity Classifier (Transformer - Corrected with Transliteration)",
    description="Enter Telugu text in English transliteration (e.g., 'neeku' for నీకు). The app will convert it to Telugu script and predict if it's toxic or non-toxic.",
    examples=[
        ["neeku antha scene ledu le anni jaripoy unnayi"],
        ["meeru naaku oka korika kshaminchali bullet bandi song cheyandi song cheyandi meeru baaga"]
    ]
)

# Launch the interface with Render-compatible settings
if __name__ == "__main__":
    interface.launch(server_name="0.0.0.0", server_port=7860)
