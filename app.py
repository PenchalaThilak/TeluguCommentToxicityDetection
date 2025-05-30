# app.py
import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re
from deep_translator import GoogleTranslator
import os
import gdown
import subprocess

# Path to the model directory
model_path = "./indic_bert_toxicity_classifier_corrected"

# Download the model if it doesn't exist
if not os.path.exists(model_path):
    print("Downloading model from Google Drive...")
    model_url = "https://drive.google.com/uc?id=18of1l7TSasaxxmqxRZ3-t1bdmV2ojO5Y"  # Replace NEW_FILE_ID with the actual file ID
    zip_path = "indic_bert_toxicity_classifier_corrected.zip"
    try:
        gdown.download(model_url, zip_path, quiet=False)
    except Exception as e:
        raise Exception(f"Failed to download model: {str(e)}")

    print("Unzipping model...")
    try:
        result = subprocess.run(["unzip", zip_path, "-d", "."], capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception(f"Failed to unzip model: {result.stderr}")
    except Exception as e:
        raise Exception(f"Unzip error: {str(e)}")

# Verify the model directory exists
if not os.path.exists(model_path):
    raise Exception(f"Model directory {model_path} not found after unzipping. Check the zip file structure.")

# Determine device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the trained model and tokenizer
try:
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.to(device)
    model.eval()
except Exception as e:
    raise Exception(f"Error loading model or tokenizer: {str(e)}")

# Initialize the translator
translator = GoogleTranslator(source='en', target='te')

# Function to clean the text
def clean_text(text):
    text = re.sub(r'[^\u0C00-\u0C7F\s.,!?]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Function to check if the text contains Telugu characters
def is_telugu_text(text):
    # Telugu Unicode range: \u0C00-\u0C7F
    return bool(re.search(r'[\u0C00-\u0C7F]', text))

# Function to transliterate English to Telugu using deep-translator
def transliterate_to_telugu(text):
    try:
        telugu_text = translator.translate(text)
        return telugu_text
    except Exception as e:
        return f"Error in transliteration: {str(e)}. Please try again or use a different input format."

# Function to predict toxicity
def predict_toxicity(input_text):
    try:
        # Check if the input is already in Telugu script
        if is_telugu_text(input_text):
            # If the input is already Telugu, skip transliteration
            telugu_text = input_text
        else:
            # Otherwise, transliterate from English to Telugu
            telugu_text = transliterate_to_telugu(input_text)
            if "Error in transliteration" in telugu_text:
                return telugu_text

        # Clean the Telugu text
        cleaned_text = clean_text(telugu_text)

        # Tokenize and predict using the transformer model
        inputs = tokenizer(cleaned_text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()
        prob = torch.softmax(outputs.logits, dim=1)[0]
        confidence = max(prob).item() * 100
        label = "Toxic" if prediction == 0 else "Non-Toxic"
        return f"Transliterated Telugu Text: {cleaned_text}\nPrediction: {label}\nConfidence: {confidence:.2f}%"
    except Exception as e:
        return f"Error in prediction: {str(e)}. Please check your input and try again."

# Create the Gradio interface with a preview step
with gr.Blocks() as interface:
    gr.Markdown(
        """
        # Telugu Text Toxicity Classifier (Transformer - Corrected with Transliteration)
        Enter Telugu text in English transliteration (e.g., 'neeku' for నీకు). The app will convert it to Telugu script using Google Translate and predict if it's toxic or non-toxic.
        Note: Google Translate may not always transliterate accurately. If the output is incorrect, try adjusting your input (e.g., use 'scene' for సీన్).
        """
    )
    
    with gr.Row():
        english_input = gr.Textbox(
            label="Enter Telugu Text (in English Transliterated Form)",
            placeholder="e.g., 'neeku antha scene ledu le anni jaripoy unnayi'",
            lines=2
        )
        telugu_preview = gr.Textbox(label="Transliterated Telugu Text (Preview)", interactive=True, lines=2)
    
    preview_button = gr.Button("Preview Transliteration")
    predict_button = gr.Button("Predict Toxicity")
    output = gr.Textbox(label="Prediction", lines=5, max_lines=10)
    
    # Step 1: Preview the transliterated text
    def preview_transliteration(english_text):
        telugu_text = transliterate_to_telugu(english_text)
        if "Error in transliteration" in telugu_text:
            return telugu_text
        cleaned_text = clean_text(telugu_text)
        return cleaned_text
    
    # Step 2: Predict toxicity using the confirmed Telugu text
    def predict_from_telugu(telugu_text):
        return predict_toxicity(telugu_text)
    
    preview_button.click(
        fn=preview_transliteration,
        inputs=english_input,
        outputs=telugu_preview
    )
    
    predict_button.click(
        fn=predict_from_telugu,
        inputs=telugu_preview,
        outputs=output
    )

# Launch the interface (Hugging Face Spaces handles port binding automatically)
interface.launch()
