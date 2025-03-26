import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re
from deep_translator import GoogleTranslator
# Load the trained model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained("/content/drive/MyDrive/indic_bert_toxicity_classifier_corrected")
tokenizer = AutoTokenizer.from_pretrained("/content/drive/MyDrive/indic_bert_toxicity_classifier_corrected")

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
    # Step 1: Transliterate the input from English to Telugu
    telugu_text = transliterate_to_telugu(english_text)

    # Step 2: Clean the Telugu text
    cleaned_text = clean_text(telugu_text)

    # Step 3: Tokenize and predict using the transformer model
    inputs = tokenizer(cleaned_text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1).item()
    prob = torch.softmax(outputs.logits, dim=1)[0]
    confidence = max(prob).item() * 100

    # Step 4: Interpret the prediction (0 = Toxic, 1 = Non-Toxic)
    label = "Toxic" if prediction == 0 else "Non-Toxic"

    return f"Transliterated Telugu Text: {cleaned_text}\nPrediction: {label}\nConfidence: {confidence:.2f}%"

# Create the Gradio interface
interface = gr.Interface(
    fn=predict_toxicity,
    inputs=gr.Textbox(
        label="Enter Telugu Text (in English Transliterated Form)",
        placeholder="e.g., 'neeku antha scene ledu le anni jaripoy unnayi'",
        lines=2,  # Allow multiple lines for input
    ),
    outputs=gr.Textbox(
        label="Prediction",
        lines=5,  # Increase lines to show full output
        max_lines=10,  # Allow scrolling if text exceeds 5 lines
    ),
    title="Telugu Text Toxicity Classifier (Transformer - Corrected with Transliteration)",
    description="Enter Telugu text in English transliteration (e.g., 'neeku' for నీకు). The app will convert it to Telugu script and predict if it's toxic or non-toxic.",
    examples=[
        ["neeku antha scene ledu le anni jaripoy unnayi"],  # Toxic example
        ["meeru naaku oka korika kshaminchali bullet bandi song cheyandi song cheyandi meeru baaga"]  # Non-toxic example
    ]
)

# Launch the interface
interface.launch()
