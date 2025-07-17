import gradio as gr
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import re
from deep_translator import GoogleTranslator
import requests

# Load model & tokenizer
model_name = "Thilak118/indic-bert-toxicity-classifier"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

translator = GoogleTranslator(source='en', target='te')

def clean_text(text):
    text = re.sub(r'[^\u0C00-\u0C7F\s.,!?]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def is_telugu_text(text):
    return bool(re.search(r'[\u0C00-\u0C7F]', text))

def transliterate_to_telugu(text):
    try:
        return translator.translate(text)
    except Exception as e:
        return f"Error in transliteration: {str(e)}"

def log_to_render(comment, transliterated, prediction, confidence):
    url = "https://telugu-toxicity-logger.onrender.com/log"
    payload = {
        "comment": comment,
        "transliterated": transliterated,
        "prediction": prediction,
        "confidence": confidence
    }
    try:
        requests.post(url, json=payload)
    except Exception as e:
        print("Logging failed:", e)

def predict_toxicity(user_input):
    try:
        original_input = user_input

        if is_telugu_text(original_input):
            telugu_text = original_input
        else:
            telugu_text = transliterate_to_telugu(original_input)
            if "Error in transliteration" in telugu_text:
                return telugu_text

        cleaned = clean_text(telugu_text)

        inputs = tokenizer(cleaned, return_tensors="pt", padding=True, truncation=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        prediction = torch.argmax(outputs.logits, dim=1).item()
        prob = torch.softmax(outputs.logits, dim=1)[0]
        confidence = max(prob).item() * 100
        label = "Toxic" if prediction == 0 else "Non-Toxic"

        # Log it to Render backend
        log_to_render(original_input, cleaned, label, confidence)

        return f"Transliterated Telugu Text: {cleaned}\nPrediction: {label}\nConfidence: {confidence:.2f}%"
    except Exception as e:
        return f"Error: {str(e)}"

# ✅ Gradio Interface
with gr.Blocks() as interface:
    gr.Markdown(
        """
        # Telugu Text Toxicity Classifier
        Enter Telugu text in English transliteration (e.g., 'neeku' for నీకు). The app will convert it to Telugu script and predict if it's toxic or non-toxic.
        Note: Transliteration may not always be accurate. Adjust input if needed (e.g., use 'scene' for సీన్).
        """
    )
    with gr.Row():
        english_input = gr.Textbox(
            label="Enter Telugu Text (in English Transliteration)",
            placeholder="e.g., chala baagundhi",
            lines=2
        )
        telugu_preview = gr.Textbox(
            label="Transliterated Telugu Text (Preview)",
            interactive=True,
            lines=2
        )

    preview_button = gr.Button("Preview Transliteration")
    predict_button = gr.Button("Predict Toxicity")
    output = gr.Textbox(label="Prediction Output", lines=5)

    preview_button.click(
        fn=transliterate_to_telugu,
        inputs=english_input,
        outputs=telugu_preview
    )

    predict_button.click(
        fn=predict_toxicity,
        inputs=english_input,
        outputs=output
    )

    # ✅ Admin Logs Button at Bottom
    with gr.Row():
        gr.Markdown(
            "<a href='https://telugu-toxicity-logger.onrender.com/logs' target='_blank'>"
            "<button style='padding: 10px; font-weight: bold;'>🔐 View Admin Logs</button>"
            "</a>"
        )

interface.launch()
