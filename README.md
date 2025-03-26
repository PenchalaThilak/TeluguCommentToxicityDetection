# Telugu Text Toxicity Classifier

This project is a web application that classifies Telugu text as toxic or non-toxic. It uses a fine-tuned ALBERT model (`indic_bert_toxicity_classifier_corrected`) to predict toxicity in Telugu text. The app accepts input in two forms:
- **English transliterated Telugu text** (e.g., `neeku` for నీకు), which is then transliterated to Telugu script using Google Translate.
- **Direct Telugu script** (e.g., నీకు), which can be entered directly in the preview box.

The app is deployed on Hugging Face Spaces and uses Gradio for the user interface.

## Live Demo
You can try the app here: [Telugu Text Toxicity Classifier on Hugging Face Spaces](https://huggingface.co/spaces/Thilak118/teluguCommentToxicityDetection/blob/main/app.py) *(Replace with your actual Space URL)*

## Features
- Transliterates English-transliterated Telugu text to Telugu script using Google Translate.
- Allows direct input of Telugu script if preferred.
- Predicts whether the text is "Toxic" or "Non-Toxic" with a confidence score.
- Built with a fine-tuned ALBERT model for sequence classification.

## Usage Instructions
1. **Enter Text**:
   - **Option 1: English Transliterated Input**:
     - In the "Enter Telugu Text (in English Transliterated Form)" textbox, enter Telugu text in English transliteration (e.g., `neeku antha scene ledu le anni jaripoy unnayi`).
     - Click "Preview Transliteration" to see the Telugu script in the "Transliterated Telugu Text (Preview)" textbox (e.g., `నీకు అంత సీన్ లేదు లే అన్నీ జరిపోయ్ ఉన్నాయి`).
     - Edit the Telugu text if the transliteration is incorrect.
   - **Option 2: Direct Telugu Input**:
     - Directly enter Telugu script in the "Transliterated Telugu Text (Preview)" textbox (e.g., `నీకు అంత సీన్ లేదు లే అన్నీ జరిపోయ్ ఉన్నాయి`).
2. **Predict Toxicity**:
   - Click "Predict Toxicity" to get the prediction.
   - The output will show:
     - The transliterated Telugu text (or the directly entered Telugu text).
     - The prediction ("Toxic" or "Non-Toxic").
     - The confidence score (as a percentage).

## Examples
- **Toxic Example**:
  - Input (English Transliterated): `neeku antha scene ledu le anni jaripoy unnayi`
  - Transliterated Telugu: `నీకు అంత సీన్ లేదు లే అన్నీ జరిపోయ్ ఉన్నాయి`
  - Prediction: `Toxic` (with confidence, e.g., 95.23%)
- **Non-Toxic Example**:
  - Input (English Transliterated): `meeru naaku oka korika kshaminchali bullet bandi song cheyandi song cheyandi meeru baaga`
  - Transliterated Telugu: `మీరు నాకు ఒక కోరిక క్షమించాలి బుల్లెట్ బండి సాంగ్ చేయండి సాంగ్ చేయండి మీరు బాగా`
  - Prediction: `Non-Toxic` (with confidence, e.g., 87.65%)
- **Direct Telugu Input**:
  - Input (Telugu Script): `నీకు అంత సీన్ లేదు లే అన్నీ జరిపోయ్ ఉన్నాయి`
  - Prediction: `Toxic` (with confidence, e.g., 95.23%)

## Notes
- Google Translate may not always transliterate accurately. If the output is incorrect, try adjusting your input (e.g., use `scene` for సీన్).
- The app runs on Hugging Face Spaces’ free tier (2 GB RAM, CPU), so it may sleep after 48 hours of inactivity. Visiting the URL will wake it up.

## Model Details
- **Model**: `indic_bert_toxicity_classifier_corrected` (a fine-tuned ALBERT model for sequence classification).
- **Size**: ~124 MB (zipped).
- **Download Link**: The model is hosted on Google Drive: [indic_bert_toxicity_classifier_corrected.zip](https://drive.google.com/uc?id=18of1l7TSasaxxmqxRZ3-t1bdmV2ojO5Y)
  - Ensure the link is set to "Anyone with the link can view" to allow the app to download it during deployment.

## Installation (For Local Development)
If you want to run this app locally, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://huggingface.co/spaces/<your-username>/telugu-toxicity-classifier
   cd telugu-toxicity-classifier
