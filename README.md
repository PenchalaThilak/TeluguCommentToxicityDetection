# 🗣️ Telugu Text Toxicity Classifier 🇮🇳

A powerful web application that classifies **Telugu comments** as 🔴 **Toxic** or 🟢 **Non-Toxic**.

Built with:
- 🤖 Fine-tuned ALBERT (IndicBERT) model
- 🎛️ Gradio frontend UI
- 📦 Hugging Face Spaces for hosting
- 🗃️ MySQL database with Flask API on Render

---

## 🚀 Live Demo

🎯 [Launch App on Hugging Face Spaces](https://huggingface.co/spaces/<your-username>/telugu-toxicity-classifier)  
🧠 [Model on Hugging Face Hub](https://huggingface.co/Thilak118/indic-bert-toxicity-classifier)  
🔧 [Backend API (Render)](https://<your-render-backend-url>)

---

## 🔥 Features

✅ Accepts **Telugu text** via:
- English-transliterated input (e.g., `neeku` → `నీకు`)
- Direct Telugu script (e.g., `నీకు`)

🔁 Transliterates automatically using **Google Translate**  
🧠 Predicts if the text is **Toxic / Non-Toxic**  
📈 Displays **confidence score** (%)  
💾 **Logs predictions** to a MySQL database (hosted on Render)  
🧾 Admin can **view / add / delete** logs

---

## ✍️ Usage Instructions

### 1️⃣ Input Options

**Option 1: English Transliteration**
- Enter: `neeku antha scene ledu`
- Click **Preview Transliteration** → `నీకు అంత సీన్ లేదు`
- Review and edit if needed

**Option 2: Direct Telugu**
- Type Telugu directly into the preview textbox

---

### 2️⃣ Predict Toxicity
- Click **Predict Toxicity**
- You’ll see:
  - 🌀 Transliterated/cleaned Telugu
  - ⚠️ Prediction: `Toxic` or `Non-Toxic`
  - 📊 Confidence Score (%)

---

## 🧪 Sample Inputs

| Input                          | Telugu Script                          | Prediction   | Confidence |
|-------------------------------|----------------------------------------|--------------|------------|
| `neeku antha scene ledu le`   | `నీకు అంత సీన్ లేదు లే`               | 🔴 Toxic     | 95.23%     |
| `meeru naaku oka korika`      | `మీరు నాకు ఒక కోరిక`                  | 🟢 Non-Toxic | 87.65%     |

---

## 🧠 Model Details

- 🔍 **Model**: [`Thilak118/indic-bert-toxicity-classifier`](https://huggingface.co/Thilak118/indic-bert-toxicity-classifier)
- 🏗️ **Base**: ai4bharat/indic-bert (ALBERT)
- 🧠 **Task**: Binary Classification (Toxic vs Non-Toxic)
- 💾 **Deployed inside** Hugging Face Space (no external download needed)
- 📊 **AUC**: 0.93, **Accuracy**: 86%

---

## 🗃️ Database Logging System

### 📌 What is Stored

| Field               | Description                                      |
|--------------------|--------------------------------------------------|
| `id`               | Auto-generated ID                                |
| `comment`          | Original user comment (English-style Telugu)     |
| `transliterated_text` | Cleaned Telugu script version                |
| `prediction`       | `Toxic` or `Non-Toxic`                           |
| `confidence`       | Prediction confidence (e.g., `93.4%`)            |

### 🌐 Backend API Endpoints (Flask + Render)

| Route                 | Method | Description                  |
|----------------------|--------|------------------------------|
| `/logs`              | GET    | View all logs                |
| `/logs`              | POST   | Add a new log entry          |
| `/logs/<id>`         | DELETE | Delete a log by ID           |

📍 **Database**: Hosted on [freesqldatabase.com](https://www.freesqldatabase.com/)  
🛠️ **API**: Hosted on [Render](https://render.com)

---

## 💻 Run Locally (Optional)

```bash
# 1. Clone the Space
git clone https://huggingface.co/spaces/<your-username>/telugu-toxicity-classifier
cd telugu-toxicity-classifier

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch Gradio app
python app.py

# 4. Run backend (optional)
cd backend
python main.py
