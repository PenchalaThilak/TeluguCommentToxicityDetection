# 🗣️ Telugu Text Toxicity Classifier 🇮🇳

This project is a **web app** that classifies Telugu comments as **toxic** or **non-toxic**.  
It supports both:
- 🔡 English-transliterated Telugu (e.g., `neeku` → `నీకు`)
- 📝 Direct Telugu script input

🧠 Powered by a fine-tuned **IndicBERT model**  
🌐 Deployed on **Hugging Face Spaces**  
📦 Uses **Gradio UI** for frontend  
🗃️ Logs all predictions in a **MySQL database** via **Flask API** hosted on **Render**

---

## 🚀 Live Demo

🔗 [Try it on Hugging Face Spaces](https://huggingface.co/spaces/Thilak118/teluguCommentToxicityDetection)  

---

## ✅ Features

- 🔤 Accepts English-style Telugu or direct Telugu input
- 🔁 Transliterates using Google Translate
- 🧠 Classifies text as **Toxic** or **Non-Toxic**
- 📈 Displays prediction with **confidence %**
- 💾 Logs every comment, prediction, and score to a MySQL database
- 👁️ Admin dashboard for viewing, adding, and deleting logs

---

## 🧾 Usage Instructions

### ✏️ Input Options

#### 🔹 Option 1: English Transliterated
1. Type: `neeku antha scene ledu`
2. Click **"Preview Transliteration"**
3. Review Telugu script and edit if needed

#### 🔹 Option 2: Direct Telugu
- Enter Telugu directly in the preview textbox

### 🔍 Predict Toxicity
Click **"Predict Toxicity"**  
You’ll get:
- Transliterated/cleaned Telugu text
- Prediction (`Toxic` or `Non-Toxic`)
- Confidence score (in %)

---

## 🧪 Examples

| Input (Transliterated)                     | Telugu Script                                   | Prediction | Confidence |
|-------------------------------------------|--------------------------------------------------|------------|------------|
| `neeku antha scene ledu le`               | `నీకు అంత సీన్ లేదు లే`                         | Toxic      | 95.23%     |
| `meeru naaku oka korika kshaminchali`     | `మీరు నాకు ఒక కోరిక క్షమించాలి`                | Non-Toxic  | 87.65%     |

---

## 🧠 Model Info

- **Name**: `indic_bert_toxicity_classifier_corrected`
- **Base**: `ai4bharat/indic-bert` (ALBERT)
- **Task**: Binary classification — Toxic vs Non-Toxic
- **Deployed**: Inside the Hugging Face Space (no external downloads)

---

## 🗃️ Database Integration

This app uses a **MySQL database** to **log predictions** in real-time.

### 🔐 Hosted on:
- **Render** (Flask API)
- **freesqldatabase.com** (MySQL DB hosting)

### 📋 Data Stored:
| Field           | Type    | Description                               |
|----------------|---------|-------------------------------------------|
| `id`           | INT     | Auto-incremented ID                       |
| `comment`      | TEXT    | Original user comment (English form)      |
| `transliterated_text` | TEXT | Telugu script version of the comment |
| `prediction`   | VARCHAR | Toxic or Non-Toxic                        |
| `confidence`   | FLOAT   | Confidence % of prediction                |

### 🌐 API Endpoints (Flask)
| Endpoint               | Method | Description                  |
|------------------------|--------|------------------------------|
| `/logs`                | GET    | View all logs                |
| `/logs`                | POST   | Add a new prediction log     |
| `/logs/<int:id>`       | DELETE | Delete a specific log        |

---

## 🧑‍💻 Local Installation

```bash
# 1. Clone the repository
git clone https://huggingface.co/spaces/<your-username>/telugu-toxicity-classifier
cd telugu-toxicity-classifier

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run Gradio app
python app.py

# 4. Run Flask API (optional)
cd backend
python main.py
