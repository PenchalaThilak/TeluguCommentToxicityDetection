import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset

# 1. Load the dataset
df = pd.read_csv("telugu_comments_toxicity_dataset.csv")  # replace with your actual dataset

# 2. Clean the dataset
df = df.dropna(subset=["Text", "Label"])
df = df[df["Label"].isin(["Toxic", "Non-Toxic"])]

# 3. Encode the labels
label_encoder = LabelEncoder()
df["encoded_label"] = label_encoder.fit_transform(df["Label"])  # Non-Toxic=0, Toxic=1

# 4. Train-validation split
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# 5. Convert to Hugging Face dataset
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

# 6. Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-bert")

# 7. Tokenization function
def tokenize_function(examples):
    return tokenizer(examples["Text"], padding="max_length", truncation=True, max_length=128)

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)

# 8. Set format for PyTorch
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "encoded_label"])
val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "encoded_label"])

# 9. Load model
model = AutoModelForSequenceClassification.from_pretrained("ai4bharat/indic-bert", num_labels=2)

# 10. Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=4,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    logging_dir="./logs",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)

# 11. Define metrics (optional)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="binary")
    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }

# 12. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# 13. Train the model
trainer.train()

# 14. Evaluate the model
trainer.evaluate()

# 15. Save the model
model.save_pretrained("indic_bert_toxicity_classifier_corrected")
tokenizer.save_pretrained("indic_bert_toxicity_classifier_corrected")


#After running the above script, your model folder indic_bert_toxicity_classifier_corrected/ will contain:
#- config.json
#- pytorch_model.bin
#- tokenizer_config.json
#- vocab.txt
#- special_tokens_map.json
#- tokenizer.json






