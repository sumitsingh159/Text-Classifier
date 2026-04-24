import pandas as pd
import torch
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

def train():
    # 1. Load data
    data_path = 'dataset/synthetic_text_data.csv'
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return

    df = pd.read_csv(data_path)
    
    # 2. Encode labels
    labels = sorted(df['label'].unique().tolist())
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for i, label in enumerate(labels)}
    df['label_idx'] = df['label'].map(label2id)

    # 3. Split data
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    # 4. Convert to Hugging Face Dataset (explicitly selecting columns to avoid conflicts)
    train_dataset = Dataset.from_dict({
        'text': train_df['text'].tolist(),
        'labels': train_df['label_idx'].tolist()
    })
    val_dataset = Dataset.from_dict({
        'text': val_df['text'].tolist(),
        'labels': val_df['label_idx'].tolist()
    })

    # 5. Tokenize
    model_name = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name)

    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=128)

    train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=['text'])
    val_dataset = val_dataset.map(tokenize_function, batched=True, remove_columns=['text'])

    # 6. Model initialization
    model = BertForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id
    )

    # 7. Metrics calculation function
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
        acc = accuracy_score(labels, preds)
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }

    # 8. Training Arguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=10,
        weight_decay=0.01,
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="none"
    )

    # 9. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    # 10. Train
    print("Starting training...")
    trainer.train()

    # 11. Save model and tokenizer
    save_path = "./fine_tuned_bert"
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    # 12. Evaluate and save metrics
    print("Evaluating model...")
    eval_results = trainer.evaluate()
    
    # Save label mappings as well
    with open(os.path.join(save_path, "label_mapping.json"), "w") as f:
        json.dump({"id2label": id2label, "label2id": label2id}, f, indent=4)

    with open("metrics.json", "w") as f:
        json.dump(eval_results, f, indent=4)

    print(f"Training completed. Model saved to '{save_path}' and metrics to 'metrics.json'.")

if __name__ == "__main__":
    train()
