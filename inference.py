import torch
from transformers import BertTokenizer, BertForSequenceClassification
import json
import os

def predict(text, model_path="./fine_tuned_bert"):
    if not os.path.exists(model_path):
        print(f"Error: Model path '{model_path}' does not exist. Please run 'train.py' first.")
        return None

    # Load tokenizer and model
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    
    # Load label mapping
    mapping_path = os.path.join(model_path, "label_mapping.json")
    if os.path.exists(mapping_path):
        with open(mapping_path, "r") as f:
            mapping = json.load(f)
            id2label = mapping["id2label"]
    else:
        print("Warning: label_mapping.json not found. Returning class ID instead of label.")
        id2label = None

    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    
    # Inference
    model.eval()
    with torch.no_grad():
        logits = model(**inputs).logits
    
    predicted_class_id = logits.argmax().item()
    
    if id2label:
        return id2label[str(predicted_class_id)]
    return predicted_class_id

if __name__ == "__main__":
    # Example usage
    sample_text = "The economy is showing signs of recovery after the latest fiscal report."
    prediction = predict(sample_text)
    if prediction is not None:
        print(f"Text: {sample_text}")
        print(f"Predicted Class: {prediction}")

    sample_text = "how world play cricket together"
    prediction = predict(sample_text)
    if prediction is not None:
        print(f"Text: {sample_text}")
        print(f"Predicted Class: {prediction}")
