

# 🧠 Semantic Text Classification

A high-performance **semantic text classification system** powered by a fine-tuned BERT transformer model.  
This project intelligently categorizes unstructured text into key domains using deep contextual understanding and optimized neural inference.

---

## 🚀 Live Demo

👉 [https://huggingface.co/spaces/sumitsingh159/Text_Classification](https://huggingface.co/spaces/sumitsingh159/Text_Classification)  

Experience real-time classification with a clean, interactive UI.

---

## 📌 Overview

This project implements a **deep learning-based NLP pipeline** that automatically classifies raw text into four major categories:

- 🖥️ **Technology**  
- ⚽ **Sports**  
- 🏛️ **Politics**  
- 🎬 **Entertainment**  

Unlike traditional keyword-based systems, this model understands:
- Semantic meaning  
- Context  
- Linguistic nuance  

---

## 🧠 Model Architecture

| Component   | Details                        |
|------------|--------------------------------|
| Base Model | bert-base-uncased              |
| Framework  | PyTorch + Transformers         |
| Task       | Sequence Classification        |
| Output     | Class probabilities            |

The model is fine-tuned to capture **domain-specific language patterns** and improve classification accuracy.

---

## ⚙️ Features

- ⚡ **Real-time inference**  
- 🧾 **Context-aware predictions**  
- 📊 **Confidence score output**  
- 🎯 **Optimized inference pipeline**  
- 🎨 **Clean Gradio UI**  
- 🔁 **Cached predictions (LRU Cache)**  

---

## 🖥️ User Interface

The application includes a minimal and intuitive UI:
- Text input panel  
- Prediction output with confidence scores  
- Example inputs for quick testing  
- Fast and responsive layout  

---

## 🏗️ Project Structure

```text
.
├── app.py
├── fine_tuned_bert/    # (Local model directory)
├── requirements.txt
├── README.md
└── label_mapping.json
```

---

## 🔧 Installation

```bash
git clone https://github.com/sumitsingh159/Text-Classifier.git
cd Text-Classifier

python -m venv .venv
source .venv/bin/activate   # Linux / Mac
# .venv\Scripts\activate    # Windows

pip install -r requirements.txt
```

### ▶️ Run Locally
```bash
python app.py
```
Open: [http://localhost:7860](http://localhost:7860)

---

## 📥 Model Handling

The trained model is not included in this repository due to GitHub file size limits. 
The app automatically falls back to: `bert-base-uncased`

To use your fine-tuned model:
1. Place it inside `fine_tuned_bert/`
2. Ensure `config.json` and `model.safetensors` are present.

---

## 🧪 Example Inputs

| Input Text | Category |
| :--- | :--- |
| AI is transforming healthcare | Technology |
| The team won the championship | Sports |
| Government introduced reforms | Politics |
| A critically acclaimed movie | Entertainment |

---

## ⚡ Performance Optimizations
- `torch.inference_mode()` for faster inference
- **LRU caching** for repeated inputs
- **Automatic CPU/GPU selection**
- **Efficient tokenization** with truncation

---

## 🔮 Future Improvements
- 🌐 Multi-language support
- 📊 More categories
- 🧠 Model optimization (quantization)
- ☁️ FastAPI deployment
- 📈 Analytics dashboard

---

## 🛠️ Tech Stack
- **Python**
- **PyTorch**
- **Hugging Face Transformers**
- **Gradio**

---

## 👤 Author

**Sumit Singh**  
GitHub: [https://github.com/sumitsingh159](https://github.com/sumitsingh159)  
Hugging Face: [https://huggingface.co/sumitsingh159](https://huggingface.co/sumitsingh159)

---

## 📄 License
This project is licensed under the MIT License.

---

## ⭐ Support
If you found this project useful, consider giving it a star ⭐
