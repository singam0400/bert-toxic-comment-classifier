# bert-toxic-comment-classifier
Toxic Comment Classifier using BERT

# 💬 Toxic Comment Classifier using BERT (Future-Ready for Attention Visualization)

A deep learning NLP project using **pre-trained BERT** fine-tuned on a **toxic comment classification task**. The model detects multi-label toxic traits in user comments such as threats, obscenity, and insults. The project is designed to be easily extensible with **self-attention visualizations** (planned as next phase).

---

## 🚀 Project Highlights

- ✅ Fine-tuned BERT (prajjwal1/bert-tiny) on real-world toxicity data  
- ✅ Multi-label classification: toxic, obscene, threat, insult, etc.  
- ✅ TensorFlow 2.x + HuggingFace Transformers  
- ⚙️ Attention visualization module placeholder (ready for extension)

---

## 📦 Dataset

- [Jigsaw Toxic Comment Classification Challenge (Kaggle)](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)
- 150,000+ Wikipedia comments  
- Each comment can belong to multiple classes (multi-label)  
- Classes: `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`

---

## 🧠 Why This Project Matters

This project showcases:
- Real-world NLP pipeline with class imbalance handling  
- Multi-label fine-tuning of transformer models  
- Modular design for research and production  
- (Optional) Hook points for **explainability**, including BERT’s attention heads

---

## 🛠️ Tech Stack

- Python 3.x  
- TensorFlow 2.x  
- HuggingFace Transformers  
- Pandas, NumPy  
- (Optional) BertViz for attention visualization (not yet integrated)

---
