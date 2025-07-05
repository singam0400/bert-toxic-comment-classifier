
# main.py

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_auc_score

from utils.data_loader import load_data, get_tokenizer, encode_texts, prepare_datasets
from model.bert_classifier import build_model

# ====== Step 1: Load and Preprocess Data ======
LABEL_COLUMNS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

print(" Loading data...")
df = load_data("data/toxic_comments.csv").sample(2000, random_state=42)  # sample for speed

tokenizer = get_tokenizer()
encodings, labels = encode_texts(df, tokenizer)

train_ds, val_ds = prepare_datasets(encodings, labels, batch_size=32)

# ====== Step 2: Build the Model ======
print("Building model...")
model = build_model()
model.summary()

# ====== Step 3: Callbacks ======
os.makedirs("output", exist_ok=True)

checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    filepath="output/best_model.h5",
    save_best_only=True,
    monitor="val_loss",
    mode="min",
    verbose=1
)

earlystop_cb = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=2,
    restore_best_weights=True,
    verbose=1
)

# ====== Step 4: Train the Model ======
print(" Starting training...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=2,
    callbacks=[checkpoint_cb, earlystop_cb]
)

# ====== Step 5: Plot Training History ======
def plot_training_history(history):
    plt.figure(figsize=(12, 5))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig("output/training_curve.png")
    plt.show()

plot_training_history(history)

# ====== Step 6: Evaluation (F1, Precision, Recall, AUC) ======
def evaluate_model(model, val_ds):
    y_true = []
    y_pred = []

    print("Evaluating model...")

    for batch in val_ds:
        inputs, labels = batch
        preds = model.predict(inputs, verbose=0)
        y_pred.append(preds)
        y_true.append(labels.numpy())

    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)

    # Convert sigmoid outputs to binary predictions
    y_pred_binary = (y_pred > 0.5).astype(int)

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred_binary, target_names=LABEL_COLUMNS, digits=4))

    try:
        auc = roc_auc_score(y_true, y_pred, average='macro')
        print(f"\n Macro ROC AUC Score: {auc:.4f}")
    except:
        print(" AUC could not be computed due to single-class prediction.")

evaluate_model(model, val_ds)


'''
from utils.attention_viz import visualize_attention
from IPython.core.display import display, HTML

sample_text = "You are the worst, disgusting human being."

html_output = visualize_attention(sample_text)
display(HTML(html_output))  # This works only inside Jupyter
'''
