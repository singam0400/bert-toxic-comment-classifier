
# utils/data_loader.py

import pandas as pd
import tensorflow as tf
from transformers import BertTokenizerFast
from sklearn.model_selection import train_test_split

MAX_LEN = 128  # You can adjust based on memory
LABEL_COLUMNS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

def load_data(path='data/toxic_comments.csv'):
    df = pd.read_csv(path)
    df = df[['comment_text'] + LABEL_COLUMNS]
    df.dropna(inplace=True)
    return df

def get_tokenizer():
    return BertTokenizerFast.from_pretrained('prajjwal1/bert-tiny')

def encode_texts(df, tokenizer):
    encodings = tokenizer(
        df["comment_text"].tolist(),
        truncation=True,
        padding='max_length',
        max_length=MAX_LEN,
        return_tensors="tf"
    )
    labels = tf.convert_to_tensor(df[LABEL_COLUMNS].values, dtype=tf.float32)
    return encodings, labels

def prepare_datasets(encodings, labels, batch_size=16, test_size=0.1):
    input_ids = encodings["input_ids"]
    attention_mask = encodings["attention_mask"]

    input_ids = encodings["input_ids"].numpy()
    attention_mask = encodings["attention_mask"].numpy()
    token_type_ids = encodings["token_type_ids"].numpy()
    y = labels.numpy()
    
    # Split raw arrays directly
    ids_train, ids_val, mask_train, mask_val, type_train, type_val, y_train, y_val = train_test_split(
        input_ids, attention_mask, token_type_ids, y, test_size=test_size, random_state=42
    )
    
    # Now wrap into dicts
    X_train = {
        "input_ids": ids_train,
        "attention_mask": mask_train,
        "token_type_ids": type_train
    }
    X_val = {
        "input_ids": ids_val,
        "attention_mask": mask_val,
        "token_type_ids": type_val
    }


    def create_tf_dataset(X_dict, y_tensor):
        ds = tf.data.Dataset.from_tensor_slices((X_dict, y_tensor))
        return ds.shuffle(10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    train_ds = create_tf_dataset(X_train, y_train)
    val_ds = create_tf_dataset(X_val, y_val)

    return train_ds, val_ds
