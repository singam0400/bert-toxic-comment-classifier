
# utils/attention_viz.py

import numpy as np
import tensorflow as tf
from transformers import TFBertModel, BertTokenizerFast

def colorize(token, weight):
    """Wrap token in HTML span with red background proportional to weight."""
    red_intensity = float(weight)
    return f"<span style='background-color: rgba(255, 0, 0, {red_intensity:.2f}); padding:2px'>{token}</span>"

def visualize_attention(text, model_name='prajjwal1/bert-tiny'):
    """
    Generate attention-weight visualization for given text using BERT.
    Returns an HTML string.
    """
    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    model = TFBertModel.from_pretrained(model_name, output_attentions=True, from_pt=True)

    # Tokenize and get outputs with attention
    inputs = tokenizer(text, return_tensors="tf", truncation=True, padding=True, max_length=128)
    outputs = model(**inputs)

    attentions = outputs.attentions  # List of (batch, heads, seq_len, seq_len)
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

    # Average over layers and heads → final shape: (seq_len, seq_len)
    avg_attn = tf.reduce_mean(tf.stack(attentions), axis=[0, 1])[0]  # (seq_len, seq_len)

    # Use attention from [CLS] token to all tokens
    cls_attn = avg_attn[0]  # Attention from [CLS] → (seq_len,)
    cls_attn /= tf.reduce_max(cls_attn)  # Normalize

    # Create HTML
    html_output = " ".join([colorize(tok, cls_attn[i].numpy()) for i, tok in enumerate(tokens)])
    return html_output
