
# model/bert_classifier.py

import tensorflow as tf

from transformers import TFBertModel

def build_model():
    bert_model = TFBertModel.from_pretrained('prajjwal1/bert-tiny', from_pt=True)

    for layer in bert_model.layers:
        layer.trainable = False  # Still freeze for speed

    input_ids = tf.keras.layers.Input(shape=(128,), dtype=tf.int32, name="input_ids")
    attention_mask = tf.keras.layers.Input(shape=(128,), dtype=tf.int32, name="attention_mask")
    token_type_ids = tf.keras.layers.Input(shape=(128,), dtype=tf.int32, name="token_type_ids")

    bert_output = bert_model(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids
    )

    cls_token = bert_output.pooler_output

    x = tf.keras.layers.Dense(128, activation='relu')(cls_token)
    x = tf.keras.layers.Dropout(0.2)(x)
    output = tf.keras.layers.Dense(6, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=[input_ids, attention_mask, token_type_ids], outputs=output)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model
