import tensorflow  as tf
import numpy as np


BATCH_SIZE = 32
SEED = 123
VALIDATION_SPLIT = 0.1

raw_train_ds = tf.keras.utils.text_dataset_from_directory(
    'aclImdb/train',
    batch_size=BATCH_SIZE,
    validation_split=VALIDATION_SPLIT,
    subset='training',
    seed=SEED)


raw_train_val = tf.keras.utils.text_dataset_from_directory(
    'aclImdb/train',
    batch_size=BATCH_SIZE,
    validation_split=VALIDATION_SPLIT,
    subset='validation',
    seed=SEED)

raw_test_ds = tf.keras.utils.text_dataset_from_directory(
    "aclImdb/train",
    batch_size = BATCH_SIZE
)

InpVectorization = tf.keras.layers.TextVectorization(
                                    output_mode = "int",
                                    max_tokens = 10000,
                                    output_sequence_length = 500
)

text_to_adapt = raw_train_ds.map(lambda x, y: x)
InpVectorization.adapt(text_to_adapt)

multi_hot_train = raw_train_ds.map(lambda x, y : (InpVectorization(x), y),
                                            num_parallel_calls=4)
multi_hot_val = raw_train_val.map(lambda x, y : (InpVectorization(x), y),
                                            num_parallel_calls=4)
multi_hot_test = raw_test_ds.map(lambda x, y : (InpVectorization(x), y), 
                                            num_parallel_calls=4)                       

Inp = tf.keras.layers.Input(shape=(None, ), dtype = "int64")
embedded = tf.keras.layers.Embedding(10000, 32, mask_zero = True)(Inp)
x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32))(embedded)
x = tf.keras.layers.Dropout(0.5)(x)
output = tf.keras.layers.Dense(1, activation = "sigmoid")(x)

model = tf.keras.models.Model(Inp, output)

model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])


model.fit(multi_hot_train, validation_data = multi_hot_val, epochs = 5)
model.evaluate(multi_hot_test)
