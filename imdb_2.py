import tensorflow  as tf
import numpy as np


BATCH_SIZE = 256
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
                                    output_mode = "multi_hot",
                                    max_tokens = 20000,
                                    ngrams = 2
)

text_to_adapt = raw_train_ds.map(lambda x, y: x)
InpVectorization.adapt(text_to_adapt)

one_gram_multi_hot_train = raw_train_ds.map(lambda x, y : (InpVectorization(x), y),
                                            num_parallel_calls=4)
one_gram_multi_hot_val = raw_train_val.map(lambda x, y : (InpVectorization(x), y),
                                            num_parallel_calls=4)
one_gram_multi_hot_test = raw_test_ds.map(lambda x, y : (InpVectorization(x), y), 
                                            num_parallel_calls=4)                       

model = tf.keras.Sequential([ 
    tf.keras.layers.Dense(16, activation = "relu"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation = "sigmoid")
])

model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])

callbacks = [tf.keras.callbacks.ModelCheckpoint("multihot_model",
                                        save_best_only = True)]
model.fit(one_gram_multi_hot_train.cache(), validation_data = one_gram_multi_hot_val.cache(), epochs = 10, callbacks = callbacks)
model.evaluate(one_gram_multi_hot_test)