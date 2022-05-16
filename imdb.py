import GRU, dense
import tensorflow as tf


(x_train, y_train), _ = tf.keras.datasets.imdb.load_data(maxlen = 100, 
                                                        num_words = 1000)
x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, padding = "post")
y = tf.expand_dims(y_train, axis = 1)



Inp = tf.keras.layers.Input(shape=(x_train.shape[-1]))
embedded = tf.keras.layers.Embedding(1000, 64)(Inp)
output = tf.keras.layers.RNN(GRU.GRU(256))(embedded)
output = dense.dense(64, activation = "relu")(output)
output = dense.dense(1, activation = "sigmoid")(output)



output_2 = tf.keras.layers.GRU(256)(embedded)
output_2 = tf.keras.layers.Dense(64, activation = "relu")(output_2)
output_2 = tf.keras.layers.Dense(1, activation = "sigmoid")(output_2)



model = tf.keras.models.Model(Inp, output_2)
model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])
model.fit(x_train, y_train, epochs = 2)


