import tensorflow as tf
import dense
import numpy as np

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = tf.reshape(x_train / 255.0, (x_train.shape[0], -1))
y_train = tf.expand_dims(y_train, axis = 1)



model = tf.keras.Sequential([
    dense.dense(256, activation="relu"),
    dense.dense(128, activation="relu"),
    dense.dense(10, activation="softmax")
])


model.compile(optimizer="adam", loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])
model.fit(x_train, y_train, epochs = 3, validation_split = 0.1)

x_test = tf.reshape(x_test / 255.0, (x_test.shape[0], -1))
y_test = tf.expand_dims(y_test, axis = 1)
y = model.predict(tf.expand_dims(x_test[0], axis=0))

print(y)
print(tf.math.reduce_sum(y))
print(tf.math.argmax(y, axis=1))
print(y_test[0])