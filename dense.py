import tensorflow as tf
import numpy as np

class dense(tf.keras.layers.Layer):
    def __init__(self, units):
        super(dense, self).__init__()
        self.units = units

    def build(self, inputShape):
        W_init = tf.random_uniform_initializer()
        self.W = tf.Variable(initial_value=W_init(shape=(inputShape[-1], self.units), dtype="float32"), trainable=True)
        b_init = tf.zeros_initializer()
        self.b = tf.Variable(initial_value = b_init(shape=1, dtype="float32"), trainable=True)

    def call(self, inputs):
        return tf.matmul(inputs, self.W) + self.b



x = tf.reshape(tf.range(0, 10000, dtype = "float32"), (10000, 1))
y = 2.0 * x + 1.0
print(x.shape, y.shape)

Inp = tf.keras.layers.Input(1)
output = dense(1)(Inp)

model = tf.keras.models.Model(Inp, output)
model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-4), loss = "mae")
model.fit(x, y, epochs = 10)
