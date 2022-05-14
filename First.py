import numpy as np
import tensorflow as tf


class MyCallBack(tf.keras.callbacks.Callback):
    def __init__(self):
        self.loss = 1e-4
    def on_epoch_end(self, epoch, logs={}):
        if logs.get("loss") > self.loss:
            self.model.save("linearEquation")

callback = MyCallBack()


x = np.arange(10000, dtype = np.float32)

y = 5 * x - 3 

print(x.shape)

Input = tf.keras.layers.Input(shape = 1)
dense_2 = tf.keras.layers.Dense(1)(Input)
model = tf.keras.models.Model(Input, dense_2)


model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 5e-5), loss = "mae")


weigths = model.layers[1].get_weights()
print(weigths)
print(model.summary())

model.fit(x, y, epochs = 1000, callbacks = [callback])
model.save("linearEquation")
print(model.predict([10]))
weigths = model.layers[1].get_weights()
print(weigths)
