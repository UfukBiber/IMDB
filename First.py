import numpy as np
import tensorflow as tf



x = np.arange(50000, dtype = np.float32)

y = 5 * x - 3 

print(x.shape)

# Input = tf.keras.layers.Input(shape = 1)
# dense_1 = tf.keras.layers.Dense(4)(Input)
# dense_2 = tf.keras.layers.Dense(1)(dense_1)
# model = tf.keras.models.Model(Input, dense_2)


# model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-5), loss = "mae")


model = tf.keras.models.load_model("linearEquation")
weigths = model.layers[1].get_weights()
print(weigths)
print(model.summary())

model.fit(x, y, epochs = 10)
model.save("linearEquation")
print(model.predict([10]))
weigths = model.layers[1].get_weights()
print(weigths)
