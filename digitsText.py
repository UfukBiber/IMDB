import tensorflow as tf

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = tf.reshape(x_train / 255.0, (x_train.shape[0], -1))
y_train = tf.expand_dims(y_train, axis = 1)



print((x_train / tf.expand_dims(tf.math.reduce_sum(x_train, axis=1), axis= 1).shape))