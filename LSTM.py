from cv2 import multiply
import tensorflow as tf



class RNN(tf.keras.layers.Layer):
    def __init__(self, unit_size):
        super(RNN, self).__init__()
        self.unit_size = unit_size
    
    def build(self, input_shape):
        print("Working\n")
        self.h = tf.Variable(tf.zeros(shape=(self.unit_size, input_shape[0]), dtype="float32"))
        Wx_init = tf.random_uniform_initializer()
        self.Wx = tf.Variable(initial_value = Wx_init(shape=(self.unit_size, input_shape[0]),dtype = "float32"),trainable=True)
        Wh_init = tf.random_uniform_initializer()
        self.Wh = tf.Variable(initial_value=Wh_init(shape=(self.unit_size, input_shape[0]),dtype = "float32"),trainable=True)
        self.b = tf.Variable(tf.zeros(shape=(self.unit_size, input_shape[0]), dtype="float32"), trainable=True)

    def call(self, inputs):
        for i in range(len(inputs[0])):
            self.h = tf.tanh(tf.math.multiply(self.Wh, self.h) + tf.math.multiply(self.Wx, inputs[:,i]) + self.b)
        return tf.tanh(tf.math.multiply(self.Wh, self.h) + tf.math.multiply(self.Wx, inputs[:,0]) + self.b)





if __name__ == "__main__":
    rnnLayer = RNN(4)
    y = rnnLayer(tf.ones(shape = (2, 3)))
    print(y)