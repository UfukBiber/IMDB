import tensorflow as tf


class BasicRNN(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        self.units = units 
        self.state_size = self.units
        super(BasicRNN, self).__init__()

    def build(self, InputShape):
        self.Wx = self.add_weight(shape=(InputShape[-1], self.units),
                                      initializer='uniform',
                                      name='W')
        self.Wh = self.add_weight(shape=(self.units, self.units),
                                      initializer='uniform',
                                      name='U')
        self.b = self.add_weight(shape = (self.units,), 
                                      initializer = "zeros",
                                      name = "b")
        

    def call(self, inputs, states):
        prev_out = states[0]
        output = tf.matmul(inputs, self.Wx) + tf.matmul(prev_out, self.Wh) + self.b
        new_state = output
        print(new_state.shape)
        return output, [new_state]
        

if __name__=="__main__":
    layer = tf.keras.layers.RNN(BasicRNN(4))
    x = tf.keras.layers.Input(shape=(8,1))
    y = layer(x)
    model = tf.keras.models.Model(x, y)
    y_2 = tf.keras.layers.SimpleRNN(4)(x)
    model_2 = tf.keras.models.Model(x, y_2)
    print(model.summary())
    print(model_2.summary())



        


