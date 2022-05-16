from random import random
import tensorflow as tf


class GRU(tf.keras.layers.Layer):
    def __init__(self, unitSize):   
        self.units = unitSize
        self.state_size = self.units
        super(GRU, self).__init__()
    def build(self, inputShape):
        Wz_init = tf.random_normal_initializer()
        Wr_init = tf.random_normal_initializer()
        W_init = tf.random_normal_initializer()

        self.Wz = tf.Variable(initial_value=Wz_init(shape=(inputShape[-1], self.units), dtype="float32"), trainable=True)
        self.Wr = tf.Variable(initial_value=Wr_init(shape = (inputShape[-1], self.units), dtype="float32"), trainable = True)    
        self.W = tf.Variable(initial_value=W_init(shape=(inputShape[-1], self.units),dtype="float32"), trainable = True)

        Uz_init = tf.random_normal_initializer()
        Ur_init = tf.random_normal_initializer()
        U_init = tf.random_normal_initializer()

        self.Uz = tf.Variable(initial_value=Uz_init(shape=(self.units, self.units), dtype="float32"), trainable=True)
        self.Ur = tf.Variable(initial_value=Ur_init(shape = (self.units, self.units), dtype="float32"), trainable = True)    
        self.U = tf.Variable(initial_value=U_init(shape=(self.units, self.units),dtype="float32"), trainable = True)

        bz_init = tf.zeros_initializer()
        br_init = tf.zeros_initializer()
        b_init = tf.zeros_initializer()

        self.bz = tf.Variable(initial_value=bz_init(shape=(self.units, ), dtype="float32"), trainable=True)
        self.br = tf.Variable(initial_value=br_init(shape = (self.units,), dtype="float32"), trainable = True)    
        self.b = tf.Variable(initial_value=b_init(shape=(self.units,), dtype="float32"), trainable = True)


    def call(self, inputs, states): 
        prev_out = states[0]
        zt = tf.sigmoid((tf.matmul(inputs, self.Wz) + tf.matmul(prev_out, self.Uz) + self.bz))
        rt = tf.sigmoid((tf.matmul(inputs, self.Wr) + tf.matmul(prev_out, self.Ur) + self.br))
        hHat = tf.tanh((tf.matmul(inputs, self.W) + tf.math.multiply(rt, prev_out) + self.b))
        new_state = tf.math.multiply(zt, hHat) + tf.math.multiply((1- zt), prev_out)
        output = new_state
        # output = tf.matmul(new_state, self.Wo) + self.bo
        return output, [new_state]


if __name__ == "__main__":
    gru = tf.keras.layers.RNN(GRU(5),
                            return_sequences = True,
                            return_state = True)

    Inp = tf.keras.layers.Input(shape = (5, 1), dtype="float32")
    Out = gru(Inp)
    model = tf.keras.models.Model(Inp, Out)
    
    # lstm = tf.keras.layers.RNN(tf.keras.layers.GRUCell(4),
    #                            return_sequences = True,
    #                            return_state = True)
    
    # Out_2 = lstm(Inp)
    # model_2 = tf.keras.models.Model(Inp, Out_2)

    # # print(model.summary())
    # # print(model_2.summary())
    weight_1 = gru.Variable
    # weight_2 = lstm.get_weights()
    for i in weight_1:
        print(i.shape)

    