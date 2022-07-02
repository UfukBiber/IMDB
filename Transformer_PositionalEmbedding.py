import tensorflow  as tf
import numpy as np


BATCH_SIZE = 32
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
                                    output_mode = "int",
                                    max_tokens = 10000,
                                    output_sequence_length = 500
)



text_to_adapt = raw_train_ds.map(lambda x, y: x)
InpVectorization.adapt(text_to_adapt)
vocabularies = InpVectorization.get_vocabulary()


multi_hot_train = raw_train_ds.map(lambda x, y : (InpVectorization(x), y),
                                            num_parallel_calls=4)
multi_hot_val = raw_train_val.map(lambda x, y : (InpVectorization(x), y),
                                            num_parallel_calls=4)
multi_hot_test = raw_test_ds.map(lambda x, y : (InpVectorization(x), y), 
                                            num_parallel_calls=4)                       


################# GLOVE EmbeddingLayer ###############################
embedding_index = {}
with open("glove.6B.100d.txt") as f:
    for line in f:
        word, coefs = line.split(maxsplit=1)
        coefs = np.fromstring(coefs, "float64", sep=" ")
        embedding_index[word] = coefs
    f.close()



words_index = dict(zip(vocabularies, range(len(vocabularies))))
embedding_matrix = np.zeros((10000, 100))
for word, i in words_index.items():
    if i < 10000:
        embedding_vector = embedding_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

#######################################################################

class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, embedDims, sequenceLength):
        super(PositionalEmbedding, self).__init__()
        self.embeddimWord = tf.keras.layers.Embedding(10000, embedDims, 
                                    embeddings_initializer = tf.keras.initializers.Constant(embedding_matrix),
                                    trainable = False)
        self.embeddingPosition = tf.keras.layers.Embedding(sequenceLength, embedDims)
    
    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(start = 0, limit = length, delta = 1)
        embeddedWords = self.embeddimWord(inputs)
        embeddedPos = self.embeddingPosition(positions)
        return embeddedWords + embeddedPos
    
    def compute_mask(self, inputs, mask = None):
        return tf.math.not_equal(inputs, 0)



class Transformer(tf.keras.layers.Layer):
    def __init__(self, num_heads, dense_dimensions, embed_dims):
        super(Transformer, self).__init__()
        self.attention = tf.keras.layers.MultiHeadAttention(num_heads, embed_dims)
        self.denseLayers = tf.keras.Sequential([
            tf.keras.layers.Dense(dense_dimensions),
            tf.keras.layers.Dense(embed_dims)
        ])
        self.layerNormalization_1 = tf.keras.layers.LayerNormalization()
        self.layerNormalization_2 = tf.keras.layers.LayerNormalization()
    
    def call(self, inputs, mask = None):
        if mask is not None:
            mask = mask[:, tf.newaxis, :]
        attentionOutput = self.attention(inputs, inputs, attention_mask=mask)
        projInput = self.layerNormalization_1(inputs + attentionOutput)
        projOut = self.denseLayers(projInput)
        return self.layerNormalization_2(projInput + projOut)

num_heads = 2
embedDims = 100
denseDims = 32
SequenceLength = 500


Inp = tf.keras.layers.Input(shape=(None, ), dtype = "int64")
embedded = PositionalEmbedding(embedDims, SequenceLength)(Inp)
x = Transformer(num_heads, denseDims, embedDims)(embedded)
x = tf.keras.layers.GlobalMaxPooling1D()(x)
x = tf.keras.layers.Dropout(0.5)(x)
output = tf.keras.layers.Dense(1, activation = "sigmoid")(x)

model = tf.keras.models.Model(Inp, output)
print(model.summary())

model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])


model.fit(multi_hot_train, validation_data = multi_hot_val, epochs = 8)
model.evaluate(multi_hot_test)