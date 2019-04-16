# coding:utf-8
from keras.layers.core import Dropout
from keras.layers.merge import concatenate
from keras.layers import TimeDistributed, Input, Dense, Embedding, Conv1D, GlobalMaxPooling1D, Flatten
from keras.models import Model
from keras import optimizers


def mlp(sourcevocabsize, targetvocabsize, word_W, input_seq_lenth, output_seq_lenth, emd_dim, sourcecharsize, char_W,
        input_word_length, char_emd_dim, batch_size):
    char_input = Input(shape=(input_seq_lenth, input_word_length,), dtype='int32')
    char_embedding = Embedding(input_dim=sourcecharsize, output_dim=char_emd_dim,
                               batch_input_shape=(batch_size, input_seq_lenth, input_word_length),
                               mask_zero=False,
                               trainable=True,
                               weights=[char_W])

    char_embedding2 = TimeDistributed(char_embedding)(char_input)
    char_cnn = TimeDistributed(Conv1D(100, 2, activation='relu', padding='valid'))(char_embedding2)
    char_macpool = TimeDistributed(GlobalMaxPooling1D())(char_cnn)
    char_macpool = Dropout(0.5)(char_macpool)

    word_input = Input(shape=(input_seq_lenth,), dtype='int32')
    word_embedding = Embedding(input_dim=sourcevocabsize + 1,
                               output_dim=emd_dim,
                               input_length=input_seq_lenth,
                               mask_zero=False,
                               trainable=True,
                               weights=[word_W])(word_input)
    word_embedding = Dropout(0.5)(word_embedding)
    embedding = concatenate([word_embedding, char_macpool], axis=-1)

    # define three hidden layer using Dense
    mlp_hidden0 = Flatten()(embedding)
    mlp_hidden1 = Dense(128, activation='relu')(mlp_hidden0)
    mlp_hidden2 = Dense(64, activation='relu')(mlp_hidden1)
    mlp_hidden3 = Dense(32, activation='relu')(mlp_hidden2)

    output = Dense(targetvocabsize, activation='softmax')(mlp_hidden3)
    model = Model([word_input, char_input], output)
    model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=0.005), metrics=['acc'])
    return model


if __name__ == "__main__":
    batch_size = 128
