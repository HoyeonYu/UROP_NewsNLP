import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input, GRU, Dense, Embedding, GRU

''''''''''''''''''''' Read Preprocessed File '''''''''''''''''''''
csv_read_neutral = 'D:/study/python/UROP/analyzing/analyzed_naverNews.csv'
data = pd.read_csv(csv_read_neutral)

''''''''''''''''''''' Add SOS, EOS Token in Decoder Input, Output '''''''''''''''''''''
data['decoder_input'] = data['title'].apply(lambda x: 'sostoken ' + x)
data['decoder_target'] = data['title'].apply(lambda x: x + ' eostoken')

''''''''''''''''''''' Split Encoder, Decoder '''''''''''''''''''''
encoder_input = np.array(data['contents'])
decoder_input = np.array(data['decoder_input'])
decoder_target = np.array(data['decoder_target'])

''''''''''''''''''''' Make Random Index for Shuffle Data '''''''''''''''''''''
indices = np.arange(encoder_input.shape[0])  # 11827 (Total Data Num)
np.random.shuffle(indices)  # Random Index Array

encoder_input = encoder_input[indices]
decoder_input = decoder_input[indices]
decoder_target = decoder_target[indices]

''''''''''''''''''''' Train : Test = 8 : 2 '''''''''''''''''''''
test_val = int(len(encoder_input) * 0.2)  # train: 9462 / test: 2365

encoder_input_train = encoder_input[:-test_val]
decoder_input_train = decoder_input[:-test_val]
decoder_target_train = decoder_target[:-test_val]

encoder_input_test = encoder_input[-test_val:]
decoder_input_test = decoder_input[-test_val:]
decoder_target_test = decoder_target[-test_val:]

''''''''''''''''''''' Tokenize : Contents '''''''''''''''''''''
contents_tokenizer = Tokenizer()
contents_tokenizer.fit_on_texts(encoder_input_train)

threshold = 5
total_cnt = len(contents_tokenizer.word_index)
rare_cnt = 0
total_freq = 0

for key, value in contents_tokenizer.word_counts.items():
    total_freq += value
    if value < threshold:
        rare_cnt += 1

print('=========================================')
print('Analyze Contents Token\n')
print('Total Word Num:', total_cnt)
print('Frequency < %d: %d (%.2f%%)' % (threshold, rare_cnt, (rare_cnt / total_cnt) * 100))
print('Frequency >= %d: %d (%.2f%%)' % (threshold, total_cnt - rare_cnt, ((total_cnt - rare_cnt) / total_cnt) * 100))
print('=========================================')

''''''''''''''''''''' Re-Tokenize by Checking Contents Word Frequency : Contents '''''''''''''''''''''
# Tokenize Contents
contents_vocab = 30000
contents_tokenizer = Tokenizer(num_words=contents_vocab)
contents_tokenizer.fit_on_texts(encoder_input_train)

''''''''''''''''''''' Text to Sequence : Contents '''''''''''''''''''''
encoder_input_train = contents_tokenizer.texts_to_sequences(encoder_input_train)
encoder_input_test = contents_tokenizer.texts_to_sequences(encoder_input_test)

''''''''''''''''''''' Tokenize : Title '''''''''''''''''''''
title_tokenizer = Tokenizer()
title_tokenizer.fit_on_texts(decoder_input_train)

threshold = 3
total_cnt = len(title_tokenizer.word_index)
rare_cnt = 0
total_freq = 0

for key, value in title_tokenizer.word_counts.items():
    total_freq += value
    if value < threshold:
        rare_cnt += 1

print('=========================================')
print('Analyze Title Token\n')
print('Total Word Num:', total_cnt)
print('Frequency < %d: %d (%.2f%%)' % (threshold, rare_cnt, (rare_cnt / total_cnt) * 100))
print('Frequency >= %d: %d (%.2f%%)' % (threshold, total_cnt - rare_cnt, ((total_cnt - rare_cnt) / total_cnt) * 100))
print('=========================================')

''''''''''''''''''''' Re-Tokenize by Checking Contents Word Frequency : Title '''''''''''''''''''''
title_vocab = 5000
title_tokenizer = Tokenizer(num_words=title_vocab)
title_tokenizer.fit_on_texts(decoder_input_train)
title_tokenizer.fit_on_texts(decoder_target_train)

''''''''''''''''''''' Text to Sequence : Title '''''''''''''''''''''
decoder_input_train = title_tokenizer.texts_to_sequences(decoder_input_train)
decoder_target_train = title_tokenizer.texts_to_sequences(decoder_target_train)
decoder_input_test = title_tokenizer.texts_to_sequences(decoder_input_test)
decoder_target_test = title_tokenizer.texts_to_sequences(decoder_target_test)

''''''''''''''''''''' Drop If Token Size < 3 '''''''''''''''''''''
drop_train = [index for index, sentence in enumerate(decoder_input_train) if len(sentence) < 3]
drop_test = [index for index, sentence in enumerate(decoder_input_test) if len(sentence) < 3]

encoder_input_train = np.delete(encoder_input_train, drop_train, axis=0)
decoder_input_train = np.delete(decoder_input_train, drop_train, axis=0)
decoder_target_train = np.delete(decoder_target_train, drop_train, axis=0)

encoder_input_test = np.delete(encoder_input_test, drop_test, axis=0)
decoder_input_test = np.delete(decoder_input_test, drop_test, axis=0)
decoder_target_test = np.delete(decoder_target_test, drop_test, axis=0)

print('=========================================')
print('Final Data Summary\n')
print('Train Input Size:', len(encoder_input_train))
print('Train Output Size:', len(decoder_input_train))
print('Test Input Size:', len(encoder_input_test))
print('Test Output Size:', len(decoder_input_test))
print('=========================================')

''''''''''''''''''''' Pad Sequences '''''''''''''''''''''
contents_pad_len = 300
title_pad_len = 10

encoder_input_train = pad_sequences(encoder_input_train, maxlen=contents_pad_len)
decoder_input_train = pad_sequences(decoder_input_train, maxlen=title_pad_len)
decoder_target_train = pad_sequences(decoder_target_train, maxlen=title_pad_len)

encoder_input_test = pad_sequences(encoder_input_test, maxlen=contents_pad_len)
decoder_input_test = pad_sequences(decoder_input_test, maxlen=title_pad_len)
decoder_target_test = pad_sequences(decoder_target_test, maxlen=title_pad_len)

''''''''''''''''''''' Build Model '''''''''''''''''''''
embedding_dim_list = [64, 128, 256]
hidden_size_list = [32, 64, 128]


def model_encoder_1_decoder_1():
    ''''''''''''''''''''' Encoder : GRU X 1 '''''''''''''''''''''
    encoder_inputs = Input(shape=(contents_pad_len,))
    enc_emb = Embedding(contents_vocab, embedding_dim)(encoder_inputs)
    encoder_gru = GRU(hidden_size, return_sequences=True, return_state=True, dropout=DROPOUT)
    encoder_outputs, state_h = encoder_gru(enc_emb)

    ''''''''''''''''''''' Decoder : GRU X 1 '''''''''''''''''''''
    decoder_inputs = Input(shape=(None,))
    dec_emb_layer = Embedding(title_vocab, embedding_dim)
    dec_emb = dec_emb_layer(decoder_inputs)

    decoder_gru = GRU(hidden_size, return_sequences=True, return_state=True, dropout=DROPOUT)
    decoder_outputs, _ = decoder_gru(dec_emb, initial_state=state_h)

    decoder_dense = Dense(title_vocab, activation='softmax')
    decoder_softmax_outputs = decoder_dense(decoder_outputs)

    ''''''''''''''''''''' Encoder + Decoder Model '''''''''''''''''''''
    model = Model([encoder_inputs, decoder_inputs], decoder_softmax_outputs)
    model.summary()

    model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')

    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10)

    history_e1d1 = model.fit(x=[encoder_input_train, decoder_input_train], y=decoder_target_train,
                             validation_data=([encoder_input_test, decoder_input_test], decoder_target_test),
                             batch_size=256, callbacks=[early_stopping_callback], epochs=1000)

    return history_e1d1


def model_encoder_1_decoder_3():
    ''''''''''''''''''''' Encoder : GRU X 1 '''''''''''''''''''''
    encoder_inputs = Input(shape=(contents_pad_len,))
    enc_emb = Embedding(contents_vocab, embedding_dim)(encoder_inputs)

    encoder_gru = GRU(hidden_size, return_sequences=True, return_state=True, dropout=DROPOUT)
    encoder_outputs, state_h = encoder_gru(enc_emb)

    ''''''''''''''''''''' Decoder : GRU X 3 '''''''''''''''''''''
    decoder_inputs = Input(shape=(None,))
    dec_emb_layer = Embedding(title_vocab, embedding_dim)
    dec_emb = dec_emb_layer(decoder_inputs)

    decoder_gru1 = GRU(hidden_size, return_sequences=True, return_state=True, dropout=DROPOUT)
    decoder_output1, _ = decoder_gru1(dec_emb, initial_state=state_h)

    decoder_gru2 = GRU(hidden_size, return_sequences=True, return_state=True, dropout=DROPOUT)
    decoder_output2, _ = decoder_gru2(decoder_output1, initial_state=state_h)

    decoder_gru3 = GRU(hidden_size, return_sequences=True, return_state=True, dropout=DROPOUT)
    decoder_outputs, _ = decoder_gru3(decoder_output2, initial_state=state_h)

    decoder_dense = Dense(title_vocab, activation='softmax')
    decoder_softmax_outputs = decoder_dense(decoder_outputs)

    ''''''''''''''''''''' Encoder + Decoder Model '''''''''''''''''''''
    model = Model([encoder_inputs, decoder_inputs], decoder_softmax_outputs)
    model.summary()

    model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')

    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10)

    history_e1d3 = model.fit(x=[encoder_input_train, decoder_input_train], y=decoder_target_train,
                             validation_data=([encoder_input_test, decoder_input_test], decoder_target_test),
                             batch_size=256, callbacks=[early_stopping_callback], epochs=1000)

    return history_e1d3


def model_encoder_3_decoder_1():
    ''''''''''''''''''''' Encoder : GRU X 3 '''''''''''''''''''''
    encoder_inputs = Input(shape=(contents_pad_len,))
    enc_emb = Embedding(contents_vocab, embedding_dim)(encoder_inputs)

    encoder_gru1 = GRU(hidden_size, return_sequences=True, return_state=True, dropout=DROPOUT)
    encoder_output1, state_h = encoder_gru1(enc_emb)

    encoder_gru2 = GRU(hidden_size, return_sequences=True, return_state=True, dropout=DROPOUT)
    encoder_output2, state_h = encoder_gru2(encoder_output1)

    encoder_gru3 = GRU(hidden_size, return_sequences=True, return_state=True, dropout=DROPOUT)
    encoder_outputs, state_h = encoder_gru3(encoder_output2)

    ''''''''''''''''''''' Decoder : GRU X 1 '''''''''''''''''''''
    decoder_inputs = Input(shape=(None,))
    dec_emb_layer = Embedding(title_vocab, embedding_dim)
    dec_emb = dec_emb_layer(decoder_inputs)

    decoder_gru = GRU(hidden_size, return_sequences=True, return_state=True, dropout=DROPOUT)
    decoder_outputs, _ = decoder_gru(dec_emb, initial_state=state_h)

    decoder_dense = Dense(title_vocab, activation='softmax')
    decoder_softmax_outputs = decoder_dense(decoder_outputs)

    ''''''''''''''''''''' Encoder + Decoder Model '''''''''''''''''''''
    model = Model([encoder_inputs, decoder_inputs], decoder_softmax_outputs)
    model.summary()

    model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')

    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10)

    history_e3d1 = model.fit(x=[encoder_input_train, decoder_input_train], y=decoder_target_train,
                             validation_data=([encoder_input_test, decoder_input_test], decoder_target_test),
                             batch_size=256, callbacks=[early_stopping_callback], epochs=1000)

    return history_e3d1


def model_encoder_3_decoder_3():
    ''''''''''''''''''''' Encoder : GRU X 3 '''''''''''''''''''''
    encoder_inputs = Input(shape=(contents_pad_len,))
    enc_emb = Embedding(contents_vocab, embedding_dim)(encoder_inputs)

    encoder_gru1 = GRU(hidden_size, return_sequences=True, return_state=True, dropout=DROPOUT)
    encoder_output1, state_h = encoder_gru1(enc_emb)

    encoder_gru2 = GRU(hidden_size, return_sequences=True, return_state=True, dropout=DROPOUT)
    encoder_output2, state_h = encoder_gru2(encoder_output1)

    encoder_gru3 = GRU(hidden_size, return_sequences=True, return_state=True, dropout=DROPOUT)
    encoder_outputs, state_h = encoder_gru3(encoder_output2)

    ''''''''''''''''''''' Decoder : GRU X 3 '''''''''''''''''''''
    decoder_inputs = Input(shape=(None,))
    dec_emb_layer = Embedding(title_vocab, embedding_dim)
    dec_emb = dec_emb_layer(decoder_inputs)

    decoder_gru1 = GRU(hidden_size, return_sequences=True, return_state=True, dropout=DROPOUT)
    decoder_output1, _ = decoder_gru1(dec_emb, initial_state=state_h)

    decoder_gru2 = GRU(hidden_size, return_sequences=True, return_state=True, dropout=DROPOUT)
    decoder_output2, _ = decoder_gru2(decoder_output1, initial_state=state_h)

    decoder_gru3 = GRU(hidden_size, return_sequences=True, return_state=True, dropout=DROPOUT)
    decoder_outputs, _ = decoder_gru3(decoder_output2, initial_state=state_h)

    decoder_dense = Dense(title_vocab, activation='softmax')
    decoder_softmax_outputs = decoder_dense(decoder_outputs)

    ''''''''''''''''''''' Encoder + Decoder Model '''''''''''''''''''''
    model = Model([encoder_inputs, decoder_inputs], decoder_softmax_outputs)
    model.summary()

    model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')

    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10)

    history_e3d3 = model.fit(x=[encoder_input_train, decoder_input_train], y=decoder_target_train,
                             validation_data=([encoder_input_test, decoder_input_test], decoder_target_test),
                             batch_size=256, callbacks=[early_stopping_callback], epochs=1000)

    return history_e3d3


if __name__ == "__main__":
    info_list = []
    loss_e1d1_list = []
    loss_e1d3_list = []
    loss_e3d1_list = []
    loss_e3d3_list = []
    DROPOUT = 0.5

    plot_dir = 'D:/study/python/UROP/model/neutral/plot/plot_simple_GRU_dropout%d' % (DROPOUT * 100)
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    for embedding_dim in embedding_dim_list:
        for hidden_size in hidden_size_list:
            info_list.append('emb: ' + str(embedding_dim) + ', hidden: ' + str(hidden_size))

            print('\n=========   e1 d1 Start, Emb: %d Hid: %d    ===========' % (embedding_dim, hidden_size))
            history_e1d1 = model_encoder_1_decoder_1()
            loss_e1d1_list.append(history_e1d1.history['val_loss'])
            print('==============   e1 d1 End    ================\n')

            print('\n=========   e1 d3 Start, Emb: %d Hid: %d    ===========' % (embedding_dim, hidden_size))
            history_e1d3 = model_encoder_1_decoder_3()
            loss_e1d3_list.append(history_e1d3.history['val_loss'])
            print('==============   e1 d3 End    ================\n')

            print('\n=========   e3 d1 Start, Emb: %d Hid: %d    ===========' % (embedding_dim, hidden_size))
            history_e3d1 = model_encoder_3_decoder_1()
            loss_e3d1_list.append(history_e3d1.history['val_loss'])
            print('==============   e3 d1 End    ================\n')

            print('\n=========   e3 d3 Start, Emb: %d Hid: %d    ===========' % (embedding_dim, hidden_size))
            history_e3d3 = model_encoder_3_decoder_3()
            loss_e3d3_list.append(history_e3d3.history['val_loss'])
            print('==============   e3 d3 End    ================\n')

            plt.figure()
            plt.plot(history_e1d1.history['val_loss'], label='test_e1d1')
            plt.plot(history_e1d3.history['val_loss'], label='test_e1d3')
            plt.plot(history_e3d1.history['val_loss'], label='test_e3d1')
            plt.plot(history_e3d3.history['val_loss'], label='test_e3d3')

            plt.ylim([1.5, 4])
            plt.legend()
            plt.title('Loss Graph (Embedding Dim: %d, Hidden Size: %d)' % (embedding_dim, hidden_size))
            plt.savefig(plot_dir + '/emb%d_hid%d.png' % (embedding_dim, hidden_size))

    loss_dataframe = []
    loss_dataframe = pd.DataFrame(loss_dataframe, columns=['info', 'e1d1', 'e1d3', 'e3d1', 'e3d3'])
    loss_dataframe['info'] = info_list
    loss_dataframe['e1d1'] = loss_e1d1_list
    loss_dataframe['e1d3'] = loss_e1d3_list
    loss_dataframe['e3d1'] = loss_e3d1_list
    loss_dataframe['e3d3'] = loss_e3d3_list
    loss_dataframe.to_csv('D:/study/python/UROP/model/neutral/loss/loss_simple_GRU_dropout%d.csv' % (DROPOUT * 100), encoding='utf-8-sig', index=True)
