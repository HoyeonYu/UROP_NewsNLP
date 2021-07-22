import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding, GRU

''''''''''''''''''''' Read Preprocessed File '''''''''''''''''''''
csv_read_neutral = 'D:/study/python/UROP/preprocessing/preprocessed_naverNews.csv'
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
contents_pad_len = 500
title_pad_len = 14

encoder_input_train = pad_sequences(encoder_input_train, maxlen=contents_pad_len)
decoder_input_train = pad_sequences(decoder_input_train, maxlen=title_pad_len)
decoder_target_train = pad_sequences(decoder_target_train, maxlen=title_pad_len)

encoder_input_test = pad_sequences(encoder_input_test, maxlen=contents_pad_len)
decoder_input_test = pad_sequences(decoder_input_test, maxlen=title_pad_len)
decoder_target_test = pad_sequences(decoder_target_test, maxlen=title_pad_len)

''''''''''''''''''''' Build Model '''''''''''''''''''''
embedding_dim_list = [32, 64, 128, 256]
hidden_size_list = [32, 64, 128, 256]


def model_encoder_1_decoder_1():
    ''''''''''''''''''''' Encoder : LSTM X 1 '''''''''''''''''''''
    encoder_inputs = Input(shape=(contents_pad_len,))
    enc_emb = Embedding(contents_vocab, embedding_dim)(encoder_inputs)
    encoder_gru = GRU(hidden_size, return_sequences=True, return_state=True, dropout=0.4)
    encoder_outputs, state_h, state_c = encoder_gru(enc_emb)

    decoder_inputs = Input(shape=(None,))

    ''''''''''''''''''''' Decoder : LSTM X 1 '''''''''''''''''''''
    dec_emb_layer = Embedding(title_vocab, embedding_dim)
    dec_emb = dec_emb_layer(decoder_inputs)

    decoder_gru = GRU(hidden_size, return_sequences=True, return_state=True, dropout=0.4)
    decoder_outputs, _, _ = decoder_gru(dec_emb, initial_state=[state_h, state_c])

    decoder_softmax_layer = Dense(title_vocab, activation='softmax')
    decoder_softmax_outputs = decoder_softmax_layer(decoder_outputs)

    ''''''''''''''''''''' Encoder + Decoder Model '''''''''''''''''''''
    model = Model([encoder_inputs, decoder_inputs], decoder_softmax_outputs)
    model.summary()

    model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')

    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10)

    history_e1d1 = model.fit(x=[encoder_input_train, decoder_input_train], y=decoder_target_train,
                             validation_data=([encoder_input_test, decoder_input_test], decoder_target_test),
                             batch_size=256, callbacks=[early_stopping_callback], epochs=50)

    model.save('neutral/e1d1_emb%dhid%d.h5' % (embedding_dim, hidden_size))
    return history_e1d1


def model_encoder_1_decoder_3():
    ''''''''''''''''''''' Encoder : LSTM X 1 '''''''''''''''''''''
    encoder_inputs = Input(shape=(contents_pad_len,))
    enc_emb = Embedding(contents_vocab, embedding_dim)(encoder_inputs)

    encoder_lstm1 = LSTM(hidden_size, return_sequences=True, return_state=True, dropout=0.4)
    encoder_outputs, state_h, state_c = encoder_lstm1(enc_emb)

    decoder_inputs = Input(shape=(None,))

    ''''''''''''''''''''' Decoder : LSTM X 3 '''''''''''''''''''''
    dec_emb_layer = Embedding(title_vocab, embedding_dim)
    dec_emb = dec_emb_layer(decoder_inputs)

    decoder_lstm1 = LSTM(hidden_size, return_sequences=True, return_state=True, dropout=0.4)
    decoder_output1, _, _ = decoder_lstm1(dec_emb, initial_state=[state_h, state_c])

    decoder_lstm2 = LSTM(hidden_size, return_sequences=True, return_state=True, dropout=0.4)
    decoder_output2, _, _ = decoder_lstm2(decoder_output1, initial_state=[state_h, state_c])

    decoder_lstm3 = LSTM(hidden_size, return_sequences=True, return_state=True, dropout=0.4)
    decoder_outputs, _, _ = decoder_lstm3(decoder_output2, initial_state=[state_h, state_c])

    decoder_softmax_layer = Dense(title_vocab, activation='softmax')
    decoder_softmax_outputs = decoder_softmax_layer(decoder_outputs)

    ''''''''''''''''''''' Encoder + Decoder Model '''''''''''''''''''''
    model = Model([encoder_inputs, decoder_inputs], decoder_softmax_outputs)
    model.summary()

    model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')

    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10)

    history_e1d3 = model.fit(x=[encoder_input_train, decoder_input_train], y=decoder_target_train,
                             validation_data=([encoder_input_test, decoder_input_test], decoder_target_test),
                             batch_size=256, callbacks=[early_stopping_callback], epochs=50)

    model.save('neutral/e1d3_emb%dhid%d.h5' % (embedding_dim, hidden_size))
    return history_e1d3


def model_encoder_3_decoder_1():
    ''''''''''''''''''''' Encoder : LSTM X 3 '''''''''''''''''''''
    encoder_inputs = Input(shape=(contents_pad_len,))
    enc_emb = Embedding(contents_vocab, embedding_dim)(encoder_inputs)

    encoder_lstm1 = LSTM(hidden_size, return_sequences=True, return_state=True, dropout=0.4)
    encoder_output1, state_h1, state_c1 = encoder_lstm1(enc_emb)

    encoder_lstm2 = LSTM(hidden_size, return_sequences=True, return_state=True, dropout=0.4)
    encoder_output2, state_h2, state_c2 = encoder_lstm2(encoder_output1)

    encoder_lstm3 = LSTM(hidden_size, return_state=True, return_sequences=True, dropout=0.4)
    encoder_outputs, state_h, state_c = encoder_lstm3(encoder_output2)

    decoder_inputs = Input(shape=(None,))

    ''''''''''''''''''''' Decoder : LSTM X 1 '''''''''''''''''''''
    dec_emb_layer = Embedding(title_vocab, embedding_dim)
    dec_emb = dec_emb_layer(decoder_inputs)

    decoder_lstm = LSTM(hidden_size, return_sequences=True, return_state=True, dropout=0.4)
    decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=[state_h, state_c])

    decoder_softmax_layer = Dense(title_vocab, activation='softmax')
    decoder_softmax_outputs = decoder_softmax_layer(decoder_outputs)

    ''''''''''''''''''''' Encoder + Decoder Model '''''''''''''''''''''
    model = Model([encoder_inputs, decoder_inputs], decoder_softmax_outputs)
    model.summary()

    model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')

    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10)

    history_e3d1 = model.fit(x=[encoder_input_train, decoder_input_train], y=decoder_target_train,
                             validation_data=([encoder_input_test, decoder_input_test], decoder_target_test),
                             batch_size=256, callbacks=[early_stopping_callback], epochs=50)

    model.save('neutral/e3d1_emb%dhid%d.h5' % (embedding_dim, hidden_size))
    return history_e3d1


def model_encoder_3_decoder_3():
    ''''''''''''''''''''' Encoder : LSTM X 3 '''''''''''''''''''''
    encoder_inputs = Input(shape=(contents_pad_len,))
    enc_emb = Embedding(contents_vocab, embedding_dim)(encoder_inputs)

    encoder_lstm1 = LSTM(hidden_size, return_sequences=True, return_state=True, dropout=0.4)
    encoder_output1, state_h1, state_c1 = encoder_lstm1(enc_emb)

    encoder_lstm2 = LSTM(hidden_size, return_sequences=True, return_state=True, dropout=0.4)
    encoder_output2, state_h2, state_c2 = encoder_lstm2(encoder_output1)

    encoder_lstm3 = LSTM(hidden_size, return_state=True, return_sequences=True, dropout=0.4)
    encoder_outputs, state_h, state_c = encoder_lstm3(encoder_output2)

    decoder_inputs = Input(shape=(None,))

    ''''''''''''''''''''' Decoder : LSTM X 3 '''''''''''''''''''''
    dec_emb_layer = Embedding(title_vocab, embedding_dim)
    dec_emb = dec_emb_layer(decoder_inputs)

    decoder_lstm1 = LSTM(hidden_size, return_sequences=True, return_state=True, dropout=0.4)
    decoder_output1, _, _ = decoder_lstm1(dec_emb, initial_state=[state_h, state_c])

    decoder_lstm2 = LSTM(hidden_size, return_sequences=True, return_state=True, dropout=0.4)
    decoder_output2, _, _ = decoder_lstm2(decoder_output1, initial_state=[state_h, state_c])

    decoder_lstm3 = LSTM(hidden_size, return_sequences=True, return_state=True, dropout=0.4)
    decoder_outputs, _, _ = decoder_lstm3(decoder_output2, initial_state=[state_h, state_c])

    decoder_softmax_layer = Dense(title_vocab, activation='softmax')
    decoder_softmax_outputs = decoder_softmax_layer(decoder_outputs)

    ''''''''''''''''''''' Encoder + Decoder Model '''''''''''''''''''''
    model = Model([encoder_inputs, decoder_inputs], decoder_softmax_outputs)
    model.summary()

    model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')

    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10)

    history_e3d3 = model.fit(x=[encoder_input_train, decoder_input_train], y=decoder_target_train,
                             validation_data=([encoder_input_test, decoder_input_test], decoder_target_test),
                             batch_size=256, callbacks=[early_stopping_callback], epochs=50)

    model.save('neutral/e3d3_emb%dhid%d.h5' % (embedding_dim, hidden_size))
    return history_e3d3


if __name__ == "__main__":
    for embedding_dim in embedding_dim_list:
        for hidden_size in hidden_size_list:
            history_e1d1 = model_encoder_1_decoder_1()
            history_e1d3 = model_encoder_1_decoder_3()
            history_e3d1 = model_encoder_3_decoder_1()
            history_e3d3 = model_encoder_3_decoder_3()

            plt.plot(history_e1d1.history['loss'], label='train_e1d1')
            plt.plot(history_e1d1.history['val_loss'], label='test_e1d1')

            plt.plot(history_e1d3.history['loss'], label='train_e1d3')
            plt.plot(history_e1d3.history['val_loss'], label='test_e1d3')

            plt.plot(history_e3d1.history['loss'], label='train_e3d1')
            plt.plot(history_e3d1.history['val_loss'], label='test_e3d1')

            plt.plot(history_e3d3.history['loss'], label='train_e3d3')
            plt.plot(history_e3d3.history['val_loss'], label='test_e3d3')

            plt.ylim([0, 3])
            plt.legend()
            plt.title('Loss Graph (Embedding Dim: %d, Hidden Size: %d)' % (embedding_dim, hidden_size))
            plt.savefig('neutral/emb%d_hid%d.png' % (embedding_dim, hidden_size))

