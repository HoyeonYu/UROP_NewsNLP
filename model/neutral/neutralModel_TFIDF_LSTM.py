import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from keras_preprocessing.text import Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding

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
indices = np.arange(encoder_input.shape[0])
np.random.shuffle(indices)

encoder_input = encoder_input[indices]
decoder_input = decoder_input[indices]
decoder_target = decoder_target[indices]

''''''''''''''''''''' Train : Test = 8 : 2 '''''''''''''''''''''
test_val = int(len(encoder_input) * 0.2)

encoder_input_train = encoder_input[:-test_val]
decoder_input_train = decoder_input[:-test_val]
decoder_target_train = decoder_target[:-test_val]

encoder_input_test = encoder_input[-test_val:]
decoder_input_test = decoder_input[-test_val:]
decoder_target_test = decoder_target[-test_val:]

''''''''''''''''''''' Tokenize : Contents '''''''''''''''''''''
contents_tfidf_tokenizer = TfidfVectorizer()  # Contents : TF-IDF, Title : Simple Tokenizer
contents_tfidf_tokenizer.fit(encoder_input_train)
total_cnt = len(contents_tfidf_tokenizer.vocabulary_)

print('=========================================')
print('Analyze Contents Token')
print('Total Word Num:', total_cnt)
print('=========================================')

''''''''''''''''''''' TF-IDF : Contents (Train) '''''''''''''''''''''
encoder_input_train_tfidf = contents_tfidf_tokenizer.transform(encoder_input_train).toarray()

contents_train_word_to_index_tfidf = contents_tfidf_tokenizer.vocabulary_
contents_train_index_to_word_tfidf = dict(map(reversed, contents_train_word_to_index_tfidf.items()))

''''''''''''''''''''' Filtering TF-IDF > 0 and Save with Map Structure : Contents (Train) '''''''''''''''''''''
encoder_input_train_tfidf_list = []
for sentence_tfidf_idx, sentence_tfidf in enumerate(encoder_input_train_tfidf):
    encoder_input_train_tfidf_map = {}
    print('contents train tfidf > 0 :', sentence_tfidf_idx, '/', len(encoder_input_train_tfidf))
    for word_idx, word_tfidf in enumerate(sentence_tfidf):
        if word_tfidf > 0:
            encoder_input_train_tfidf_map[word_idx] = word_tfidf
    encoder_input_train_tfidf_list.append(encoder_input_train_tfidf_map)

# print(encoder_input_train_tfidf_list)
print('contents train filtering tfidf > 0 done')

''''''''''''''''''''' Sort TF-IDF Descending Order, Filtering Rank < 40 : Contents (Train) '''''''''''''''''''''
contents_tfidf_len = 40

contents_train_tfidf_token_list = []
for sentence_idx, encoder_input_train_tfidf_map in enumerate(encoder_input_train_tfidf_list):
    contents_train_tfidf_token = []
    encoder_input_train_tfidf_map = sorted(encoder_input_train_tfidf_map.items(), key=lambda x: x[1], reverse=True)
    print('contents train tfidf sort, rank < %d :' % contents_tfidf_len, sentence_idx, '/', len(encoder_input_train))
    for key, value in encoder_input_train_tfidf_map:
        if len(contents_train_tfidf_token) >= contents_tfidf_len:
            break
        contents_train_tfidf_token.append(contents_train_index_to_word_tfidf[key])
    contents_train_tfidf_token_list.append(contents_train_tfidf_token)

print('contents train filtering rank done')

''''''''''''''''''''' Make New Sentence by Using Tokenizer and Filtered TF-IDF '''''''''''''''''''''
contents_tokenizer = Tokenizer()
contents_tokenizer.fit_on_texts(encoder_input_train)

encoder_input_train = contents_tokenizer.texts_to_sequences(encoder_input_train)
contents_index_to_word = contents_tokenizer.index_word

new_encoder_input_train = []
for sentence_idx, sentence in enumerate(encoder_input_train):
    new_encoder_input_train_word = []
    new_encoder_input_train_sentence = ''

    for word_idx, word in enumerate(sentence):
        if contents_index_to_word[word] in contents_train_tfidf_token_list[sentence_idx] \
                and contents_index_to_word[word] not in new_encoder_input_train_word:
            new_encoder_input_train_word.append(contents_index_to_word[word])
            new_encoder_input_train_sentence += contents_index_to_word[word] + ' '
            print(contents_index_to_word[word], end=' ')

    new_encoder_input_train.append(new_encoder_input_train_sentence)
    print('\n')

print(new_encoder_input_train)

''''''''''''''''''''' TF-IDF : Contents (Test) '''''''''''''''''''''
encoder_input_test_tfidf = contents_tfidf_tokenizer.transform(encoder_input_test).toarray()

contents_test_word_to_index_tfidf = contents_tfidf_tokenizer.vocabulary_
contents_test_index_to_word_tfidf = dict(map(reversed, contents_test_word_to_index_tfidf.items()))

''''''''''''''''''''' Filtering TF-IDF > 0 and Save with Map Structure : Contents (Test) '''''''''''''''''''''
encoder_input_test_tfidf_list = []
for sentence_tfidf_idx, sentence_tfidf in enumerate(encoder_input_test_tfidf):
    encoder_input_test_tfidf_map = {}
    print('contents test tfidf > 0 :', sentence_tfidf_idx, '/', len(encoder_input_test_tfidf))
    for word_idx, word_tfidf in enumerate(sentence_tfidf):
        if word_tfidf > 0:
            encoder_input_test_tfidf_map[word_idx] = word_tfidf
    encoder_input_test_tfidf_list.append(encoder_input_test_tfidf_map)

# print(encoder_input_test_tfidf_list)
print('contents test filtering tfidf > 0 done')

''''''''''''''''''''' Sort TF-IDF Descending Order, Filtering Rank < 40 : Contents (Test) '''''''''''''''''''''
contents_tfidf_len = 40

contents_test_tfidf_token_list = []
for sentence_idx, encoder_input_test_tfidf_map in enumerate(encoder_input_test_tfidf_list):
    contents_test_tfidf_token = []
    encoder_input_test_tfidf_map = sorted(encoder_input_test_tfidf_map.items(), key=lambda x: x[1], reverse=True)
    print('contents test tfidf sort, rank < %d :' % contents_tfidf_len, sentence_idx, '/', len(encoder_input_test))
    for key, value in encoder_input_test_tfidf_map:
        if len(contents_test_tfidf_token) >= contents_tfidf_len:
            break
        contents_test_tfidf_token.append(contents_test_index_to_word_tfidf[key])
    contents_test_tfidf_token_list.append(contents_test_tfidf_token)

print('contents test filtering rank done')

''''''''''''''''''''' Make New Sentence by Using Tokenizer and Filtered TF-IDF '''''''''''''''''''''
contents_tokenizer = Tokenizer()
contents_tokenizer.fit_on_texts(encoder_input_test)

encoder_input_test = contents_tokenizer.texts_to_sequences(encoder_input_test)
contents_index_to_word = contents_tokenizer.index_word

new_encoder_input_test = []
for sentence_idx, sentence in enumerate(encoder_input_test):
    new_encoder_input_test_word = []
    new_encoder_input_test_sentence = ''

    for word_idx, word in enumerate(sentence):
        if contents_index_to_word[word] in contents_test_tfidf_token_list[sentence_idx] \
                and contents_index_to_word[word] not in new_encoder_input_test_word:
            new_encoder_input_test_word.append(contents_index_to_word[word])
            new_encoder_input_test_sentence += contents_index_to_word[word] + ' '
            print(contents_index_to_word[word], end=' ')

    new_encoder_input_test.append(new_encoder_input_test_sentence)
    print('\n')

print(new_encoder_input_test)

''''''''''''''''''''' ReTokenize with New Sentence '''''''''''''''''''''
contents_tokenizer = Tokenizer()
contents_tokenizer.fit_on_texts(new_encoder_input_train)
contents_tokenizer.fit_on_texts(new_encoder_input_test)

new_encoder_input_train = contents_tokenizer.texts_to_sequences(new_encoder_input_train)
new_encoder_input_test = contents_tokenizer.texts_to_sequences(new_encoder_input_test)
contents_vocab = len(contents_tokenizer.word_index) + 1

''''''''''''''''''''' Tokenize : Title '''''''''''''''''''''
title_tokenizer = Tokenizer()
title_tokenizer.fit_on_texts(decoder_input_train)
title_vocab = len(title_tokenizer.word_index) + 1

#
# total_cnt = len(title_tokenizer.vocabulary_)
#
# print('=========================================')
# print('Analyze Title Token')
# print('Total Word Num:', total_cnt)
# print('=========================================')
#
# ''''''''''''''''''''' Re-Tokenize by Checking Contents Word Frequency : Title '''''''''''''''''''''
# title_vocab = 10300
# title_tokenizer = TfidfVectorizer(max_features=title_vocab)
# title_tokenizer.fit(decoder_input_train + decoder_target_train)

''''''''''''''''''''' Text to Sequence : Title '''''''''''''''''''''
decoder_input_train = title_tokenizer.texts_to_sequences(decoder_input_train)
decoder_target_train = title_tokenizer.texts_to_sequences(decoder_target_train)
decoder_input_test = title_tokenizer.texts_to_sequences(decoder_input_test)
decoder_target_test = title_tokenizer.texts_to_sequences(decoder_target_test)

''''''''''''''''''''' Pad Sequences '''''''''''''''''''''
contents_pad_len = contents_tfidf_len
title_pad_len = 10

encoder_input_train = pad_sequences(new_encoder_input_train, maxlen=contents_pad_len)
decoder_input_train = pad_sequences(decoder_input_train, maxlen=title_pad_len)
decoder_target_train = pad_sequences(decoder_target_train, maxlen=title_pad_len)

encoder_input_test = pad_sequences(new_encoder_input_test, maxlen=contents_pad_len)
decoder_input_test = pad_sequences(decoder_input_test, maxlen=title_pad_len)
decoder_target_test = pad_sequences(decoder_target_test, maxlen=title_pad_len)

print(new_encoder_input_train)
print(new_encoder_input_test)

''''''''''''''''''''' Build Model '''''''''''''''''''''
embedding_dim_list = [32, 64, 128, 256]
hidden_size_list = [32, 64, 128, 256]


def model_encoder_1_decoder_1():
    ''''''''''''''''''''' Encoder : LSTM X 1 '''''''''''''''''''''
    encoder_inputs = Input(shape=(contents_pad_len,))
    enc_emb = Embedding(contents_vocab, embedding_dim)(encoder_inputs)

    encoder_lstm1 = LSTM(hidden_size, return_sequences=True, return_state=True, dropout=0.4)
    encoder_outputs, state_h, state_c = encoder_lstm1(enc_emb)

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

    history_e1d1 = model.fit(x=[encoder_input_train, decoder_input_train], y=decoder_target_train,
                             validation_data=([encoder_input_test, decoder_input_test], decoder_target_test),
                             batch_size=256, callbacks=[early_stopping_callback], epochs=1000)

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
                             batch_size=256, callbacks=[early_stopping_callback], epochs=1000)

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
                             batch_size=256, callbacks=[early_stopping_callback], epochs=1000)

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
                             batch_size=256, callbacks=[early_stopping_callback], epochs=1000)

    return history_e3d3


if __name__ == "__main__":
    info_list = []
    loss_e1d1_list = []
    loss_e1d3_list = []
    loss_e3d1_list = []
    loss_e3d3_list = []
    DROPOUT = 0.5

    plot_dir = 'D:/study/python/UROP/model/neutral/plot/plot_TFIDF_LSTM_dropout%d' % (DROPOUT * 100)
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

            plt.ylim([0, 4])
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
        loss_dataframe.to_csv('D:/study/python/UROP/model/neutral/loss/loss_TFIDF_LSTM_dropout%d.csv' % (DROPOUT * 100),
                              encoding='utf-8-sig', index=True)
