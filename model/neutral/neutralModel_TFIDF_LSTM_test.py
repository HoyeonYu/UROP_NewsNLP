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

encoder_input_train_original = encoder_input[:-test_val]
encoder_input_train = encoder_input[:-test_val]
decoder_input_train = decoder_input[:-test_val]
decoder_target_train = decoder_target[:-test_val]

encoder_input_test_original = encoder_input[-test_val:]
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
    if sentence_tfidf_idx % 500 == 0:
        print(sentence_tfidf_idx, '/', len(encoder_input_train_tfidf))
    for word_idx, word_tfidf in enumerate(sentence_tfidf):
        if word_tfidf > 0:
            encoder_input_train_tfidf_map[word_idx] = word_tfidf
    encoder_input_train_tfidf_list.append(encoder_input_train_tfidf_map)

# print(encoder_input_train_tfidf_list)
print('contents train filtering tfidf > 0 done')

''''''''''''''''''''' Sort TF-IDF Descending Order, Filtering Rank < 40 : Contents (Train) '''''''''''''''''''''
contents_tfidf_len = 40

contents_train_tfidf_token_list = []
for encoder_input_train_tfidf_map_idx, encoder_input_train_tfidf_map in enumerate(encoder_input_train_tfidf_list):
    contents_train_tfidf_token = []
    encoder_input_train_tfidf_map = sorted(encoder_input_train_tfidf_map.items(), key=lambda x: x[1], reverse=True)
    if encoder_input_train_tfidf_map_idx % 500 == 0:
        print(encoder_input_train_tfidf_map_idx, '/', len(encoder_input_train_tfidf_list))
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
            # print(contents_index_to_word[word], end=' ')

    new_encoder_input_train.append(new_encoder_input_train_sentence)
    # print('\n')
# print(new_encoder_input_train)
print('new train sentence done')

''''''''''''''''''''' TF-IDF : Contents (Test) '''''''''''''''''''''
encoder_input_test_tfidf = contents_tfidf_tokenizer.transform(encoder_input_test).toarray()

contents_test_word_to_index_tfidf = contents_tfidf_tokenizer.vocabulary_
contents_test_index_to_word_tfidf = dict(map(reversed, contents_test_word_to_index_tfidf.items()))

''''''''''''''''''''' Filtering TF-IDF > 0 and Save with Map Structure : Contents (Test) '''''''''''''''''''''
encoder_input_test_tfidf_list = []
for sentence_tfidf_idx, sentence_tfidf in enumerate(encoder_input_test_tfidf):
    encoder_input_test_tfidf_map = {}
    if sentence_tfidf_idx % 500 == 0:
        print(sentence_tfidf_idx, '/', len(encoder_input_test_tfidf))
    for word_idx, word_tfidf in enumerate(sentence_tfidf):
        if word_tfidf > 0:
            encoder_input_test_tfidf_map[word_idx] = word_tfidf
    encoder_input_test_tfidf_list.append(encoder_input_test_tfidf_map)

# print(encoder_input_test_tfidf_list)
print('contents test filtering tfidf > 0 done')

''''''''''''''''''''' Sort TF-IDF Descending Order, Filtering Rank < 40 : Contents (Test) '''''''''''''''''''''
contents_tfidf_len = 40

contents_test_tfidf_token_list = []
for encoder_input_test_tfidf_map_idx, encoder_input_test_tfidf_map in enumerate(encoder_input_test_tfidf_list):
    contents_test_tfidf_token = []
    encoder_input_test_tfidf_map = sorted(encoder_input_test_tfidf_map.items(), key=lambda x: x[1], reverse=True)
    if encoder_input_test_tfidf_map_idx % 500 == 0:
        print(encoder_input_test_tfidf_map_idx, '/', len(encoder_input_test_tfidf_list))
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
            # print(contents_index_to_word[word], end=' ')

    new_encoder_input_test.append(new_encoder_input_test_sentence)
    # print('\n')

# print(new_encoder_input_test)
print('new test sentence done')

''''''''''''''''''''' ReTokenize with New Sentence : Contents '''''''''''''''''''''
contents_vocab = 22000
contents_tokenizer = Tokenizer(num_words=contents_vocab)
contents_tokenizer.fit_on_texts(new_encoder_input_train)
# contents_tokenizer.fit_on_texts(new_encoder_input_test)
contents_vocab = len(contents_tokenizer.word_index) + 1

new_encoder_input_train = contents_tokenizer.texts_to_sequences(new_encoder_input_train)
new_encoder_input_test = contents_tokenizer.texts_to_sequences(new_encoder_input_test)

print('contents_vocab: ', contents_vocab)

''''''''''''''''''''' Tokenize : Title '''''''''''''''''''''
title_vocab = 3300
title_tokenizer = Tokenizer(num_words=title_vocab)
title_tokenizer.fit_on_texts(decoder_input_train)
title_tokenizer.fit_on_texts(decoder_target_train)  ################# Changed ####################
# title_tokenizer.fit_on_texts(decoder_input_test)  ################# Changed ####################
# title_tokenizer.fit_on_texts(decoder_target_test)  ################# Changed ####################
title_vocab = len(title_tokenizer.word_index) + 1

print("=================================")
print('contents_vocab: ', contents_vocab)
print('title_vocab: ', title_vocab)
print("=================================")

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

# print(new_encoder_input_train)
# print(new_encoder_input_test)

''''''''''''''''''''' Build Model '''''''''''''''''''''
embedding_dim = 128
hidden_size = 32

''''''''''''''''''''' Encoder : LSTM X 1 '''''''''''''''''''''
encoder_inputs = Input(shape=(contents_pad_len,))
enc_emb = Embedding(contents_vocab, embedding_dim)(encoder_inputs)

encoder_lstm1 = LSTM(embedding_dim, return_sequences=True, return_state=True, dropout=0.4)
encoder_outputs, state_h, state_c = encoder_lstm1(enc_emb)

decoder_inputs = Input(shape=(None,))

''''''''''''''''''''' Decoder : LSTM X 1 '''''''''''''''''''''
dec_emb_layer = Embedding(title_vocab, embedding_dim)
dec_emb = dec_emb_layer(decoder_inputs)

decoder_lstm = LSTM(embedding_dim, return_sequences=True, return_state=True, dropout=0.4)
decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=[state_h, state_c])

decoder_softmax_layer = Dense(title_vocab, activation='softmax')
decoder_softmax_outputs = decoder_softmax_layer(decoder_outputs)

''''''''''''''''''''' Encoder + Decoder Model '''''''''''''''''''''
model = Model([encoder_inputs, decoder_inputs], decoder_softmax_outputs)
model.summary()

model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')

early_stopping_callback = EarlyStopping(monitor='val_loss', patience=20)

history_e1d1 = model.fit(x=[encoder_input_train, decoder_input_train], y=decoder_target_train,
                         validation_data=([encoder_input_test, decoder_input_test], decoder_target_test),
                         batch_size=256, callbacks=[early_stopping_callback], epochs=1000)

plt.figure()
plt.plot(history_e1d1.history['val_loss'], label='test_e1d1')
plt.legend()
plt.title('Loss Graph (Embedding Dim: %d, Hidden Size: %d)' % (embedding_dim, hidden_size))
plt.savefig('emb%d_hid%d.png' % (embedding_dim, hidden_size))

''''''''''''''''''''' Test Model '''''''''''''''''''''
title_word_to_index = title_tokenizer.word_index
title_index_to_word = title_tokenizer.index_word
# print(title_index_to_word)

encoder_model = Model(inputs=encoder_inputs, outputs=[encoder_outputs, state_h, state_c])

decoder_state_input_h = Input(shape=(embedding_dim,))
decoder_state_input_c = Input(shape=(embedding_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

dec_emb2 = dec_emb_layer(decoder_inputs)
decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2, initial_state=decoder_states_inputs)
decoder_states = [state_h2, state_c2]

decoder_outputs2 = decoder_softmax_layer(decoder_outputs2)

decoder_model = Model([decoder_inputs] + decoder_states_inputs,
                      [decoder_outputs2] + decoder_states)


def decode_sequence(input_seq):
    e_out, e_h, e_c = encoder_model.predict(input_seq)
    states_value = [e_h, e_c]

    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = title_word_to_index['sostoken']

    decoded_sentence = ' '

    while True:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_token = title_index_to_word[sampled_token_index + 1]

        if sampled_token == 'eostoken' or len(decoded_sentence.split(' ')) > title_pad_len - 1:
            break

        else:
            decoded_sentence += sampled_token + ' '

        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index

        states_value = [h, c]

    return decoded_sentence


def seq2summary(input_seq):
    temp = ''
    for i in input_seq:
        if (i != 0 and i != title_word_to_index['sostoken']) and i != title_word_to_index['eostoken']:
            temp = temp + title_index_to_word[i] + ' '
    return temp


for i in range(5):
    print("뉴스 전문 : ", encoder_input_test_original[i])
    print("실제 뉴스 제목 :", seq2summary(decoder_input_test[i]))
    print("예측 뉴스 제목 :", decode_sequence(encoder_input_test[i].reshape(1, contents_pad_len)))
    # print("예측 뉴스 제목 :", decode_sequence(encoder_input_test[i]))
    print("\n")
