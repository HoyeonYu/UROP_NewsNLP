import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
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
contents_tokenizer = TfidfVectorizer()
contents_tokenizer.fit(encoder_input_train)
total_cnt = len(contents_tokenizer.vocabulary_)

print('=========================================')
print('Analyze Contents Token')
print('Total Word Num:', total_cnt)
print('=========================================')

''''''''''''''''''''' Re-Tokenize by Checking Contents Word Frequency : Contents '''''''''''''''''''''
# Tokenize Contents
contents_vocab = 22000
contents_tokenizer = TfidfVectorizer(max_features=contents_vocab)
contents_tokenizer.fit(encoder_input_train)

''''''''''''''''''''' Text to Sequence : Contents '''''''''''''''''''''
encoder_input_train = contents_tokenizer.transform(encoder_input_train).toarray()
encoder_input_test = contents_tokenizer.transform(encoder_input_test).toarray()

''''''''''''''''''''' Tokenize : Title '''''''''''''''''''''
title_tokenizer = TfidfVectorizer()
title_tokenizer.fit(decoder_input_train)

total_cnt = len(title_tokenizer.vocabulary_)

print('=========================================')
print('Analyze Title Token')
print('Total Word Num:', total_cnt)
print('=========================================')

''''''''''''''''''''' Re-Tokenize by Checking Contents Word Frequency : Title '''''''''''''''''''''
title_vocab = 10300
title_tokenizer = TfidfVectorizer(max_features=title_vocab)
title_tokenizer.fit(decoder_input_train + decoder_target_train)

''''''''''''''''''''' Text to Sequence : Title '''''''''''''''''''''
decoder_input_train = title_tokenizer.transform(decoder_input_train).toarray()
decoder_target_train = title_tokenizer.transform(decoder_target_train).toarray()
decoder_input_test = title_tokenizer.transform(decoder_input_test).toarray()
decoder_target_test = title_tokenizer.transform(decoder_target_test).toarray()

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

# ''''''''''''''''''''' Build Model '''''''''''''''''''''
# embedding_dim = 64
# hidden_size = 64
#
# ''''''''''''''''''''' Encoder : LSTM X 1 '''''''''''''''''''''
# encoder_inputs = Input(shape=(contents_pad_len,))
# enc_emb = Embedding(contents_vocab, embedding_dim)(encoder_inputs)
# encoder_lstm = LSTM(hidden_size, return_state=True, dropout=0.4)
# encoder_outputs, state_h, state_c = encoder_lstm(enc_emb)
# encoder_states = [state_h, state_c]
#
# ''''''''''''''''''''' Decoder : LSTM X 1 '''''''''''''''''''''
# decoder_inputs = Input(shape=(None,))
# dec_emb_layer = Embedding(title_vocab, embedding_dim)
# dec_emb = dec_emb_layer(decoder_inputs)
# decoder_lstm = LSTM(hidden_size, return_sequences=True, return_state=True, dropout=0.4)
# decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=encoder_states)
#
# decoder_dense = Dense(title_vocab, activation='softmax')
# decoder_outputs = decoder_dense(decoder_outputs)
#
# ''''''''''''''''''''' Encoder + Decoder Model '''''''''''''''''''''
# model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
# model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')
# early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10)
# model.fit(x=[encoder_input_train, decoder_input_train], y=decoder_target_train,
#           validation_data=([encoder_input_test, decoder_input_test], decoder_target_test),
#           batch_size=256, callbacks=[early_stopping_callback], epochs=3)
# model.summary()
#
# ''''''''''''''''''''' Test Model '''''''''''''''''''''
contents_word_to_index = contents_tokenizer.vocabulary_
contents_index_to_word = dict(map(reversed, contents_word_to_index.items()))
title_word_to_index = title_tokenizer.vocabulary_
title_index_to_word = dict(map(reversed, title_word_to_index.items()))
# print(title_word_to_index)
# print(title_index_to_word)
#
# encoder_model = Model(inputs=encoder_inputs, outputs=[encoder_outputs, state_h, state_c])
#
# decoder_state_input_h = Input(shape=(hidden_size,))
# decoder_state_input_c = Input(shape=(hidden_size,))
# decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
#
# dec_emb2 = dec_emb_layer(decoder_inputs)
# decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2, initial_state=decoder_states_inputs)
# decoder_states = [state_h2, state_c2]
#
# decoder_outputs2 = decoder_dense(decoder_outputs2)
#
# decoder_model = Model([decoder_inputs] + decoder_states_inputs,
#                       [decoder_outputs2] + decoder_states)


# def decode_sequence(input_seq):
#     e_out, e_h, e_c = encoder_model.predict(input_seq)
#     states_value = [e_h, e_c]
#
#     target_seq = np.zeros((1, 1))
#     target_seq[0, 0] = title_word_to_index['sostoken']
#
#     decoded_sentence = ''
#
#     while True:
#         output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
#         sampled_token_index = np.argmax(output_tokens[0, -1, :])
#         sampled_token = title_index_to_word[sampled_token_index + 1]
#
#         if sampled_token == 'eostoken' or len(decoded_sentence.split(' ')) > title_pad_len - 1:
#             break
#
#         else:
#             decoded_sentence += sampled_token + ' '
#
#         target_seq = np.zeros((1, 1))
#         target_seq[0, 0] = sampled_token_index
#
#         states_value = [h, c]
#
#     return decoded_sentence


def seq2text(input_seq):
    temp = ''
    for i in input_seq:
        if i != 0:
            temp = temp + contents_index_to_word[i] + ' '
    return temp


def seq2summary(input_seq):
    temp = ''
    for i in input_seq:
        if (i != 0 and i != title_word_to_index['sostoken']) and i != title_word_to_index['eostoken']:
            temp = temp + title_index_to_word[i] + ' '
    return temp


for i in range(500, 600):
    print("원문 : ", seq2text(encoder_input_test[i]))
    print("실제 요약문 :", seq2summary(decoder_input_test[i]))
    # print("예측 요약문 :", decode_sequence(encoder_input_test[i].reshape(1, contents_pad_len)))
    print("\n")
