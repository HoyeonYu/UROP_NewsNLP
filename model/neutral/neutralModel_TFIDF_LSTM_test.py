import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from keras_preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding

''''''''''''''''''''' Read Preprocessed File '''''''''''''''''''''
csv_read_neutral = 'D:/study/python/UROP/preprocessing/preprocessed_TFIDF_naverNews.csv'
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

''''''''''''''''''''' ReTokenize with New Sentence : Contents '''''''''''''''''''''
contents_tokenizer = Tokenizer()
contents_tokenizer.fit_on_texts(encoder_input)
contents_vocab = len(contents_tokenizer.index_word) + 1
print('contents_vocab: ', contents_vocab)

contents_vocab = 20000
contents_tokenizer = Tokenizer(num_words=contents_vocab)
contents_tokenizer.fit_on_texts(encoder_input)

encoder_input_train = contents_tokenizer.texts_to_sequences(encoder_input_train)
encoder_input_test = contents_tokenizer.texts_to_sequences(encoder_input_test)

''''''''''''''''''''' Tokenize : Title '''''''''''''''''''''
title_tokenizer = Tokenizer()
title_tokenizer.fit_on_texts(decoder_input)
title_tokenizer.fit_on_texts(decoder_target)
title_vocab = len(title_tokenizer.index_word) + 1
print('title_vocab: ', title_vocab)

title_vocab = 3000
title_tokenizer = Tokenizer(num_words=title_vocab)
title_tokenizer.fit_on_texts(decoder_input)
title_tokenizer.fit_on_texts(decoder_target)

''''''''''''''''''''' Text to Sequence : Title '''''''''''''''''''''
decoder_input_train = title_tokenizer.texts_to_sequences(decoder_input_train)
decoder_input_test = title_tokenizer.texts_to_sequences(decoder_input_test)
decoder_target_train = title_tokenizer.texts_to_sequences(decoder_target_train)
decoder_target_test = title_tokenizer.texts_to_sequences(decoder_target_test)

''''''''''''''''''''' Analyze Length '''''''''''''''''''''
title_len = [len(tokens) for tokens in decoder_input_train]
contents_len = [len(tokens) for tokens in encoder_input_train]

print('Title Min Length: {}'.format(np.min(title_len)))
print('Title Max Length: {}'.format(np.max(title_len)))
print('Title Avg Length : {}'.format(np.mean(title_len)))
print('Contents Min Length : {}'.format(np.min(contents_len)))
print('Contents Max Length: {}'.format(np.max(contents_len)))
print('Contents Avg Length: {}'.format(np.mean(contents_len)))

''''''''''''''''''''' Drop If (Contents Token Size < 10), (Title Token Size < 4) '''''''''''''''''''''
drop_idx_train = [index for index, sentence in enumerate(encoder_input_train) if len(sentence) < 10]
drop_idx_train += [index for index, sentence in enumerate(decoder_input_train) if len(sentence) < 4]
drop_idx_train += [index for index, sentence in enumerate(decoder_target_train) if len(sentence) < 4]

drop_idx_test = [index for index, sentence in enumerate(encoder_input_test) if len(sentence) < 10]
drop_idx_test += [index for index, sentence in enumerate(decoder_input_test) if len(sentence) < 4]
drop_idx_test += [index for index, sentence in enumerate(decoder_target_test) if len(sentence) < 4]

encoder_input_train = np.delete(encoder_input_train, drop_idx_train, axis=0)
decoder_input_train = np.delete(decoder_input_train, drop_idx_train, axis=0)
decoder_target_train = np.delete(decoder_target_train, drop_idx_train, axis=0)

encoder_input_test = np.delete(encoder_input_test, drop_idx_test, axis=0)
decoder_input_test = np.delete(decoder_input_test, drop_idx_test, axis=0)
decoder_target_test = np.delete(decoder_target_test, drop_idx_test, axis=0)

''''''''''''''''''''' Pad Sequences '''''''''''''''''''''
contents_pad_len = 20
title_pad_len = 8

encoder_input_train = pad_sequences(encoder_input_train, maxlen=contents_pad_len, padding='post')
decoder_input_train = pad_sequences(decoder_input_train, maxlen=title_pad_len, padding='post')
decoder_target_train = pad_sequences(decoder_target_train, maxlen=title_pad_len, padding='post')

encoder_input_test = pad_sequences(encoder_input_test, maxlen=contents_pad_len, padding='post')
decoder_input_test = pad_sequences(decoder_input_test, maxlen=title_pad_len, padding='post')
decoder_target_test = pad_sequences(decoder_target_test, maxlen=title_pad_len, padding='post')

''''''''''''''''''''' Reverse Sequences '''''''''''''''''''''
encoder_input_train = encoder_input_train[:, ::-1]
encoder_input_test = encoder_input_test[:, ::-1]

print('=========================================')
print('Final Data Summary\n')
print('Train Input Size:', len(encoder_input_train))
print('Train Output Size:', len(decoder_input_train))
print('Test Input Size:', len(encoder_input_test))
print('Test Output Size:', len(decoder_input_test))
print('=========================================')

''''''''''''''''''''' Build Model '''''''''''''''''''''
embedding_dim = 16
hidden_size = 64
DROPOUT = 0.4

''''''''''''''''''''' Encoder : LSTM X 1 '''''''''''''''''''''
encoder_inputs = Input(shape=(contents_pad_len,))
enc_emb = Embedding(contents_vocab, embedding_dim, trainable=True)(encoder_inputs)
encoder_lstm = LSTM(hidden_size, return_sequences=True, return_state=True, dropout=DROPOUT)
encoder_outputs, state_h, state_c = encoder_lstm(enc_emb)

''''''''''''''''''''' Decoder : LSTM X 1 '''''''''''''''''''''
decoder_inputs = Input(shape=(None,))
dec_emb_layer = Embedding(title_vocab, embedding_dim, trainable=True)
dec_emb = dec_emb_layer(decoder_inputs)
decoder_lstm = LSTM(hidden_size, return_sequences=True, return_state=True, dropout=DROPOUT)
decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=[state_h, state_c])
decoder_dense = Dense(title_vocab, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

''''''''''''''''''''' Encoder + Decoder Model '''''''''''''''''''''
model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_outputs)
model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10)
history_e1d1 = model.fit(x=[encoder_input_train, decoder_input_train], y=decoder_target_train,
                         validation_data=([encoder_input_test, decoder_input_test], decoder_target_test),
                         batch_size=64, callbacks=[early_stopping_callback], epochs=300)

model.summary()

plt.figure()
plt.plot(history_e1d1.history['loss'], label='train')
plt.plot(history_e1d1.history['val_loss'], label='test')
plt.ylim([3, 6])
plt.legend()
plt.title('Loss Graph (Embedding Dim: %d, Hidden Size: %d)' % (embedding_dim, hidden_size))
plt.show()
# plt.savefig('emb%d_hid%d.png' % (embedding_dim, hidden_size))

''''''''''''''''''''' Test Model '''''''''''''''''''''
contents_index_to_word = contents_tokenizer.index_word
title_word_to_index = title_tokenizer.word_index
title_index_to_word = title_tokenizer.index_word

encoder_model = Model(inputs=encoder_inputs, outputs=[encoder_outputs, state_h, state_c])

decoder_state_input_h = Input(shape=(hidden_size,))
decoder_state_input_c = Input(shape=(hidden_size,))
decoder_states_inputs = Input(shape=(contents_pad_len, hidden_size))

dec_emb2 = dec_emb_layer(decoder_inputs)
decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2, initial_state=[decoder_state_input_h,
                                                                             decoder_state_input_c])
decoder_outputs2 = decoder_dense(decoder_outputs2)

decoder_model = Model([decoder_inputs] + [decoder_states_inputs, decoder_state_input_h, decoder_state_input_c],
                      [decoder_outputs2] + [state_h2, state_c2])


def decode_sequence(input_seq):
    e_out, e_h, e_c = encoder_model.predict(input_seq)

    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = title_word_to_index['sostoken']
    target_seq_idx = [target_seq[0, 0]]

    decoded_sentence = ' '

    while True:
        output_tokens, h, c = decoder_model.predict([target_seq] + [e_out, e_h, e_c])
        sampled_token_index = np.argmax(output_tokens[0, -1, :])

        if sampled_token_index != 0:
            sampled_token_word = title_index_to_word[sampled_token_index]

            if sampled_token_word == 'eostoken' or len(target_seq_idx) > title_pad_len - 1:
                break

            elif sampled_token_index not in target_seq_idx:
                decoded_sentence += sampled_token_word + ' '

            target_seq = np.zeros((1, 1))
            target_seq[0, 0] = sampled_token_index

        target_seq_idx.append(sampled_token_index)
        e_h, e_c = h, c

    return decoded_sentence


def seq2contents(input_seq):
    temp = ''
    for i in input_seq:
        if i == 0:
            continue
        temp += contents_index_to_word[i] + ' '
    return temp


def seq2title(input_seq):
    temp = ''
    for i in input_seq:
        if i == 0:
            continue
        temp = temp + title_index_to_word[i] + ' '

    return temp


for i in range(30):
    print('TF-IDF contents : ', seq2contents(encoder_input_test[i]))
    print('decoder input:', decoder_input_test[i])
    print('decoder target:', decoder_target_test[i])
    print('decoder input index to word:', seq2title(decoder_input_test[i]))
    print('predict title :', decode_sequence(encoder_input_test[i].reshape(1, contents_pad_len)))
    print("\n")
