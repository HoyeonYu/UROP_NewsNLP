from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

''''''''''''''''''''' Read Preprocessed File '''''''''''''''''''''
csv_read_neutral = 'D:/study/python/UROP/analyzing/analyzed_naverNews.csv'
data = pd.read_csv(csv_read_neutral)
contents_df = data['contents']

''''''''''''''''''''' Tokenize : Contents '''''''''''''''''''''
contents_tfidf_tokenizer = TfidfVectorizer()
contents_tfidf_tokenizer.fit(contents_df)
total_cnt = len(contents_tfidf_tokenizer.vocabulary_)

''''''''''''''''''''' TF-IDF : Contents '''''''''''''''''''''
encoder_input_tfidf = contents_tfidf_tokenizer.transform(contents_df).toarray()

contents_word_to_index_tfidf = contents_tfidf_tokenizer.vocabulary_
contents_index_to_word_tfidf = dict(map(reversed, contents_word_to_index_tfidf.items()))

''''''''''''''''''''' Filtering TF-IDF > 0 and Save as Map '''''''''''''''''''''
encoder_input_tfidf_list = []
for sentence_tfidf_idx, sentence_tfidf in enumerate(encoder_input_tfidf):
    encoder_input_tfidf_map = {}
    for word_idx, word_tfidf in enumerate(sentence_tfidf):
        if word_tfidf > 0:
            encoder_input_tfidf_map[word_idx] = word_tfidf
    encoder_input_tfidf_list.append(encoder_input_tfidf_map)

# print(encoder_input_tfidf_list)
print('contents filtering tfidf > 0 done')

''''''''''''''''''''' Sort TF-IDF Descending Order, Filtering Rank < 20 '''''''''''''''''''''
contents_tfidf_len = 20
contents_tfidf_token_list = []

for encoder_input_tfidf_map_idx, encoder_input_tfidf_map in enumerate(encoder_input_tfidf_list):
    contents_tfidf_sentence = ''
    encoder_input_tfidf_map = sorted(encoder_input_tfidf_map.items(), key=lambda x: x[1], reverse=True)

    for key, value in encoder_input_tfidf_map:
        if len(contents_tfidf_sentence.split(' ')) >= contents_tfidf_len:
            break
        contents_tfidf_sentence += contents_index_to_word_tfidf[key] + ' '
    contents_tfidf_token_list.append(contents_tfidf_sentence)

print('contents train filtering rank done')

dataFrame = []
dataFrame = pd.DataFrame(dataFrame, columns=['title', 'contents'])
dataFrame['title'] = data['title']
dataFrame['contents'] = contents_tfidf_token_list

dataFrame.to_csv('preprocessed_TFIDF_naverNews.csv', encoding='utf-8-sig', index=True)


# csv_read_before = 'D:/study/python/UROP/analyzing/analyzed_naverNews.csv'
# before_data = pd.read_csv(csv_read_before)
# csv_read_after = 'D:/study/python/UROP/preprocessing/preprocessed_TFIDF_naverNews.csv'
# after_data = pd.read_csv(csv_read_after)
#
# for i in range (5):
#     print('Title:', before_data['title'][i])
#     print('Before TF-IDF:', before_data['contents'][i])
#     print('After TF-IDF:', after_data['contents'][i])
#     print()
