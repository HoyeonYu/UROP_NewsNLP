import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

csv_read_list = ['D:/study/python/UROP/preprocessing/preprocessed_dropMin_naverNews.csv',
                 # 'D:/study/python/UROP/preprocessing/preprocessed_ruliWeb.csv',
                 # 'D:/study/python/UROP/preprocessing/preprocessed_natePann.csv',
                 # 'D:/study/python/UROP/preprocessing/preprocessed_kidsBook.csv',
                 # 'D:/study/python/UROP/preprocessing/preprocessed_kidsSong.csv'
                 ]

csv_save_list = ['D:/study/python/UROP/analyzing/analyzed_naverNews.csv',
                 # 'D:/study/python/UROP/analyzing/analyzed_ruliWeb.csv',
                 # 'D:/study/python/UROP/analyzing/analyzed_natePann.csv',
                 # 'D:/study/python/UROP/analyzing/analyzed_kidsBook.csv',
                 # 'D:/study/python/UROP/analyzing/analyzed_kidsSong.csv'
                 ]

csv_save_concat_list = ['D:/study/python/UROP/analyzing/concatenated_neutral.csv',
                        'D:/study/python/UROP/analyzing/concatenated_clicked.csv',
                        'D:/study/python/UROP/analyzing/concatenated_kids.csv']

title_max_len = [10, 12, 12, 20, 20]
contents_max_len = [300, 1000, 1000, 700, 700]

for idx in range(len(csv_read_list)):
    print('======================================================')
    print('Read -> ', csv_read_list[idx], '\n')
    data = pd.read_csv(csv_read_list[idx], nrows=100000)
    data = data[['title', 'contents']]

    title_len = [len(s.split()) for s in data['title']]
    contents_len = [len(s.split()) for s in data['contents']]

    print('Title Min Length: {}'.format(np.min(title_len)))
    print('Title Max Length: {}'.format(np.max(title_len)))
    print('Title Avg Length : {}'.format(np.mean(title_len)))
    print('Contents Min Length : {}'.format(np.min(contents_len)))
    print('Contents Max Length: {}'.format(np.max(contents_len)))
    print('Contents Avg Length: {}'.format(np.mean(contents_len)))

    plt.subplot(1, 2, 1)
    plt.boxplot(title_len)
    plt.title('Title')

    plt.subplot(1, 2, 2)
    plt.boxplot(contents_len)
    plt.title('Contents')
    plt.tight_layout()
    plt.show()

    print('Before Drop Big Size, Length: ', (len(data)))
    data = data[data['title'].apply(lambda x: len(x.split()) <= title_max_len[idx])]
    data = data[data['contents'].apply(lambda x: len(x.split()) <= contents_max_len[idx])]
    print('After Drop Big Size, Length: ', (len(data)))

    data['decoder_input'] = data['title'].apply(lambda x: 'sostoken ' + x)
    data['decoder_target'] = data['title'].apply(lambda x: x + ' eostoken')

    data.to_csv(csv_save_list[idx], encoding='utf-8-sig', index=False)
    if idx == 0:
        data.to_csv(csv_save_concat_list[0], encoding='utf-8-sig', index=True)

    print('======================================================')

# for idx in range(2):
#     print(csv_save_list[(idx * 2) + 1])
#     print(csv_save_list[(idx * 2) + 2])
#     dataClick1 = pd.read_csv(csv_save_list[(idx * 2) + 1], nrows=100000)
#     dataClick2 = pd.read_csv(csv_save_list[(idx * 2) + 2], nrows=100000)
#
#     dataClick1 = pd.concat([dataClick1, dataClick2])
#     dataClick1.to_csv(csv_save_concat_list[idx + 1], encoding='utf-8-sig', index=True)
