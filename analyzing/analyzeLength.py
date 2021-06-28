import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

csv_read_list = ['D:/study/python/UROP/preprocessing/preprocessed_naverNews.csv',
                 'D:/study/python/UROP/preprocessing/preprocessed_ruliWeb.csv',
                 'D:/study/python/UROP/preprocessing/preprocessed_natePann.csv',
                 'D:/study/python/UROP/preprocessing/preprocessed_kidsBook.csv',
                 'D:/study/python/UROP/preprocessing/preprocessed_kidsSong.csv']

csv_save_list = ['D:/study/python/UROP/analyzing/analyzed_naverNews.csv',
                 'D:/study/python/UROP/analyzing/analyzed_ruliWeb.csv',
                 'D:/study/python/UROP/analyzing/analyzed_natePann.csv',
                 'D:/study/python/UROP/analyzing/analyzed_kidsBook.csv',
                 'D:/study/python/UROP/analyzing/analyzed_kidsSong.csv']

title_max_len = [4, 10, 12, 20, 5]
contents_max_len = [1500, 1000, 1000, 700, 180]

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
    # plt.show()

    plt.title('Title')
    plt.hist(title_len, bins=40)
    plt.xlabel('length of samples')
    plt.ylabel('number of samples')
    # plt.show()

    plt.title('Contents')
    plt.hist(contents_len, bins=40)
    plt.xlabel('length of samples')
    plt.ylabel('number of samples')
    # plt.show()
    print('======================================================')

    data = data[data['title'].apply(lambda x: len(x.split()) <= title_max_len[idx])]
    data = data[data['contents'].apply(lambda x: len(x.split()) <= contents_max_len[idx])]

    data['decoder_input'] = data['title'].apply(lambda x: 'sostoken ' + x)
    data['decoder_target'] = data['title'].apply(lambda x: x + ' eostoken')

    data.to_csv(csv_save_list[idx], encoding='utf-8-sig', index=True)
