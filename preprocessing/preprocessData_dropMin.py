import numpy as np
import pandas as pd

csv_read_list = ['D:/study/python/UROP/preprocessing/preprocessed_naverNews.csv',
                 # 'D:/study/python/UROP/crawling/crawled_ruliWeb.csv',
                 # 'D:/study/python/UROP/crawling/crawled_natePann.csv',
                 # 'D:/study/python/UROP/crawling/crawled_kidsBook.csv',
                 # 'D:/study/python/UROP/crawling/crawled_kidsSong.csv'
                 ]

csv_save_list = ['D:/study/python/UROP/preprocessing/preprocessed_dropMin_naverNews.csv',
                 # 'D:/study/python/UROP/preprocessing/preprocessed_dropMin__ruliWeb.csv',
                 # 'D:/study/python/UROP/preprocessing/preprocessed_dropMin__natePann.csv',
                 # 'D:/study/python/UROP/preprocessing/preprocessed_dropMin__kidsBook.csv',
                 # 'D:/study/python/UROP/preprocessing/preprocessed_dropMin__kidsSong.csv'
                 ]


for idx in range(len(csv_read_list)):
    print('======================================================')
    print('Read -> ', csv_read_list[idx], '\n')
    data = pd.read_csv(csv_read_list[idx], nrows=100000)
    print('Initial,\tLength:', (len(data)))

    data = data[['title', 'contents']]
    data.dropna(axis=0, inplace=True)
    print('After Drop Null,\tLength: ', len(data))

    dataFrame = []
    dataFrame = pd.DataFrame(dataFrame, columns=['title', 'contents'])
    dataFrame['title'] = data['title'].replace('스브스뉴스', ' ')
    dataFrame['title'] = data['title'].replace('자막뉴스', ' ')
    dataFrame['title'] = data['title'].replace('아침신문', ' ')
    dataFrame['title'] = data['title'].replace('최근', ' ')
    dataFrame['title'] = data['title'].replace('거리두', '거리두기')

    dataFrame['contents'] = data['contents'].replace('스브스뉴스', ' ')
    dataFrame['contents'] = data['contents'].replace('자막뉴스', ' ')
    dataFrame['contents'] = data['contents'].replace('아침신문', ' ')
    dataFrame['contents'] = data['contents'].replace('최근', ' ')
    dataFrame['contents'] = data['contents'].replace('거리두', '거리두기')
    print('After Preprocessing Function,\tLength: ', len(dataFrame))

    dataFrame = dataFrame[dataFrame['title'].apply(lambda x: len(x.split()) > 2)]
    dataFrame = dataFrame[dataFrame['contents'].apply(lambda x: len(x.split()) > 5)]
    print('After Drop Less than 10, Data Length: ', len(dataFrame))
    print('\nFinal Len:\t', len(dataFrame))

    dataFrame.to_csv(csv_save_list[idx], encoding='utf-8-sig', index=True)

    print('\nDone -> ', csv_save_list[idx])
    print('======================================================\n')
