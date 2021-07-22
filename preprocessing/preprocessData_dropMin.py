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

    titleDF = []
    contentsDF = []

    for sentence in data['title']:
        sentence = sentence.replace('스브스뉴스', ' ')
        sentence = sentence.replace('스브스', ' ')
        sentence = sentence.replace('예능맛집', ' ')
        sentence = sentence.replace('인턴기', ' ')
        sentence = sentence.replace('자막뉴스', ' ')
        sentence = sentence.replace('아침신문', ' ')
        sentence = sentence.replace('소름돋', ' ')
        sentence = sentence.replace('초간단', ' ')
        sentence = sentence.replace('정치성향테스트', ' ')
        sentence = sentence.replace('최근', ' ')
        sentence = sentence.replace('지금', ' ')
        sentence = sentence.replace('거리두', '거리두기')
        titleDF.append(sentence)

    for sentence in data['contents']:
        sentence = sentence.replace('스브스뉴스', ' ')
        sentence = sentence.replace('스브스', ' ')
        sentence = sentence.replace('예능맛집', ' ')
        sentence = sentence.replace('인턴기', ' ')
        sentence = sentence.replace('자막뉴스', ' ')
        sentence = sentence.replace('아침신문', ' ')
        sentence = sentence.replace('소름돋', ' ')
        sentence = sentence.replace('초간단', ' ')
        sentence = sentence.replace('정치성향테스트', ' ')
        sentence = sentence.replace('최근', ' ')
        sentence = sentence.replace('지금', ' ')
        sentence = sentence.replace('거리두', '거리두기')
        contentsDF.append(sentence)

    dataFrame = []
    dataFrame = pd.DataFrame(dataFrame, columns=['title', 'contents'])
    dataFrame['title'] = titleDF
    dataFrame['contents'] = contentsDF
    print('After Preprocessing Function,\tLength: ', len(dataFrame))

    dataFrame = dataFrame[dataFrame['title'].apply(lambda x: len(x.split()) > 2)]
    dataFrame = dataFrame[dataFrame['contents'].apply(lambda x: len(x.split()) > 10)]
    print('After Drop Title < 4, Contents < 10, Data Length: ', len(dataFrame))
    print('\nFinal Len:\t', len(dataFrame))

    dataFrame.to_csv(csv_save_list[idx], encoding='utf-8-sig', index=True)

    print('\nDone -> ', csv_save_list[idx])
    print('======================================================\n')
