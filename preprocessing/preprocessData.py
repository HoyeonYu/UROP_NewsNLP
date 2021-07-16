import numpy as np
import pandas as pd
import re
from konlpy.tag import Hannanum


def preprocess_sentence(sentence, remove_stopwords=True):
    stop_words = '단독 동영상 연합뉴스 노컷뉴스 구독 클릭 무단전재 재배포 기자 관련기사 저작권자 제보 무단 전재 금지 중앙일보 ' \
                 '네이버에서 뉴스 채널 구독하기 깔끔하게 훑어주는 세상의 이슈 와이퍼 이슈는 어디서 뉴스룸 시청자와 뉴스 제보하기 ' \
                 '한국일보닷컴 바로가기 헤밀 헤럴드경제 네이버 채널 헤럴드에코 밀리터리 전문 콘텐츠 카카오톡 무단복제 촬영기자 ' \
                 '기사문의 카톡 라인 네이버에서 구독하세요 생방송 만나보기 균형있는 뉴스 다운받기 인터뷰 배포금지 앵커 사진 ' \
                 '고품격 뉴스레터 원클릭으로 구독하세요 한국경제신문과 모바일한경으로 보세요 한국경제 금지 원본출처 정치읽기 ' \
                 '뉴스데스크 엠빅뉴스 이데일리 앵커멘트 뉴스데스크 자료사진 경향신문 뉴스투데이 서울신문 일간스포츠 아시아경제 '\
                 '출처 기사 캡처 CBS 이미지투데 서울경제 짤롱뉴스 시나뉴스 관련 자료나우뉴스 속보 화면제공 캡쳐 사진 내용 기사문 '
    stop_words = stop_words.split(' ')
    sentence = re.sub('[^가-힣]', ' ', sentence)
    sentence = re.sub('[ ]{2,}', ' ', sentence)
    print(sentence)
    if sentence == ' ':
        return sentence

    # 불용어 제거 (Contents)
    if remove_stopwords:
        hannanum = Hannanum()
        kor_nouns = hannanum.nouns(sentence)
        tokens = ' '.join(word for word in kor_nouns if not word in stop_words if len(word) > 1)

    # 불용어 미제거 (Title)
    else:
        tokens = ' '.join(word for word in sentence.split() if len(word) > 1)
    print(tokens)
    return tokens


np.random.seed(seed=0)

csv_read_list = [# 'D:/study/python/UROP/crawling/crawled_naverNews.csv',
                 'D:/study/python/UROP/crawling/crawled_ruliWeb.csv',
                 'D:/study/python/UROP/crawling/crawled_natePann.csv',
                 'D:/study/python/UROP/crawling/crawled_kidsBook.csv',
                 'D:/study/python/UROP/crawling/crawled_kidsSong.csv']

csv_save_list = [# 'D:/study/python/UROP/preprocessing/preprocessed_naverNews.csv',
                 'D:/study/python/UROP/preprocessing/preprocessed_ruliWeb.csv',
                 'D:/study/python/UROP/preprocessing/preprocessed_natePann.csv',
                 'D:/study/python/UROP/preprocessing/preprocessed_kidsBook.csv',
                 'D:/study/python/UROP/preprocessing/preprocessed_kidsSong.csv']

for idx in range(len(csv_read_list)):
    print('======================================================')
    print('Read -> ', csv_read_list[idx], '\n')
    data = pd.read_csv(csv_read_list[idx], nrows=100000)
    print('Initial,\tLength:', (len(data)))

    data = data[['제목', '내용']]

    print('Duplicate Check, Title: ', data['제목'].nunique(), ' / Contents: ', data['내용'].nunique())
    data.drop_duplicates(subset=['내용'], inplace=True)
    print('After Drop Duplicate, Length: ', len(data))

    data.dropna(axis=0, inplace=True)
    print('After Drop Null,\tLength: ', len(data))

    titleDF = []
    for idxEnum, title in enumerate(data['제목']):
        print(idxEnum, '/', len(data['제목']), end=" ")
        titleDF.append(preprocess_sentence(title, False))

    contentsDF = []
    for idxEnum, contents in enumerate(data['내용']):
        print(idxEnum, '/', len(data['내용']), end=" ")
        print(contents)
        contentsDF.append(preprocess_sentence(contents))

    dataFrame = []
    dataFrame = pd.DataFrame(dataFrame, columns=['title', 'contents'])
    dataFrame['title'] = titleDF
    dataFrame['contents'] = contentsDF
    print('After Preprocessing Function,\tLength: ', len(dataFrame))

    dataFrame.replace('', np.nan, inplace=True)
    dataFrame.dropna(axis=0, inplace=True)
    print('After Drop Null,\tLength: ', len(dataFrame))

    dataFrame = dataFrame[dataFrame['contents'].apply(lambda x: len(x.split()) > 5)]
    print('After Drop Less than 10, Data Length: ', len(dataFrame))
    print('\nFinal Len:\t', len(dataFrame))

    dataFrame.to_csv(csv_save_list[idx], encoding='utf-8-sig', index=True)

    print('\nDone -> ', csv_save_list[idx])
    print('======================================================\n')
