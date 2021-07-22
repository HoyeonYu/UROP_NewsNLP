import numpy as np
import pandas as pd
import re
from konlpy.tag import Hannanum


def preprocess_sentence(sentence):
    stop_words = '단독 동영상 연합뉴스 노컷뉴스 구독 클릭 무단전재 재배포 기자 관련기사 저작권자 제보 무단 전재 금지 중앙일보 ' \
                 '네이버에서 뉴스 채널 구독하기 깔끔하게 훑어주는 세상의 이슈 와이퍼 이슈는 어디서 뉴스룸 시청자와 뉴스 제보하기 ' \
                 '한국일보닷컴 바로가기 헤밀 헤럴드경제 네이버 채널 헤럴드에코 밀리터리 전문 콘텐츠 카카오톡 무단복제 촬영기자 ' \
                 '기사문의 카톡 라인 네이버에서 구독하세요 생방송 만나보기 균형있는 뉴스 다운받기 인터뷰 배포금지 앵커 사진 ' \
                 '고품격 뉴스레터 원클릭으로 구독하세요 한국경제신문과 모바일한경으로 보세요 한국경제 금지 원본출처 정치읽기 ' \
                 '뉴스데스크 엠빅뉴스 이데일리 앵커멘트 뉴스데스크 자료사진 경향신문 뉴스투데이 서울신문 일간스포츠 아시아경제 ' \
                 '출처 기사 캡처 CBS 이미지투데 서울경제 짤롱뉴스 시나뉴스 관련 자료나우뉴스 속보 화면제공 캡쳐 사진 내용 기사문 ' \
                 '디지털타임스 뉴스스탠드 홈페이지 바로가기 파이낸셜뉴스 갈무리 현지 언론 매경닷컴 영상디자인 ' \
                 '로이터 특파원 제공 오늘 어제 내일 이번주 저번주 지난주 다음주 오전 오후 아침 저녁 나우뉴스 현지시간 소셜미디어 ' \
                 '미디어 지난해 단독 인턴기자 게티이미지 누리꾼 이상 이하 현재 뉴스닷컴 스타투데이 조선일보 온라인 커뮤니티 ' \
                 '내달 이번달 저번달 다음달 전날 정오뉴스 이미지출처 참고사진 게티이미지뱅크 뉴시스 신문 국민일보 수집 재배포금지 ' \
                 '실시간 뉴스딱 안녕 오늘 소식 고현준 시사평론가 소식 이야기인데요 기사내용 무관 뉴스카페 기록 영상 경향포럼 ' \
                 '올해 작년 내년 평일 주말 경제뉴스 이코노미스트 게티이미지닷컴 세계일보 이미지출처 풀이 정통사주 운세 토정비결 ' \
                 '당신들 이야기 오마이뉴스 취재 후원하기 앤츠랩 메뉴 실검 어디 이야기 모닝 뉴스리뷰 전문방송 정치팀장 ' \
                 '지식레터 매콤달콤 취업비법 한달 무료 진행 고화질 온에어 서비스 선임기자 끼니로그 라이브 메인 머니투데이 ' \
                 '월드리포트 한밤중 금융이슈 파인애플 모아시스 헉스 동아일보 익스플로러 브라우저 모바일한경 팩트체크 ' \
                 '원클릭 한국경제신문 모바일한경 조선일보 기자들 지디넷코리아 소식 우주이야기 나우뉴스 세상 튜브뉴스 누리꾼들 ' \
                 '궁금 한겨레신문 어디 원문 링크 영상편집 저희 당신 소중 순간 신문 멀티미디어 스토리텔링 조선비즈 백화점상품권' \
                 '스벅쿠폰 아이뉴스 재밌는 조선제보 청춘뉘우스 스냅타 종합 경제정보 미디어 여러분 검색해 시리즈 에디터 ' \
                 '매콤달콤 미리보기 빡침해소 내일부터 '

    stop_words = stop_words.split(' ')
    sentence = re.sub('[^가-힣]', ' ', sentence)
    sentence = re.sub('[ ]{2,}', ' ', sentence)
    sentence = re.sub('^.*.기자$', ' ', sentence)
    sentence = re.sub('^.*.앵커$', ' ', sentence)
    if sentence == ' ':
        return sentence

    hannanum = Hannanum()
    kor_nouns = hannanum.nouns(sentence)

    tokens = ' '
    for word in kor_nouns:
        if word not in stop_words:
            if len(word) > 1:
                tokens += word + ' '

    print(tokens)
    return tokens


np.random.seed(seed=0)

csv_read_list = ['D:/study/python/UROP/crawling/crawled_naverNews.csv',
                 # 'D:/study/python/UROP/crawling/crawled_ruliWeb.csv',
                 # 'D:/study/python/UROP/crawling/crawled_natePann.csv',
                 # 'D:/study/python/UROP/crawling/crawled_kidsBook.csv',
                 # 'D:/study/python/UROP/crawling/crawled_kidsSong.csv'
                 ]

csv_save_list = ['D:/study/python/UROP/preprocessing/preprocessed_naverNews.csv',
                 # 'D:/study/python/UROP/preprocessing/preprocessed_ruliWeb.csv',
                 # 'D:/study/python/UROP/preprocessing/preprocessed_natePann.csv',
                 # 'D:/study/python/UROP/preprocessing/preprocessed_kidsBook.csv',
                 # 'D:/study/python/UROP/preprocessing/preprocessed_kidsSong.csv'
                 ]

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
        titleDF.append(preprocess_sentence(title))

    contentsDF = []
    for idxEnum, contents in enumerate(data['내용']):
        print(idxEnum, '/', len(data['내용']), end=" ")
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
