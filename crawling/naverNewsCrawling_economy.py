import pandas as pd
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException

driver = webdriver.Chrome('C:\\chromeDriver\\chromedriver.exe')


def check_exists_by_xpath(xpath):
    try:
        driver.find_element_by_xpath(xpath)
    except NoSuchElementException:
        return False
    return True


def getNaverNewsData():
    search_date_num = 1000
    start_period = 20210729
    date_list = []
    title_list = []
    description_list = []
    total_cnt = 0
    url = 'https://news.naver.com/main/list.naver?mode=LS2D&mid=shm&sid2=258&sid1=101&date=%d' % start_period

    for dateIdx in range(search_date_num):
        dateIdx += 1
        driver.get(url)

        if check_exists_by_xpath('//*[@id="main_content"]/div[4]/span[3]'):
            date = driver.find_element_by_xpath('//*[@id="main_content"]/div[4]/span[3]').text
            print(date)

            for pageIdx in range(70):
                pageIdx += 1
                move_url = url + '&page=%d' % pageIdx
                driver.get(move_url)

                if check_exists_by_xpath('//*[@id="main_content"]/div[3]/strong'):
                    if driver.find_element_by_xpath('//*[@id="main_content"]/div[3]/strong').text != str(pageIdx):
                        print(driver.find_element_by_xpath('//*[@id="main_content"]/div[3]/strong').text)
                        print(pageIdx)
                        break

                href_list = []

                for upperListIdx in range(2):
                    upperListIdx += 1
                    for lowerListIdx in range(10):
                        lowerListIdx += 1
                        if check_exists_by_xpath('//*[@id="main_content"]/div[2]/ul[%d]/li[%d]/dl/dt[2]/a'
                                                 % (upperListIdx, lowerListIdx)):
                            href = driver.find_element_by_xpath(
                                '//*[@id="main_content"]/div[2]/ul[%d]/li[%d]/dl/dt[2]/a'
                                % (upperListIdx, lowerListIdx)).get_attribute('href')
                            href_list.append(href)

                for href in href_list:
                    driver.get(href)

                    if check_exists_by_xpath('//*[@id="articleTitle"]'):
                        title = driver.find_element_by_xpath('//*[@id="articleTitle"]').text

                        if check_exists_by_xpath('//*[@id="articleBodyContents"]'):
                            description = driver.find_element_by_xpath('//*[@id="articleBodyContents"]').text

                            date_list.append(date + " ")
                            title_list.append(title)
                            description_list.append(description + " ")
                            total_cnt += 1

                            print("total %d, idx %d, %s, %s" % (total_cnt, len(title_list), date, title))

            if check_exists_by_xpath('//*[@id="main_content"]/div[4]/a[3]'):
                url = driver.find_element_by_xpath('//*[@id="main_content"]/div[4]/a[3]').get_attribute('href')

        if dateIdx % 100 == 0:
            naverNewsDic = {
                '날짜': date_list,
                '제목': title_list,
                '내용': description_list
            }

            naverNewsDf = pd.DataFrame(naverNewsDic)
            naverNewsDf.to_csv('crawled_data_economy/crawled_naverNews_economy_%d.csv' % (dateIdx / 100),
                               encoding='utf-8-sig', index=True)

            date_list = []
            title_list = []
            description_list = []

    driver.close()


getNaverNewsData()
