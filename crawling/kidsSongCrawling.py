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


def getKidsSongData():
    url = 'https://music.bugs.co.kr/album/20095145'
    driver.get(url)
    search_period_num = 300
    href_list = []
    title_list = []
    description_list = []

    for songIdx in range(search_period_num):
        print('song idx: ', songIdx)

        if check_exists_by_xpath('//*[@id="ALBUMTRACK20095145"]/table/tbody/tr[' + (str)(songIdx + 1) + ']/td[3]/a'):
            href = driver.find_element_by_xpath('//*[@id="ALBUMTRACK20095145"]/table/tbody/tr['
                                                + (str)(songIdx + 1) + ']/td[3]/a').get_attribute('href')
            href_list.append(href)
            print(href)
    cnt = 1

    for href in href_list:
        print("cnt: ", cnt)
        driver.get(href)

        print(href)

        if check_exists_by_xpath('//*[@id="container"]/header/div/h1'):
            title = driver.find_element_by_xpath('//*[@id="container"]/header/div/h1').text
            title_list.append(title)
            print("title: " + title)

            description = driver.find_element_by_xpath('//*[@id="container"]/section[2]/div/div/xmp').text
            description_list.append(description + " ")
            print("description: " + description)

        cnt += 1
        driver.get(url)

    kidsSongDic = {
        '제목': title_list,
        '내용': description_list
    }

    naverNewsDf = pd.DataFrame(kidsSongDic)
    naverNewsDf.to_csv('crawled_kidsSong.csv', encoding='utf-8-sig', index=True)

    driver.close()


getKidsSongData()
