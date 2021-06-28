import time

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


def getKidsBookData():
    url = 'http://book.interpark.com/display/collectlist.do?_method=BestsellerHourNew201605&bestTp=1&dispNo=028031'

    search_period_num = 200
    start_period = 2021050
    find_period = start_period
    title_list = []
    description_list = []

    for dateIdx in range(search_period_num):
        if dateIdx == 100:
            url = 'http://book.interpark.com/display/collectlist.do?_method=BestsellerHourNew201605&bestTp=1&dispNo=028008#'
            find_period = start_period

        driver.get(url)
        driver.find_element_by_xpath('//*[@id="cateTabId3"]/a').click()
        driver.execute_script('goBestMonth(\'28\',\'' + str(find_period) + '\',\'monthSch\')')
        href_list = []
        time.sleep(2.5)

        for book_idx in range(15):
            if check_exists_by_xpath('//*[@id="content"]/div[3]/div[3]/div[2]/div/div[1]/ol/\
                li[' + str(book_idx + 1) + ']/div/a'):
                href = driver.find_element_by_xpath('//*[@id="content"]/div[3]/div[3]/div[2]/div/div[1]/ol/\
                li[' + str(book_idx + 1) + ']/div/a').get_attribute('href')
                href_list.append(href)

        cnt = 1

        for href in href_list:
            print("cnt: ", cnt)
            driver.get(href)

            try:
                alert = driver.switch_to.alert
                alert.accept()
            except:
                pass

            print(find_period)

            if check_exists_by_xpath('//*[@id="inc_titWrap"]/div[1]/div/h2'):
                title = driver.find_element_by_xpath('//*[@id="inc_titWrap"]/div[1]/div/h2').text
                title_list.append(title)
                print("title: " + title)

                description = driver.find_element_by_xpath('//*[@id="bookInfoWrap"]/div[2]').text
                description_list.append(description + " ")
                # print("description: " + description)

            cnt += 1

        find_period -= 20

        if find_period % 1000 == 0 or find_period % 1000 > 120:
            find_period -= 880

    kidsBookDic = {
        '제목': title_list,
        '내용': description_list
    }

    kidsBookDf = pd.DataFrame(kidsBookDic)
    kidsBookDf.to_csv('crawled_kidsBook.csv', encoding='utf-8-sig', index=True)

    driver.close()

getKidsBookData()
