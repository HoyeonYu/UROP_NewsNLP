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
    url = 'https://news.naver.com/main/ranking/popularDay.nhn?date='
    search_period_num = 40
    start_period = 20210624
    find_period = start_period
    title_list = []
    description_list = []

    for dateIdx in range(search_period_num):
        move_url = url + str(find_period)
        driver.get(move_url)
        href_list = []

        for box_idx in range(12):
            for innerBox_idx in range(5):
                if check_exists_by_xpath('// *[ @ id = "wrap"] / div[4] / div[2] / div / \
                    div[' + str(box_idx + 1) + '] / ul / li[' + str(innerBox_idx + 1) + '] / div / a'):
                    href = driver.find_element_by_xpath('// *[ @ id = "wrap"] / div[4] / div[2] / div / \
                    div[' + str(box_idx + 1) + '] / ul / li[' + str(innerBox_idx + 1) + '] / div / a').get_attribute('href')
                    href_list.append(href)
                    print('box_idx: ' + str(box_idx) + ', innerBox_idx: ' + str(innerBox_idx))

        cnt = 1

        for href in href_list:
            print("cnt: ", cnt)
            driver.get(href)

            print(move_url)

            if check_exists_by_xpath('//*[@id="articleTitle"]'):
                title = driver.find_element_by_xpath('//*[@id="articleTitle"]').text
                title_list.append(title)
                print("title: " + title)

                description = driver.find_element_by_xpath('//*[@id="articleBodyContents"]').text
                description_list.append(description + " ")
                #print("description: " + description)

            cnt += 1

        find_period -= 4

        if find_period % 100 == 0 or find_period % 100 > 28:
            find_period -= 72

        if (find_period // 100) % 100 == 0 or (find_period // 100) % 100 > 12:
            find_period -= 8800

    naverNewsDic = {
        '제목': title_list,
        '내용': description_list
    }

    naverNewsDf = pd.DataFrame(naverNewsDic)
    naverNewsDf.to_csv('naverNewsHot.csv', encoding='utf-8-sig', index=True)

    driver.close()


getNaverNewsData()
