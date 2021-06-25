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

def getRuliWebData():
    url = 'https://bbs.ruliweb.com/best/selection?orderby=readcount&range=30d&page='
    search_page_num = 40
    title_list = []
    description_list = []

    for pageIdx in range(search_page_num):
        print(pageIdx)
        move_url = url + str(pageIdx + 1)
        driver.get(move_url)
        links = driver.find_elements_by_class_name('table_body')


        print("len: ", len(links))

        href_list = []
        for linkIdx in range(len(links)):
            href = driver.find_element_by_xpath('// *[ @ id = "best_body"] / table / tbody / \
            tr[' + str(linkIdx + 1) + '] / td[2] / a').get_attribute('href')
            href_list.append(href)
            print(href)

        cnt = 1

        for href in href_list:
            print("cnt: ", cnt)
            driver.get(href)

            print(move_url)

            if check_exists_by_xpath('//*[@id="board_read"]/div/div[2]/div[1]/div[1]/div[1]/h4/span/span[2]'):
                title = driver.find_element_by_xpath('//*[@id="board_read"]/div/div[2]/div[1]/div[1]/div[1]/h4/span/span[2]').text
                title_list.append(title)
                print("title: " + title)

                description = driver.find_element_by_xpath('//*[@id="board_read"]/div/div[2]/div[2]/div[1]').text
                description_list.append(description + " ")
                #print("description: " + description)

            cnt += 1


    ruliWebDic = {
        '제목': title_list,
        '내용': description_list
    }

    ruliWebDf = pd.DataFrame(ruliWebDic)
    ruliWebDf.to_csv('ruliWebHot.csv', encoding='utf-8-sig', index=True)

    driver.close()


getRuliWebData()
