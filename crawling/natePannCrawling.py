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

def getNatePannData():
    url = 'https://pann.nate.com/talk/ranking/m?stdt='
    search_period_month = 60
    start_period = 202105
    find_period = start_period

    title_list = []
    description_list = []

    for monthIdx in range(search_period_month):
        print(find_period)
        move_url = url + str(find_period) + '&page=1'
        driver.get(move_url)
        links = driver.find_elements_by_class_name('rankNum')

        print("len: ", len(links))

        href_list = []

        for linkIdx in range(len(links)):
            href = driver.find_element_by_xpath('//*[@id="container"]/div[3]/div[1]/div[2]/div[2]/ul/li['
                                                + str(linkIdx + 1) + ']/dl/dt/a').get_attribute('href')
            href_list.append(href)
            print(href)

        cnt = 1

        for href in href_list:
            print("cnt: ", cnt)
            driver.get(href)

            if check_exists_by_xpath('//*[@id="container"]/div[4]/div[1]/div[1]/div[1]/h4'):
                title = driver.find_element_by_xpath('//*[@id="container"]/div[4]/div[1]/div[1]/div[1]/h4').text
                title_list.append(title)
                print("title: " + title)

                description = driver.find_element_by_xpath('//*[@id="contentArea"]').text
                description_list.append(description + " ")
                #print("description: " + description)

            cnt += 1

        find_period -= 1

        if find_period % 100 == 0:
            find_period -= 88


    natePannDic = {
        '제목': title_list,
        '내용': description_list
    }

    natePannDf = pd.DataFrame(natePannDic)
    natePannDf.to_csv('natePannHot.csv', encoding='utf-8-sig', index=True)

    driver.close()


getNatePannData()
