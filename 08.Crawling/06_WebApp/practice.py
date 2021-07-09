import time
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common import by
from selenium.webdriver.support.ui import WebDriverWait

driver = webdriver.Chrome("c:/driver/chromedriver")
driver.get("https://google.com")

#검색창의 html 코드
'''
<input class="gLFyf gsfi" jsaction="paste:puy29d;" 
maxlength="2048" name="q" type="text" aria-autocomplete="both" 
aria-haspopup="false" autocapitalize="off" autocomplete="off" 
autocorrect="off" autofocus="" role="combobox" spellcheck="false"
title="검색" value="" aria-label="검색" data-ved="0ahUKEwjb-5fA383xAhXRzDgGHSifC1cQ39UDCAQ">
'''
tag = driver.find_element_by_name("q")
tag.clear()
tag.send_keys("아이유")
tag.submit()

link = driver.find_element_by_xpath("//*[@id='hdtb-msb']/div[1]/div/div[2]/a")
link.click()

'''
<a class="VFACy kGQAp sMi44c lNHeqe WGvvNb" data-ved="2ahUKEwi3up7c483xAhVZR5QKHQYVAXUQr4kDegUIARDVAQ" jsname="uy6ald" rel="noopener" target="_blank" href="https://www.chosun.com/entertainments/entertain_photo/2020/12/24/QFJHZ6CTPBLLHUNKHZHWZLTICA/" jsaction="focus:kvVbVb;mousedown:kvVbVb;touchstart:kvVbVb;" title="이래서 아이유, 아이유 하는구나…성탄절 1억원 기부→올해 6억↑ “날개없는 천사” - 조선일보">이래서 아이유, 아이유 하는구나…성탄절 1억원 기부→올해 6억↑ “날개없는 천사” - 조선일보<div class="fxgdke">chosun.com</div></a>
'''
'''
<a class="VFACy kGQAp sMi44c lNHeqe WGvvNb" data-ved="2ahUKEwi3up7c483xAhVZR5QKHQYVAXUQr4kDegUIARDXAQ" jsname="uy6ald" rel="noopener" target="_blank" href="https://programs.sbs.co.kr/star/iu/main" jsaction="focus:kvVbVb;mousedown:kvVbVb;touchstart:kvVbVb;" title="아이유(IU) 스타채널 : SBS">아이유(IU) 스타채널 : SBS<div class="fxgdke">programs.sbs.co.kr</div></a>
'''

lst = []
try :
    soup = BeautifulSoup(driver.page_source, "lxml")
    datas = soup.select(".VFACy")

    driver.implicitly_wait(10)
    time.sleep(5)
    #print(datas)
    # print(type(datas))
    print('----------------')

    href_lst = []

    for data in datas:
        href = data["href"]
        title = data["title"]   

        lst.append({"기사 제목" : title, "기사 링크" : href})


except Exception as e:
    print("page parseing error", e)    

finally:
    time.sleep(3)
    driver.close()




