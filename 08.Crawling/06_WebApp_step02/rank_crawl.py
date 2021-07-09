from selenium import webdriver 
from bs4 import BeautifulSoup 
import time
import pandas as pd



#url = "https://youtube-rank.com/board/bbs/board.php?bo_table=youtube" 
#browser.get(url)
def crawl():
    # 사이트의 정보를 크롤링
    browser = webdriver.Chrome('c:/driver/chromedriver.exe')
    results = []
    for page in range(1,11):
        url = "https://youtube-rank.com/board/bbs/board.php?bo_table=youtube&page={page}" 
        browser.get(url)
        time.sleep(1)

        html = browser.page_source
        soup = BeautifulSoup(html, 'html.parser')
        channel_list = soup.select('form > table > tbody > tr')
        time.sleep(3)

        for channel in channel_list:
            title = channel.select('h1 > a')[0].text.strip() 
            category = channel.select('p.category')[0].text.strip()
            subscriber = channel.select('.subscriber_cnt')[0].text 
            view = channel.select('.view_cnt')[0].text
            video = channel.select('.video_cnt')[0].text
            data = [title, category, subscriber, view, video]
            results.append(data)

    # 받아온 데이터 값을 넣은 results라는 리스트를 데이터 프레임으로 만들어줌
    df = pd.DataFrame(results)
    df.columns = ['title', 'category', 'subscriber', 'view', 'video']

    #데이터 프레임을 엑셀로 저장
    df.to_excel('./files/youtube_rank1.xlsx', index = False)

    #크롤링 종료
    browser.implicitly_wait(3)
    browser.quit()

    return results