#미션 : 구글에서 검색 가능하게 step01 처럼 작업 권장

from selenium import webdriver
import time

driver = webdriver.Chrome("c:/driver/chromedriver")
driver.get("https://www.google.com/")

tag = driver.find_element_by_name("q")

tag.clear()
tag.send_keys("data")
tag.submit()

time.sleep(5)
driver.quit()