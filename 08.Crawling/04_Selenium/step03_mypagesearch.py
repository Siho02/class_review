from selenium import webdriver
import time

driver = webdriver.Chrome("c:/driver/chromedriver")
driver.get("http://127.0.0.1:5500/08.Crawling/04_Selenium/step03mypage.html")

input_box = driver.find_element_by_name("data")
btn = driver.find_element_by_id("btn") #id 속성으로 찾는 함수

input_box.clear()
input_box.send_keys("encore")

btn.click()


time.sleep(10)
driver.quit()