#pip install selenium & chrome driver 셋팅
from selenium import webdriver
#실행을 잠시 중지(sleep(초단위))
import time

# step01 : 네이버에 실행 후  AI 키워드로 검색
# 브라우저 실행
# c:/driver/chromedriver.exe (Window)
# c:/driver/chromedriver (Mac)
driver = webdriver.Chrome("c:/driver/chromedriver")

# http의 문서를 획득(브라우저에서 실행시킬 사이트 url)
# http 프로토콜 관점에선  http://www.naver,com/ 즉 url로 직접 요청하는 기본 방식을 get 방식
driver.get("https://www.naver.com/")

#tag 검색
'''
<input id="query" name="query" type="text" title="검색어 입력" maxlength="255" 
    class="input_text" tabindex="1" accesskey="s" style="ime-mode:active;" autocomplete="off" 
    placeholder="검색어를 입력해 주세요." onclick="document.getElementById('fbm').value=1;" 
    value="" data-atcmp-element="">
'''
tag = driver.find_element_by_name("query")
#검색한 input tag의 내용이 있으면 수정(삭제)
tag.clear()  

#input  tag에 데이터 입력 - 키보드로 입력하는 것과 동일
tag.send_keys("AI")  # 입력

#전송 버튼 클릭
#입력된 데이터로 검색을 하는 기능(사용자가 키보드 enter키를 누르는 것과 동일한 액션) 
#네이버 자체적으로 구현된 기능으로 구현된 페이지로 이동
tag.submit()

#이동된 웹페이지가 실행되는 초단위
time.sleep(20)  #실행중인 프로그램 중지(sleep) 

#자원반환 측면에서 driver도 중지
driver.quit()

'''
1단계 : 브라우저 실행
2단계 : 검색 입력창 claer
3단계 : AI 입력
4단계 : 검색 버튼 클릭
5단계 : AI 데이터 검색 화면 이동 
'''
