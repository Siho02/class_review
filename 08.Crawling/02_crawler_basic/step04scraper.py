''' step05_scraper.py
1. 주제
    한빛 미디어의 책 list page에서 각각의 list의 link 정보 추출해서 개별 상세 page의 url값 스크래핑 하기
    스크래핑한 자료를 DB에 저장

    책제목(title) & link 정보 - DB에 저장

2. 활용 사이트
    http://www.hanbit.co.kr/store/books/full_book_list.html
    한빛 -> Store -> 전체 목록

3. 의도 및 구현 방식
    1. 파이썬으로 스크래핑 흐름 이해하기
        - 전체도서목록 page -> 각 도서별 링크 정보 스크래핑 해서 -> DB에 데이터 저장하기
        - 실행후 books.db가 존재하는 경로인 이 python예제가 실행된 경로내에서 검색
            >sqlite3 books.db
            sqlite>select * from books;
        

    2. 스크래핑 해서 DataBase에 저장되는 정보
    1. 리스트의 도서명(title)
    2. 도서명의 하이퍼링크 url 값
    
    3. process
        3-1. 웹페이지 추출하기 : fetch()
        3-2. 스크래핑 하기 : scrape()
        3-3. db에 데이터를 저장하기 : save()

        호출 순서 : fetch() >> scrape() >> save() : 정형 프로세스
        개발 권장 방식 : 순서(process)에 맞게 호출하는 별도의 함수 개발

    4. 구현 함수 상세 내용
        - 각 기능을 함수로 처리 
        4-1. 웹페이지 추출하기 : fetch(url)
            : 매개 변수로 url을 받고 지정한 url의 웹 페이지를 추출

        4-2. 필요한 정보만 스크래핑 하기 : scrape(html)
            : 매개변수로 html을 받고, 정규 표현식을 사용해 HTML에서 도서 정보 추출

        4-3. 데이터를 RDBMS에 저장하기 : save(data)
        : 매개변수로 저장할 정보를 받고, Oracld DB 에 저장
'''

import re

#웹 크롤링 지원 API
from urllib.request import urlopen
from html import unescape

import cx_Oracle
import os


#로직 - 완성
def main():
    """
    호출 순서
        fetch(), scrape(), save() 순으로 호출
    """
    html = fetch('http://www.hanbit.co.kr/store/books/full_book_list.html')
    # print(html)


    books = scrape(html)
    #print(books)
    
    save(books)


# 완성
# 웹페이지 추출하기 : fetch()
# 웹 페이지 인코딩 형식은 추출한 인코딩 정보를 기반으로 추출
def fetch(url):  
    f = urlopen(url) #url에 해당하는 web page를 통으로 크롤링
    
    # HTTP 헤더를 기반으로 인코딩 형식을 추출
    encoding = f.info().get_content_charset(failobj="utf-8")
    print("--- 1. 인코딩 정보 : ", encoding)
    
    # 추출한 인코딩 형식을 기반으로 문자열을 디코딩, 정보로 디코딩 안할 경우 할글 깨짐 현상 발생 가능성 있음
    # why? : 특정 사이트는 encoding 정보가 EUC-KR 으로 되어 있기도 함, 해당 사이트 구현되어 있는 정보로 디코딩 수행
    # f 변수가 보유한 페이지의 모든 정보를 utf-8 포맷으로 복원(디코딩)
    html = f.read().decode(encoding)

    #print("--- 2. 모든 문서 내용 추출 ", html)
    return html

    

#로직 - 미완성
# 필요한 정보만 스크래핑 하기 : scrape()
# 다운로드한 웹페이지에서 필요한 정보를 추출하는 기술
'''
추출할 데이터 일부 예시
url : http://www.hanbit.co.kr/store/books/look.php?p_code=B7198274060
title :  재미있고 빠른 한글 1권 : 기본 모음과 자음
'''
def scrape(html):
    """
    매개변수 html로 받은 HTML 문서의 내용을 정규 표현식을 사용해서 도서 정보를 추출
    반환값: 도서(dict) 리스트
    """
    books = []

    #title과 link url 추출
    #반복문의 in 연산자에 html에서 특정 영역을 도출
    '''
        <td class="left"><a href="/store/books/look.php?p_code=B4300598719">리눅스 입문자를 위한 명령어 사전</a></td>
    '''
    for data in re.findall(r'<td class="left"><a.*?/td>' , html):
        print('------')
        print(data)
        d1 = re.search(r'/store(.*)"', data).group()
        d2 = re.sub('"', '', d1)  
        link = re.search(r'<a href="(.*?)">', data)
        #print(link.group(1))
        url = 'http://www.hanbit.co.kr' + link.group(1)
        title = re.sub(r'<.*?>', '',data)
        
        books.append({'url' : url, 'title' : title})
    
    return books

#로직 - 미완성
# 데이터를 RDBMS에 저장하기 - books table에 저장하기
def save(books):
    # 접속
    con = cx_Oracle.connect("SCOTT/TIGER@localhost:1521/xe")
    # 커서를 추출
    cur = con.cursor()
    cur.execute('''DROP TABLE books''') #에러 발생시 다음 로직 실행 불가(예외처리 해두지 않아서!)
    # books 테이블을 생성 - 
    cur.execute('''
        CREATE TABLE books (
            title varchar2(200),
            url varchar2(200)
        )
    ''')
    # executemany() 메서드를 사용하면 매개변수로 리스트를 지정할 수 있음
    cur.executemany('INSERT INTO books VALUES (:title, :url)', books)
    con.commit()
    cur.close()
    con.close()

if __name__ == '__main__':
    main()