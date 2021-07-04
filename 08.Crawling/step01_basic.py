#html 필수 표현법
'''
    . : class 속성
    # : id 속성
    이름 : tag 속성?
'''

html_doc = """<html><head><title>The Dormouse's story</title></head>
<body>
<p class="title"><b>The Dormouse's story</b></p>

<p class="story">Once upon a time there were three little sisters; and their names were
<a href="http://example.com/elsie" class="sister" id="link1">Elsie</a>,
<a href="http://example.com/lacie" class="sister" id="link2">Lacie</a> and
<a href="http://example.com/tillie" class="sister" id="link3">Tillie</a>;
and they lived at the bottom of a well.</p>

<p class="story">...</p>
"""

html='''
<html>
    <body>
        <h1>스크래핑이란?</h1>
        <h1>스크래핑이란?!?!</h1>
        <p id="one">웹페이지 1</p>
        <p id="two">웹페이지 2</p>
        <span class="redColor">
            <p>웹페이지3</p>
        </span>
        <ul>
            <li><a href="www.daum.net">다음</a></li>
            <li><a href="www.naver.com">네이버</a></li>
        </ul>        
    </body>
</html>
'''

# bs4 - html문서를 tag, 속성, css 등으로 섬세하게 관리 가능한 API
from bs4 import BeautifulSoup
#soup = BeautifulSoup(html_doc, 'html.parser')

#크롤링 대상의 데이터와 구문해석, 문법체크 , 변환가능한 parser 설정
soup = BeautifulSoup(html, 'html.parser')

#print(soup.prettify())
print("--------2 : find() 함수--------")
print(soup.find(id = "one"))        #<p id="one">웹페이지 1</p>
print(soup.find(id = "one").string)        #웹페이지 1
print(soup.select('.redColor'))     #웹페이지3
print(soup.select('.redColor p')[0].get_text())


print("--------1.----------")
print(soup.html.h1)     #<h1>스크래핑이란?</h1>
print(type(soup.html.p))    #<class 'bs4.element.Tag'>

print(soup.html.p)      #<p id="one">웹페이지 1</p>
print(soup.html.p.next_sibling)     #
print(soup.html.p.next_sibling.next_sibling)    #<p id="two">웹페이지 2</p>

#<p id="one">웹페이지 1</p>
# <p id="two">웹페이지 2</p>
#html(xml) 문서는 트리 구조
#html 상에서 newline(br tag)는 text 자식으로 간주
#next_sibling : 현 위치 상에서 다음 나의 형제


#web page3 찾아가기 
#html 문서의 스팬 태그 하위의 텍스트데이터
print(soup.html.span.p)     #<p>웹페이지3</p>
print(soup.html.span.p.string)      #웹페이지3
print(soup.html.span.p.get_text())      #웹페이지3


