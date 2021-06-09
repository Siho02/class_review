# 데이터 표현 클래스 : VO pattern or DTO pattern
'''
개발 권장 구조
    1. 모든 멤버 변수 선언 위치 : __init__
'''

class Book:         # 제목 / 저자 / 출판사
    def __init__(self, title, author, price):
        #멤버 변수 선언 - self.변수 표현으로 선언 권장
        self.title = title
        self.author = author
        self.price = 0
        
        if price > 100:
            self.price = price 
        #self.setPrice(price)

    def getTitle(self):
        return self.title
    def setTitle(self, new_title):
        self.title = new_title

    def getAuthor(self):
        return self.author
    def setAuthor(self, new_author):
        self.author = new_author
    
    def getPrice(self):
        return self.price
    def setPrice(self, new_price):
        #self.price 
        if new_price > 100:
            self.price = new_price
        else:
            self.price = 0