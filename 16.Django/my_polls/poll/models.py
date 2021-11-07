from django.db import models


'''
# Question(질문) 테이블과 연결된 모델 클래스
    -qustion_text = 질문 내용 - varchar
    -pub_date = 등록 일시 - datetime
    -id(PK - 자동생성되도록 할거임) - 1씩 자동적으로 증가하는 값
''' 

#모델 클래스
#    -models.Model를 상속
#    -컬럼 관련 변수들을 class변수로 정의 >> Field
#    -class이름 : 테이블명으로 이름 지정, Pascal표기법 사용
class Question(models.Model):
    #Field 클래스 변수 : 변수명-컬럼명, 값-field객체 >> 데이터 타입, 추가 설정
    #Primary key 컬럼 생략 >> 자동으로 id라는 이름의 자동증가 정수형 컬럼이 생성된다

    question_text = models.CharField(max_length=200) #charfiled : 문자열(varchar)
    pub_date = models.DateTimeField(auto_now_add=True) #Datetimefiled : 일시타입(date) - auto_now_add=True : 일시를 자동으로 등록

        #sql에서 다음과 비슷
        #create table Question(
        #   question_text varchar(200) not null primary
        # )

    def __str__(self):
        return self.question_text

#Choice - 보기
#   -choice_text : 보기 내용 - varchar
#   -vote : 몇번 선택 되었는지 - int(number)
#   -question : 질문에 대한 보기 - Quesion의 Foreign key 컬럼
#   -id : 자동증가 PK (따로 만들지 않겠다)
class Choice(models.Model):
    #field 정의
    choice_text = models.CharField(max_length=200)
    vote = models.PositiveIntegerField(default=0) #PositiveIntegerField:양수정수타입 / defalut:기본값 설정
    question = models.ForeignKey(to=Question, on_delete=models.CASCADE) #ForiegnKey : ForeignKey필드 선언 / to : 부모 테이블, on_delete : 참조 데이터 삭제시 설정(CASECADE : 부모가 지워질시 참조하는 자식 테이블도 삭제)
    
    def __str__(self):
        return self.choice_text
    

# 모델 클래스를 작성/변경
# python manage.py makemigrations : DB에 테이블을 어떻게 만들지 정의
# python manage.py migrate        : DB에 적용(테이블 생성, 변경) / model과 db를 연결(연동)하는 역할