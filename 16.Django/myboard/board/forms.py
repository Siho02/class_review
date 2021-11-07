#board/forms.py - Board에서 사용할 Form/Model을 구현할 모델

from django import forms
from .models import Post


#Form field : 사용자로부터 입력받는 부분
#   -label, widget(입력양식), 에러메시지
#class PostForm(forms.Form):
#    title = forms.CharField() #텍스트를 입력 받는 input (input type = 'text')
#    content = forms.CharField(widget = forms.Textarea) #textarear

#Form Field들을 Model을 이용해서 구현하는 Form
#       + save() : insert/update
class PostForm(forms.ModelForm):

    class Meta:
        model = Post    #Form필드를 만들때 참조 / save()시 데이터를 저장할 Model클래스 지정
        exclude = ['[writer']
        fields = "__all__"
        
        #feilds = ['title', 'content']   #두개 필드만 생성
        #exclude = ['category']  #category 뺀 나머지 Feild생성
        #feilds : 모델의 Feild중에서 Form Field로 사용할 Feild 목록을 지정
        #       - model의 모든 필드들을 사용할 경우 : "__all__"을 지정
        #       - model의 필드들을 몇개만 사용할 경우 
        #            - feild = ['필드명', ...] 
        #            - exclude = ['제외할 필드명', ...]