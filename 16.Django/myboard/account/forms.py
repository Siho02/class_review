from django import forms
from django.contrib.auth import get_user_model #settings.py에 등록된 AUTH_USER_MODEL클래스를 반환
from django.contrib.auth.forms import UserCreationForm

# UserCreateForm
#   - 기본 User 모델에서 가입을 위해 제공되는 ModelForm
#   - username/password1/password2
# AbstractUser를 상속해서 사용자정의 UserModel을 만들어 사용하는 경우 ModelForm도 그것에 맞게 수정을 해야 한다.
#   -UserCreateForm을 상속해서 구현
class CustomUserCreateForm(UserCreationForm):
    #password1, password2 form field를 재정의 (UsercreationForm 정의된 것을 변경 )
    password1 = forms.CharField(label='Password', widget=forms.PasswordInput())
    password2 = forms.CharField(label='Password 확인', widget=forms.PasswordInput())

    class Meta:
        model = get_user_model() #account.models.CustomUser 클래스를 반환
        fields = ['username','password1', 'password2', 'name', 'email', 'gender']