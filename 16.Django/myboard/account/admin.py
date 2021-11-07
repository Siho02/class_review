from django.contrib import admin
from .models import CustomUser
from django.contrib.auth.admin import UserAdmin

#UserAdmin : 관리자 앱에서 사용자 관리를 위한 화면 구성
# 사용자 관리 화면 변경시 UserAdmin을 상속 받아서 구현

class CustormUserAdmin(UserAdmin):
    #class변수를 변경
    UserAdmin.fieldsets[1][1]['fields'] = ('name', 'email', 'gender')

    #사용자 목록 화면에 나올 field 정의
    list_display = ['name', 'email', 'gender', 'date_joined']

admin.site.register(CustomUser, CustormUserAdmin)

