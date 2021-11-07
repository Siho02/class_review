"""confi URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from re import template
from django.contrib import admin
from django.urls import path, include
from django.views.generic import TemplateView #뒤쪽에서 배움 그냥 그러려니 일단 하시오

urlpatterns = [
    #http://127.0.0.1:8000/ >> home.html을 직접 호출
    path("", TemplateView.as_view(template_name = 'home.html'), name = 'home'), #
    path('admin/', admin.site.urls),
    path('poll/', include("poll.urls")), #url path가 poll/
    
]
