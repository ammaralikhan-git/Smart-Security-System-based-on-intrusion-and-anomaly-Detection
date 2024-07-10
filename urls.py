# facedetection/urls.py
from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static


urlpatterns = [
    path('login/', views.login_view, name='login'),
    path('register/', views.register, name='register'),
    path('main/', views.main, name='main'),
    path('', views.index, name='index'),
    path('register/page1/', views.registration_page1, name='registration_page1'),
    path('register/page2/', views.registration_page2, name='registration_page2'),
    path('register/page3/', views.registration_page3, name='registration_page3'),
    path('logout/', views.logout_view, name='logout'),
    path('start_face_detection/', views.start_face_detection, name='start_face_detection'),
    path('start_weapon_detection/', views.start_weapon_detection, name='start_weapon_detection'),


] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)