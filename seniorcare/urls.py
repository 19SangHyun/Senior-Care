from django.urls import path
from . import views

urlpatterns = [
    path('', views.post_list, name='post_list'),
    path('whisper', views.upload_and_transcribe, name='upload_and_transcribe'),
    path('sentiment', views.naver_sentiment, name='naver_sentiment'),
    path('chatcomp', views.chatgpt_completion, name='chatgpt_completion'),
]