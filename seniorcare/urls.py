from django.urls import path
from . import views

urlpatterns = [
    path('', views.post_list, name='post_list'),
    path('whisper', views.upload_and_transcribe, name='upload_and_transcribe'),
    path('sentiment', views.naver_sentiment, name='naver_sentiment'),
    path('chatcomp', views.chatgpt_completion, name='chatgpt_completion'),
    path('opentts', views.open_tts, name='open_tts'),
    path('openvision', views.open_vision, name='open_vision'),
    path('health/', views.health_check, name='health_check'),
]