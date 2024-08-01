from django.shortcuts import render
from django.http import JsonResponse
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from openai import OpenAI
import os
from django.conf import settings
import logging
import requests
import json
from django.shortcuts import redirect
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from .forms import UploadFileForm
from io import BytesIO
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import base64
import io


secrets_file_path = os.path.join(settings.BASE_DIR, 'secrets.json')
with open(secrets_file_path) as f:
  secrets = json.load(f)

SECRET_KEY = secrets["SECRET_KEY"]
API_KEY = secrets["API_KEY"] #gpt api key
CLIENT_ID = secrets["client_id"]
CLIENT_SECRET = secrets["client_secret"]

def post_list(request):
    return render(request, 'seniorcare/post_list.html', {})

class ClovaSpeechClient:
    # Clova Speech invoke URL
    invoke_url = 'https://clovaspeech-gw.ncloud.com/external/v1/7421/f5e06aecb198b9141483f5c41f10d801eb612d8c808ae261c7e6402e9d935bbf'
    # Clova Speech secret key
    secret = SECRET_KEY

    def req_url(self, url, completion, callback=None, userdata=None, forbiddens=None, boostings=None, wordAlignment=True, fullText=True, diarization=None, sed=None):
        request_body = {
            'url': url,
            'language': 'ko-KR',
            'completion': completion,
            'callback': callback,
            'userdata': userdata,
            'wordAlignment': wordAlignment,
            'fullText': fullText,
            'forbiddens': forbiddens,
            'boostings': boostings,
            'diarization': diarization,
            'sed': sed,
        }
        headers = {
            'Accept': 'application/json;UTF-8',
            'Content-Type': 'application/json;UTF-8',
            'X-CLOVASPEECH-API-KEY': self.secret
        }
        return requests.post(headers=headers,
                             url=self.invoke_url + '/recognizer/url',
                             data=json.dumps(request_body).encode('UTF-8'))

    def req_object_storage(self, data_key, completion, callback=None, userdata=None, forbiddens=None, boostings=None,
                           wordAlignment=True, fullText=True, diarization=None, sed=None):
        request_body = {
            'dataKey': data_key,
            'language': 'ko-KR',
            'completion': completion,
            'callback': callback,
            'userdata': userdata,
            'wordAlignment': wordAlignment,
            'fullText': fullText,
            'forbiddens': forbiddens,
            'boostings': boostings,
            'diarization': diarization,
            'sed': sed,
        }
        headers = {
            'Accept': 'application/json;UTF-8',
            'Content-Type': 'application/json;UTF-8',
            'X-CLOVASPEECH-API-KEY': self.secret
        }
        return requests.post(headers=headers,
                             url=self.invoke_url + '/recognizer/object-storage',
                             data=json.dumps(request_body).encode('UTF-8'))

    def req_upload(self, file, completion, callback=None, userdata=None, forbiddens=None, boostings=None,
                   wordAlignment=True, fullText=True, diarization=None, sed=None):
        request_body = {
            'language': 'ko-KR',
            'completion': completion,
            'callback': callback,
            'userdata': userdata,
            'wordAlignment': wordAlignment,
            'fullText': fullText,
            'forbiddens': forbiddens,
            'boostings': boostings,
            'diarization': diarization,
            'sed': sed,
        }
        headers = {
            'Accept': 'application/json;UTF-8',
            'X-CLOVASPEECH-API-KEY': self.secret
        }
        print(json.dumps(request_body, ensure_ascii=False).encode('UTF-8'))
        files = {
            'media': open(file, 'rb'),
            'params': (None, json.dumps(request_body, ensure_ascii=False).encode('UTF-8'), 'application/json')
        }
        response = requests.post(headers=headers, url=self.invoke_url + '/recognizer/upload', files=files)
        return response
    


@csrf_exempt
def upload_and_transcribe(request):
    logger = logging.getLogger(__name__)
    client = OpenAI(api_key=API_KEY)
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            # 업로드된 파일을 임시 저장
            file = request.FILES['file']
            file_name = default_storage.save(os.path.join('temp', file.name), ContentFile(file.read()))
            temp_file_path = os.path.join(settings.MEDIA_ROOT, file_name)
            
            try:
                # OpenAI API 호출
                with open(temp_file_path, 'rb') as file_to_transcribe:
                    transcript = client.audio.transcriptions.create(
                        file=file_to_transcribe,
                        model="whisper-1",
                        language="ko",
                        response_format="text",
                        temperature=0.0,
                    )

                print(transcript)
                return HttpResponse(transcript, content_type="text/plain")  


            except Exception as e:
                # 예외 발생 시 에러 메시지 반환
                return JsonResponse({'status': 'error', 'message': str(e)})
            finally:
                # 처리가 완료되면, 임시로 저장된 파일을 삭제
                default_storage.delete(file_name)

            return HttpResponse(transcript, content_type="text/plain")
    else:
        form = UploadFileForm()
    return render(request, 'fileupload/whisper_file_upload.html', {'form': form})




conversation_history = {} #대화를 저장할 임시 저장소

@csrf_exempt
def chatgpt_completion(request):
    client = OpenAI(api_key=API_KEY)
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            content = data.get('content')
            user_id = data.get('user_id')
            signnum = data.get('signnum')
            if not content or not user_id or signnum is None:
                return JsonResponse({'error': 'Invalid request. User ID, content, or signnum missing.'}, status=400)
            
            # 1번(signnum)이 들어오면 해당 사용자의 대화 기록을 삭제합니다.
            if signnum == 1:
                if user_id in conversation_history:
                    del conversation_history[user_id]
                return JsonResponse({'message': 'User conversation history deleted.'}, status=200)
            
            # 사용자 ID에 해당하는 대화 기록을 가져옵니다.
            user_history = conversation_history.get(user_id, [])
            
            chat_completion = client.chat.completions.create(
                messages=user_history + [
                    {
                        "role": "user",
                        "content": content,
                    }
                ],
                model="gpt-3.5-turbo",
            )

            response = chat_completion.choices[0].message.content

            user_history.append({
                "role": "user",
                "content": content,
            })
            user_history.append({
                "role": "assistant",
                "content": response,
            })
            
            # 사용자 ID에 대한 대화 기록을 업데이트합니다.
            conversation_history[user_id] = user_history

            return HttpResponse(response, content_type="text/plain")
            #return JsonResponse({'response': response}, status=200)
        
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON'}, status=400)
    else:
        return JsonResponse({'error': 'Only POST method is supported'}, status=405)
    


@csrf_exempt
def naver_sentiment(request):
    url = "https://naveropenapi.apigw.ntruss.com/sentiment-analysis/v1/analyze"
    headers = {
        "X-NCP-APIGW-API-KEY-ID": CLIENT_ID,
        "X-NCP-APIGW-API-KEY": CLIENT_SECRET,
        "Content-Type": "application/json"
    }

    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            content = data.get('content')
            if not content:
                return JsonResponse({'error': 'No content provided'}, status=400)

            # 내용을 900자로 제한합니다.
            data = {
                "content": content[:900]
            }
            response = requests.post(url, data=json.dumps(data), headers=headers)
            result = json.loads(response.text)
            if response.status_code == 200:
                # API 응답 결과를 클라이언트에게 전달합니다.
                sentiment_result = result["document"]["sentiment"]
                return HttpResponse(sentiment_result, content_type="text/plain")
            else:
                return JsonResponse({'error': 'Failed to analyze sentiment'}, status=response.status_code)
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON'}, status=400)
    else:
        return JsonResponse({'error': 'Only POST method is supported'}, status=405)
    

    
@csrf_exempt
def open_tts(request):
    client = OpenAI(api_key=API_KEY)
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            content = data.get('content')
            if content is None:
                return JsonResponse({'error': 'Invalid request. Content is required.'}, status=400)
            
            # TTS 모델을 사용하여 음성 파일 생성
            with client.audio.speech.with_streaming_response.create(
                model="tts-1",
                voice="alloy",
                input=content
            ) as response:
                response.stream_to_file("speech.mp3")
            
            # 생성된 음성 파일을 읽어서 HTTP 응답으로 반환
            with open("speech.mp3", "rb") as audio_file:
                response = HttpResponse(audio_file.read(), content_type='audio/mpeg')
                #response['Content-Disposition'] = 'attachment; filename="speech.mp3"' #다운로드용 코드
                response['Content-Disposition'] = 'inline; filename="speech.mp3"' #즉시 재생용 코드
                return response
        
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON.'}, status=400)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    
    return JsonResponse({'error': 'Invalid request method.'}, status=405)



@csrf_exempt
def open_vision(request):
    client = OpenAI(api_key=API_KEY)

    # 이미지를 base64로 인코딩하고 PIL 이미지 객체를 반환하는 함수
    def load_and_encode_images(image_sources):
        encoded_images = []
        pil_images = []
        for source in image_sources:
            if source.startswith('http'):  # URL인 경우
                response = requests.get(source)
                image_data = response.content
            else:  # 파일 경로인 경우
                with open(source, "rb") as image_file:
                    image_data = image_file.read()

            pil_images.append(Image.open(io.BytesIO(image_data)))
            encoded_images.append(base64.b64encode(image_data).decode('utf-8'))
        return encoded_images, pil_images


    # 응답결과와 이미지를 출력하기 위한 함수
    def display_response(pil_images, response_text):
        # 이미지 로딩 및 서브플롯 생성
        fig, axes = plt.subplots(nrows=1, ncols=len(pil_images), figsize=(5 * len(pil_images), 5))
        if len(pil_images) == 1:  # 하나의 이미지인 경우
            axes = [axes]

        # 이미지들 표시
        for i, img in enumerate(pil_images):
            axes[i].imshow(img)
            axes[i].axis('off')  # 축 정보 숨기기
            axes[i].set_title(f'Image #{i + 1}')

        # 전체 플롯 표시
        #plt.show()

        print(response_text)


    # 이미지 경로 또는 URL과 프롬프트를 처리하는 함수
    def process_and_display_images(image_sources, prompt):
        # 이미지 로드, base64 인코딩 및 PIL 이미지 객체 생성
        base64_images, pil_images = load_and_encode_images(image_sources)

        # OpenAI에 요청 보내기
        messages = [
            {
                "role": "user",
                "content": [
                            {"type": "text", "text": prompt}
                        ] + [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}} for
                                base64_image in base64_images]
            }
        ]

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=1000
        )

        response_text = response.choices[0].message.content

        # 응답과 이미지 표시
        display_response(pil_images, response.choices[0].message.content)

        return response_text


    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            # 업로드된 파일을 임시 저장
            file = request.FILES['file']
            file_name = default_storage.save(os.path.join('temp', file.name), ContentFile(file.read()))
            temp_file_path = os.path.join(settings.MEDIA_ROOT, file_name)
            
            try:
                # OpenAI API 호출
                image_sources = [temp_file_path]
                prompt = "이 사진에서 텍스트 추출해서 OCR 수행해줘"

                response_text = process_and_display_images(image_sources, prompt)
                print(response_text)
                return HttpResponse(response_text, content_type="text/plain")  


            except Exception as e:
                # 예외 발생 시 에러 메시지 반환
                return JsonResponse({'status': 'error', 'message': str(e)})
            finally:
                # 처리가 완료되면, 임시로 저장된 파일을 삭제
                default_storage.delete(file_name)

            return HttpResponse(transcript, content_type="text/plain")
    else:
        form = UploadFileForm()
    return render(request, 'fileupload/whisper_file_upload.html', {'form': form})

