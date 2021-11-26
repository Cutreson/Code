import speech_recognition as sr
from gtts import gTTS
import playsound

######################################
def text_to_Speed(text):
    tts = gTTS(text=text,lang="vi")
    filename = "voice_lay_do_4.mp3"
    tts.save(filename)
    playsound.playsound(filename)
#######################################
def speed_to_Text():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Recognizing...")
        audio_data = r.record(source,duration=3)
    try:
        text = r.recognize_google(audio_data,language="vi")
    except:
        text = ""
    print(text)
    return text
########################################

text_to_Speed("Mời bạn lấy đồ ở tủ số 4")
#speed_to_Text()