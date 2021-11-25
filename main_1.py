import sys
import cv2
from time import sleep
import cv2
import numpy as np
import sqlite3
from PIL import Image
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QObject
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow
from GUI import Ui_MainWindow
import speech_recognition as sr
from gtts import gTTS
import os
import time
import playsound
from datetime import datetime

duration = 3
Confidence = 50
###########################################
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("recoginizer/trainningData.yml")
#############################################
def voice_gui_do(SoTu):
    if(SoTu == 1):
        playsound.playsound("voice_gui_do_1.mp3")
    if(SoTu == 2):
        playsound.playsound("voice_gui_do_2.mp3")
    if(SoTu == 3):
        playsound.playsound("voice_gui_do_3.mp3")
    if(SoTu == 4):
        playsound.playsound("voice_gui_do_4.mp3")

def voice_lay_do(SoTu):
    if(SoTu == 1):
        playsound.playsound("voice_lay_do_1.mp3")
    if(SoTu == 2):
        playsound.playsound("voice_lay_do_2.mp3")
    if(SoTu == 3):
        playsound.playsound("voice_lay_do_3.mp3")
    if(SoTu == 4):
        playsound.playsound("voice_lay_do_4.mp3")
def speed_to_Text():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Recognizing...")
        audio_data = r.record(source,duration)
    try:
        text = r.recognize_google(audio_data,language="vi")
    except:
        text = ""
    print(text)
    return text

######################################
def text_to_Speed(text):
    tts = gTTS(text=text,lang="vi")
    filename = "voice.mp3"
    tts.save(filename)
    playsound.playsound(filename)

def check_Record():
    conn = sqlite3.connect("database.db")
    for SoTu in range(1,5):
        query = "SELECT * FROM data WHERE SoTu = " + str(SoTu)
        cusror = conn.execute(query)
        isRecordExist = 0
        for row in cusror:
            isRecordExist = 1
        if(isRecordExist == 0):
            conn.commit()
            conn.close() 
            print(SoTu)
            return SoTu
    conn.commit()
    conn.close() 
    print("0")
    return 0

#Insert vao Database, lay du lieu nhan dien
def check_DataSet():
    pathss, dirs, files = next(os.walk("dataSet"))
    file_count = len(files)
    if(file_count == 0):
        return False
    else:
        return True
#############################################

def insertRecord(SoTu):
    conn = sqlite3.connect("database.db")
    query = "SELECT * FROM data WHERE SoTu = " + str(SoTu)
    cusror = conn.execute(query)
    isRecordExist = 0
    for row in cusror:
        isRecordExist = 1
    if(isRecordExist == 0):
        query = "INSERT INTO data (SoTu,ThoiGian) VALUES (" + str(SoTu) + ",'" + str(datetime.now().strftime("%d.%m.%Y %H.%M")) + "')"
    else:
        query = "UPDATE data SET ThoiGian = '"+ str(datetime.now().strftime("%d.%m.%Y %H.%M")) +"' WHERE SoTu =" + str(SoTu)
    conn.execute(query)
    conn.commit()
    conn.close() 

def get_Face(SoTu):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    sampleNum = 0
    time_out = time.time() + 10
    while(True):
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray,1.3,5)
        for(x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            if not os.path.exists("dataSet"):
                os.makedirs("dataSet")
            sampleNum += 1
            cv2.imwrite("dataSet/TuSo."+str(SoTu)+"."+str(sampleNum)+".jpg",gray[y : y+h,x : x+w])
        #cv2.imshow("frame",frame)
        cv2.waitKey(1)
        if sampleNum > 10:
            if not os.path.exists("dataFace"):
                os.makedirs("dataFace")
            cv2.imwrite("dataFace/TuSo."+str(SoTu) + "_" + str(datetime.now().strftime("%d.%m.%Y %H.%M")) + ".jpg",frame[y-80 : y+h+80,x-60 : x+w+60])
            cap.release()
            cv2.destroyAllWindows()
            print("Lấy data thành công")
            return True
        if time.time() > time_out :
            cap.release()
            cv2.destroyAllWindows()
            print("Lấy data thất bại")
            return False
    

#Xoa ban ghi trong database, xoa anh nhan dien
def deleteRecord(SoTu):
    conn = sqlite3.connect("database.db")
    query = "DELETE FROM data WHERE SoTu = " + str(SoTu)
    conn.execute(query)
    conn.commit()
    conn.close() 

    path = "dataSet"
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
    for imagePath in imagePaths:
        if(SoTu == int(imagePath.split("\\")[1].split(".")[1])):
            os.remove(imagePath)

#########################
def getProfile(SoTu):
    conn = sqlite3.connect("database.db")
    query = "SELECT * FROM data WHERE SoTu = " + str(SoTu)
    cusror = conn.execute(query)
    profile = None
    for row in cusror:
        profile = row
    conn.close()
    return profile
#####################################
def train_Data():
    path = "dataSet"
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
    pathss, dirs, files = next(os.walk("dataSet"))
    file_count = len(files)
    if(file_count == 0):
        return False
    else:
        faces = []
        SoTus = []
        for imagePath in imagePaths:
            faceImg = Image.open(imagePath).convert("L")
            faceNp = np.array(faceImg,"uint8")
            #print(faceNp)
            SoTu = int(imagePath.split("\\")[1].split(".")[1])
            faces.append(faceNp)
            SoTus.append(SoTu)

        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.train(faces, np.array(SoTus))
        if not os.path.exists("recoginizer"):
            os.makedirs("recoginizer")
        recognizer.save("recoginizer/trainningData.yml")
        return True
##########################################################
def nhanDien():
    print("Nhan dien")
    if(check_DataSet() == False):
        return 0
    else:
        train_Data()
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read("recoginizer/trainningData.yml")
        cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray,1.3,5)
        for(x,y,w,h) in faces:
            roi_gray = gray[y:y+h,x:x+w]
            SoTu,confidence = recognizer.predict(roi_gray)
            if confidence < Confidence:
                cap.release()
                return SoTu
            else:
                cap.release()
                return 0
######################################
######################################
class capture_video(QThread):
    signal = pyqtSignal(np.ndarray)
    def __init__(self, index):
        self.index = index
        print("start threading", self.index)
        super(capture_video, self).__init__()

    def run(self):
        cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)     
        while True:
            ret, cv_img = cap.read()
            if ret:
                self.signal.emit(cv_img)

    def stop(self):
        print("stop threading", self.index)
        self.terminate()
#############################################
class Speed_Reco(QThread):
    signal = pyqtSignal(str)
    def __init__(self, index):
        self.index = index
        print("start threading", self.index)
        super(Speed_Reco, self).__init__()

    def run(self):   
        while(True):
            text = speed_to_Text()
            self.signal.emit(text)

    def stop(self):
        print("stop threading", self.index)
        self.terminate()
############################################
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.uic = Ui_MainWindow()
        self.uic.setupUi(self)
        self.thread = {}

    def closeEvent(self, event):
        self.stop_capture_video()

    def stop_capture_video(self):
        self.thread[1].stop()
    
    def stop_long_Task(self):
        self.thread[1].stop()

    #def start_capture_video(self):
     #   self.thread[1] = capture_video(index=1)
     #   self.thread[1].start()
     #   self.thread[1].signal.connect(self.show_wedcam)

    def show_wedcam(self, cv_img):
        qt_img = self.convert_cv_qt(cv_img)
        self.uic.label_Camera.setPixmap(qt_img)

    def convert_cv_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(800, 600, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

###################################################
    def show_Console(self,text):
        self.uic.label_display.setText("Console..!!")
        if(text == ""):
            self.uic.label_display.setText("Đang nhận dạng...")
        else:
            self.uic.label_display.setText(text)
############################################################
    def task_main(self,text):
        self.show_Console(text)
        if(text=="gửi đồ"):
            self.show_Console(text)
            SoTu = check_Record()
            print(SoTu)
            if(SoTu == 0):
                print("Xin lỗi, tủ đã đầy")
                playsound.playsound("voice_2.mp3")
            else:
                print("Mời bạn nhận diện khuôn mặt")
                playsound.playsound("voice_1.mp3")
                print(nhanDien())
                if(nhanDien() == 0):
                    if(get_Face(SoTu) == True):
                        insertRecord(SoTu)
                        print("Nhận diện thành công")
                        playsound.playsound("voice_3.mp3")
                        voice_gui_do(SoTu)
                    else:
                        print("Vui lòng thử lại")
                        playsound.playsound("voice_5.mp3")
                else:
                    print("Vui lòng thử lại")
                    playsound.playsound("voice_5.mp3")

        elif (text == "lấy đồ"):
            self.show_Console(text)
            SoTu = nhanDien()
            if(SoTu == 0):
                print("Vui lòng thử lại")
                playsound.playsound("voice_5.mp3")
            else:
                voice_lay_do(SoTu)
                deleteRecord(SoTu)
        else :
            self.show_Console(text)
            print(text)
    def run_Task(self):
        self.thread[1] = Speed_Reco(index=1)
        self.thread[1].start()
        self.thread[1].signal.connect(self.task_main)
##########################################################
if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_win = MainWindow()
    main_win.show()
    #main_win.start_capture_video()
    main_win.show_Console
    main_win.run_Task()
    #main_win.runLongTask()
    sys.exit(app.exec())