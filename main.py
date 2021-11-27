import sys
import cv2
from time import sleep
import numpy as np
import sqlite3
from PIL import Image
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QSize, QThread, pyqtSignal, Qt, QObject
from PyQt5.QtGui import QFont, QIcon, QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QTableWidgetItem, QLabel
from GUI import Ui_MainWindow
import speech_recognition as sr
from gtts import gTTS
import os, shutil
import time
import playsound
from datetime import datetime
import serial
import serial.tools.list_ports

duration = 3
Confidence = 50
time_Open = 2
###########################################
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("recoginizer/trainningData.yml")
#############################################
def check_port():
    serPort = ""
    int1 = 0
    str1 = ""
    str2 = ""
    # Find Live Ports
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
    #print (p) # This causes each port's information to be printed out.
            # To search this p data, use p[1].
        while (int1 < 9):   # Loop checks "COM0" to "COM8" for Adruino Port Info. 
            if "CH340" in p[1]:  # Looks for "CH340" in P[1].
                    str2 = str(int1) # Converts an Integer to a String, allowing:
                    str1 = "COM" + str2 # add the strings together
            if "CH340" in p[1] and str1 in p[1]: # Looks for "CH340" and "COM#"
                print ("Found Arduino Uno on " + str1)
                int1 = 9 # Causes loop to end.
            if int1 == 8:
                print ("UNO not found!")
                sys.exit() # Terminates Script.
            int1 = int1 + 1
    return str1
port = serial.Serial(check_port(),9600)
##############################################
def voice_gui_do(SoTu):
    if(SoTu == 1):
        playsound.playsound("voice_gui_do_1.mp3")
        port.write(str.encode('1'))
        time.sleep(time_Open)
        port.write(str.encode('0'))
    if(SoTu == 2):
        playsound.playsound("voice_gui_do_2.mp3")
        port.write(str.encode('3'))
        time.sleep(time_Open)
        port.write(str.encode('2'))
    if(SoTu == 3):
        playsound.playsound("voice_gui_do_3.mp3")
        port.write(str.encode('5'))
        time.sleep(time_Open)
        port.write(str.encode('4'))
    if(SoTu == 4):
        playsound.playsound("voice_gui_do_4.mp3")
        port.write(str.encode('7'))
        time.sleep(time_Open)
        port.write(str.encode('6'))

def voice_lay_do(SoTu):
    if(SoTu == 1):
        playsound.playsound("voice_lay_do_1.mp3")
        port.write(str.encode('1'))
        time.sleep(time_Open)
        port.write(str.encode('0'))
    if(SoTu == 2):
        playsound.playsound("voice_lay_do_2.mp3")
        port.write(str.encode('3'))
        time.sleep(time_Open)
        port.write(str.encode('2'))
    if(SoTu == 3):
        playsound.playsound("voice_lay_do_3.mp3")
        port.write(str.encode('5'))
        time.sleep(time_Open)
        port.write(str.encode('4'))
    if(SoTu == 4):
        playsound.playsound("voice_lay_do_4.mp3")
        port.write(str.encode('7'))
        time.sleep(time_Open)
        port.write(str.encode('6'))
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
        cv2.imshow("frame",frame)
        cv2.waitKey(1)
        if sampleNum > 10:
            if not os.path.exists("dataFace"):
                os.makedirs("dataFace")
            cv2.imwrite("dataFace/TuSo."+str(SoTu) + ". " + str(datetime.now().strftime("%d.%m.%Y %Hh%M")) + ".jpg",frame[y : y+h,x : x+w])
            #cv2.imwrite("dataFace/TuSo."+str(SoTu) + ". " + str(datetime.now().strftime("%d.%m.%Y %Hh%M")) + ".jpg",frame[y-80 : y+h+80,x-60 : x+w+60])
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

def delete_dataFace(SoTu):
    path = "dataFace"
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
            if(text=="gửi đồ"):
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
                SoTu = nhanDien()
                if(SoTu == 0):
                    print("Vui lòng thử lại")
                    playsound.playsound("voice_5.mp3")
                else:
                    print("Nhận diện thành công")
                    playsound.playsound("voice_3.mp3")
                    voice_lay_do(SoTu)
                    deleteRecord(SoTu)
                    delete_dataFace(SoTu)
            else :
                print(text)

    def stop(self):
        print("stop threading", self.index)
        self.terminate()
############################################
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.uic = Ui_MainWindow()
        self.uic.setupUi(self)
        self.delete_Table()
        self.thread = {}
########################################################
    
    def btn_delete_All_Table(self):
        path = "History"
        imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
        for imagePath in imagePaths: 
            os.remove(imagePath)
        self.delete_Table()

    def btn_delete_Row(self):
        self.delete_Table()
        row = self.uic.table_Data.currentRow()
        print(row)
        path = "History"
        imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
        pathss, dirs, files = next(os.walk(path))
        file_count = len(files)
        if(row == 0):
            os.remove(imagePaths[0])
            self.show_Table()
            return
        if(file_count != 0 & row != -1 & row < file_count):
            os.remove(imagePaths[row])
            self.show_Table()
            return
        

    def btn_Tu_1(self):
        self.uic.label_Tu_1.setText("Trống")
        port.write(str.encode('1'))
        time.sleep(time_Open)
        port.write(str.encode('0'))
        path = "dataFace"
        pathss, dirs, files = next(os.walk("dataFace"))
        file_count = len(files)
        if(file_count != 0):
            imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
            for imagePath in imagePaths:
                if(1 == int(imagePath.split("\\")[1].split(".")[1])):
                    shutil.move(imagePath, "History")
                    deleteRecord(1)
                    train_Data()
    def btn_Tu_2(self):
        self.uic.label_Tu_2.setText("Trống")
        port.write(str.encode('3'))
        time.sleep(time_Open)
        port.write(str.encode('2'))
        path = "dataFace"
        pathss, dirs, files = next(os.walk("dataFace"))
        file_count = len(files)
        if(file_count != 0):
            imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
            for imagePath in imagePaths:
                if(2 == int(imagePath.split("\\")[1].split(".")[1])):
                    shutil.move(imagePath, "History")
                    deleteRecord(2)
                    train_Data()    
    def btn_Tu_3(self):
        self.uic.label_Tu_3.setText("Trống")
        port.write(str.encode('5'))
        time.sleep(time_Open)
        port.write(str.encode('4'))
        path = "dataFace"
        pathss, dirs, files = next(os.walk("dataFace"))
        file_count = len(files)
        if(file_count != 0):
            imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
            for imagePath in imagePaths:
                if(3 == int(imagePath.split("\\")[1].split(".")[1])):
                    shutil.move(imagePath, "History")
                    deleteRecord(3)
                    train_Data()    
    def btn_Tu_4(self):
        self.uic.label_Tu_4.setText("Trống")
        port.write(str.encode('7'))
        time.sleep(time_Open)
        port.write(str.encode('6'))
        path = "dataFace"
        pathss, dirs, files = next(os.walk("dataFace"))
        file_count = len(files)
        if(file_count != 0):
            imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
            for imagePath in imagePaths:
                if(4 == int(imagePath.split("\\")[1].split(".")[1])):
                    shutil.move(imagePath, "History")
                    deleteRecord(4)
                    train_Data()       
##################################################################
    def delete_Table(self):
        for row in range(29):
            self.uic.table_Data.setItem(row,0,QTableWidgetItem(""))
            self.uic.table_Data.setItem(row,1,QTableWidgetItem(""))
            self.uic.table_Data.setItem(row,2,QTableWidgetItem(""))
    def show_Table(self):
        path = "History"
        imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
        pathss, dirs, files = next(os.walk(path))
        file_count = len(files)
        print(file_count)
        if(file_count == 0):
            return
        row = 0
        self.uic.table_Data.setIconSize(QSize(180,180))
        for imagePath in imagePaths:          
            item = QTableWidgetItem()
            item.setSizeHint(QSize(185,185))
            item.setIcon(QIcon(imagePath))
            self.uic.table_Data.setItem(row,0,item)
            self.uic.table_Data.setItem(row,1,QTableWidgetItem("Tủ số " + imagePath.split("\\")[1].split(".")[1]))
            self.uic.table_Data.setItem(row,2,QTableWidgetItem(imagePath.split("\\")[1].split(" ")[1] + " : " + imagePath.split("\\")[1].split(" ")[2].split(".")[0]))
            row+=1

##############################################################
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
            self.uic.label_display.setText("Nhận diện giọng nói : Đang nhận diện...")
        else:
            self.uic.label_display.setText("Nhận diện giọng nói : " + text)

############################################################
    def show_img1(self):
        path = "dataFace"
        pathss, dirs, files = next(os.walk("dataFace"))
        file_count = len(files)
        if(file_count == 0):
            self.uic.label_Tu_1.setText("Trống")
        imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
        for imagePath in imagePaths:
            if(1 == int(imagePath.split("\\")[1].split(".")[1])):
                pixmap_1 = QPixmap(imagePath)
                pixmap_1.scaled(self.uic.label_Tu_1.size(),QtCore.Qt.KeepAspectRatio)
                self.uic.label_Tu_1.setScaledContents(True)
                self.uic.label_Tu_1.setPixmap(pixmap_1)
                return
            else:
                self.uic.label_Tu_1.setText("Trống")
    def show_img2(self):
        path = "dataFace"
        pathss, dirs, files = next(os.walk("dataFace"))
        file_count = len(files)
        if(file_count == 0):
            self.uic.label_Tu_2.setText("Trống")
        imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
        for imagePath in imagePaths:
            if(2 == int(imagePath.split("\\")[1].split(".")[1])):
                pixmap_2 = QPixmap(imagePath)
                pixmap_2.scaled(self.uic.label_Tu_1.size(),QtCore.Qt.KeepAspectRatio)
                self.uic.label_Tu_2.setScaledContents(True)
                self.uic.label_Tu_2.setPixmap(pixmap_2)
                return
            else:
                self.uic.label_Tu_2.setText("Trống")
    def show_img3(self):
        path = "dataFace"
        pathss, dirs, files = next(os.walk("dataFace"))
        file_count = len(files)
        if(file_count == 0):
            self.uic.label_Tu_3.setText("Trống")
        imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
        for imagePath in imagePaths:
            if(3 == int(imagePath.split("\\")[1].split(".")[1])):
                pixmap_1 = QPixmap(imagePath)
                pixmap_1.scaled(self.uic.label_Tu_1.size(),QtCore.Qt.KeepAspectRatio)
                self.uic.label_Tu_3.setScaledContents(True)
                self.uic.label_Tu_3.setPixmap(pixmap_1)
                return
            else:
                self.uic.label_Tu_3.setText("Trống")
    def show_img4(self):
        path = "dataFace"
        pathss, dirs, files = next(os.walk("dataFace"))
        file_count = len(files)
        if(file_count == 0):
            self.uic.label_Tu_4.setText("Trống")
        imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
        for imagePath in imagePaths:
            if(4 == int(imagePath.split("\\")[1].split(".")[1])):
                pixmap_1 = QPixmap(imagePath)
                pixmap_1.scaled(self.uic.label_Tu_1.size(),QtCore.Qt.KeepAspectRatio)
                self.uic.label_Tu_4.setScaledContents(True)
                self.uic.label_Tu_4.setPixmap(pixmap_1)
                return
            else:
                self.uic.label_Tu_4.setText("Trống")
    def show_all_Img(self):
        self.show_img1()
        self.show_img2()
        self.show_img3()
        self.show_img4()
############################################################
    def task_main(self,text):
        self.show_Console(text)
        self.show_all_Img()
        self.show_Table()
    def run_Task(self):
        self.thread[1] = Speed_Reco(index=1)
        self.thread[1].start()
        self.thread[1].signal.connect(self.task_main)
##########################################################
if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_win = MainWindow()
    main_win.show()
    main_win.show_Console
    main_win.uic.btn_Tu_1.clicked.connect(main_win.btn_Tu_1)
    main_win.uic.btn_Tu_2.clicked.connect(main_win.btn_Tu_2)
    main_win.uic.btn_Tu_3.clicked.connect(main_win.btn_Tu_3)
    main_win.uic.btn_Tu_4.clicked.connect(main_win.btn_Tu_4)
    main_win.uic.btn_delete_One.clicked.connect(main_win.btn_delete_Row)
    main_win.uic.btn_delete_All.clicked.connect(main_win.btn_delete_All_Table)
    main_win.run_Task()
    sys.exit(app.exec())