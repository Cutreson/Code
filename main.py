import sys
from typing import Any
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
#from Data import *
#from Trainning import *
###########################################
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("recoginizer/trainningData.yml")
cap = cv2.VideoCapture(0,cv2.CAP_DSHOW) 
ret, frame = cap.read()
gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
#faces = face_cascade.detectMultiScale(gray,1.3,5)
#############################################
class capture_video(QThread):
    signal = pyqtSignal(np.ndarray)
    def __init__(self, index):
        self.index = index
        print("start threading", self.index)
        super(capture_video, self).__init__()

    def run(self):    
        while True:
            ret, cv_img = cap.read()
            #gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 4)
            for (x, y, w, h) in faces:
                cv2.rectangle(cv_img, (x, y), (x+w, y+h), (0, 0, 255), 2)
            if ret:
                self.signal.emit(cv_img)

    def stop(self):
        print("stop threading", self.index)
        self.terminate()
#############################################
class Worker(QThread):
    signal = pyqtSignal(np.ndarray)
    def __init__(self, index):
        self.index = index
        print("start threading", self.index)
        super(Worker, self).__init__()

    def getProfile(SoTu):
        conn = sqlite3.connect("database.db")
        query = "SELECT * FROM data WHERE SoTu = " + str(SoTu)
        cusror = conn.execute(query)
        profile = None
        for row in cusror:
            profile = row
        conn.close()
        return profile

    def nhanDien(self):
        #cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
        #ret, frame = cap.read()
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray,1.3,5)
        for(x,y,w,h) in faces:
            roi_gray = gray[y:y+h,x:x+w]
            SoTu,confidence = recognizer.predict(roi_gray)
            if confidence < 70:
                conn = sqlite3.connect("database.db")
                query = "SELECT * FROM data WHERE SoTu = " + str(SoTu)
                cusror = conn.execute(query)
                profile = None
                for row in cusror:
                    profile = row
                conn.close()
                if(profile != None):
                    print(SoTu)
                    cap.release()
            else:
                print("False")
                cap.release()

    def stop(self):
        print("stop threading", self.index)
        self.terminate()

    def run(self):
        while(True):
            print("Lon")
            sleep(1)
            self.nhanDien()
############################################
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.uic = Ui_MainWindow()
        self.uic.setupUi(self)
        self.thread = {}

    def closeEvent(self, event):
        self.stop_capture_video()
        self.stop_long_Task()

    def stop_capture_video(self):
        self.thread[1].stop()
    
    def stop_long_Task(self):
        self.thread[2].stop()

    def start_capture_video(self):
        self.thread[1] = capture_video(index=1)
        self.thread[1].start()
        self.thread[1].signal.connect(self.show_wedcam)

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

    def runLongTask(self):
        self.thread[2] = Worker(index=2)
        self.thread[2].start()
##############################################
if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_win = MainWindow()
    main_win.show()
    main_win.start_capture_video()
    main_win.runLongTask()
    sys.exit(app.exec())