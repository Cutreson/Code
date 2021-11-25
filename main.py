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
#from Data import *
#from Trainning import *
###########################################
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("recoginizer/trainningData.yml")
#############################################
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
            frame = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            if ret:
                self.signal.emit(frame)

    def stop(self):
        print("stop threading", self.index)
        self.terminate()
#############################################
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
        self.thread[2].stop()
    
    def stop_long_Task(self):
        self.thread[2].stop()

    def start_capture_video(self):
        self.thread[1] = capture_video(index=1)
        self.thread[1].start()
        self.thread[1].signal.connect(self.show_wedcam)

    def show_wedcam(self, frame):
        qt_img = self.convert_cv_qt(frame)
        self.uic.label_Camera.setPixmap(qt_img)

    def convert_cv_qt(self, frame):
        rgb_image = frame
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(800, 600, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)
##############################################
    def getProfile(SoTu):
        conn = sqlite3.connect("database.db")
        query = "SELECT * FROM data WHERE SoTu = " + str(SoTu)
        cusror = conn.execute(query)
        profile = None
        for row in cusror:
            profile = row
        conn.close()
        return profile

    def nhanDien(self,frame):
        gray = frame
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
            else:
                print("False")

    def runLongTask(self):
        self.thread[2] = capture_video(index=2)
        self.thread[2].start()
        self.thread[2].signal.connect(self.nhanDien)
##############################################
if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_win = MainWindow()
    main_win.show()
    main_win.start_capture_video()
    main_win.runLongTask()
    sys.exit(app.exec())