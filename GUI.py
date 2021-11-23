# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'GUI.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1000, 650)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.tabWidget.setFont(font)
        self.tabWidget.setStyleSheet("color: rgb(0, 0, 255);")
        self.tabWidget.setObjectName("tabWidget")
        self.tab_Home = QtWidgets.QWidget()
        self.tab_Home.setObjectName("tab_Home")
        self.label = QtWidgets.QLabel(self.tab_Home)
        self.label.setGeometry(QtCore.QRect(170, 20, 641, 61))
        font = QtGui.QFont()
        font.setPointSize(20)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.label_Camera = QtWidgets.QLabel(self.tab_Home)
        self.label_Camera.setGeometry(QtCore.QRect(70, 130, 400, 400))
        self.label_Camera.setMinimumSize(QtCore.QSize(400, 0))
        self.label_Camera.setMaximumSize(QtCore.QSize(400, 400))
        font = QtGui.QFont()
        font.setPointSize(20)
        font.setBold(True)
        font.setWeight(75)
        self.label_Camera.setFont(font)
        self.label_Camera.setStyleSheet("background-color: rgb(0, 255, 0);")
        self.label_Camera.setAlignment(QtCore.Qt.AlignCenter)
        self.label_Camera.setObjectName("label_Camera")
        self.label_Tu_1 = QtWidgets.QLabel(self.tab_Home)
        self.label_Tu_1.setGeometry(QtCore.QRect(530, 130, 150, 150))
        self.label_Tu_1.setMinimumSize(QtCore.QSize(150, 0))
        self.label_Tu_1.setMaximumSize(QtCore.QSize(150, 150))
        font = QtGui.QFont()
        font.setPointSize(20)
        font.setBold(True)
        font.setWeight(75)
        self.label_Tu_1.setFont(font)
        self.label_Tu_1.setStyleSheet("background-color: rgb(0, 255, 0);")
        self.label_Tu_1.setAlignment(QtCore.Qt.AlignCenter)
        self.label_Tu_1.setObjectName("label_Tu_1")
        self.btn_Tu_1 = QtWidgets.QPushButton(self.tab_Home)
        self.btn_Tu_1.setGeometry(QtCore.QRect(550, 290, 111, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.btn_Tu_1.setFont(font)
        self.btn_Tu_1.setStyleSheet("background-color: rgb(170, 255, 255);\n"
"border-color: rgb(255, 0, 0);")
        self.btn_Tu_1.setObjectName("btn_Tu_1")
        self.label_Tu_2 = QtWidgets.QLabel(self.tab_Home)
        self.label_Tu_2.setGeometry(QtCore.QRect(740, 130, 150, 150))
        self.label_Tu_2.setMinimumSize(QtCore.QSize(150, 0))
        self.label_Tu_2.setMaximumSize(QtCore.QSize(150, 150))
        font = QtGui.QFont()
        font.setPointSize(20)
        font.setBold(True)
        font.setWeight(75)
        self.label_Tu_2.setFont(font)
        self.label_Tu_2.setStyleSheet("background-color: rgb(0, 255, 0);")
        self.label_Tu_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_Tu_2.setObjectName("label_Tu_2")
        self.btn_Tu_2 = QtWidgets.QPushButton(self.tab_Home)
        self.btn_Tu_2.setGeometry(QtCore.QRect(760, 290, 111, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.btn_Tu_2.setFont(font)
        self.btn_Tu_2.setStyleSheet("background-color: rgb(170, 255, 255);\n"
"border-color: rgb(255, 0, 0);")
        self.btn_Tu_2.setObjectName("btn_Tu_2")
        self.label_Tu_3 = QtWidgets.QLabel(self.tab_Home)
        self.label_Tu_3.setGeometry(QtCore.QRect(530, 340, 150, 150))
        self.label_Tu_3.setMinimumSize(QtCore.QSize(150, 0))
        self.label_Tu_3.setMaximumSize(QtCore.QSize(150, 150))
        font = QtGui.QFont()
        font.setPointSize(20)
        font.setBold(True)
        font.setWeight(75)
        self.label_Tu_3.setFont(font)
        self.label_Tu_3.setStyleSheet("background-color: rgb(0, 255, 0);")
        self.label_Tu_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_Tu_3.setObjectName("label_Tu_3")
        self.btn_Tu_4 = QtWidgets.QPushButton(self.tab_Home)
        self.btn_Tu_4.setGeometry(QtCore.QRect(760, 500, 111, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.btn_Tu_4.setFont(font)
        self.btn_Tu_4.setStyleSheet("background-color: rgb(170, 255, 255);\n"
"border-color: rgb(255, 0, 0);")
        self.btn_Tu_4.setObjectName("btn_Tu_4")
        self.btn_Tu_3 = QtWidgets.QPushButton(self.tab_Home)
        self.btn_Tu_3.setGeometry(QtCore.QRect(550, 500, 111, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.btn_Tu_3.setFont(font)
        self.btn_Tu_3.setStyleSheet("background-color: rgb(170, 255, 255);\n"
"border-color: rgb(255, 0, 0);")
        self.btn_Tu_3.setObjectName("btn_Tu_3")
        self.label_Tu_4 = QtWidgets.QLabel(self.tab_Home)
        self.label_Tu_4.setGeometry(QtCore.QRect(740, 340, 150, 150))
        self.label_Tu_4.setMinimumSize(QtCore.QSize(150, 0))
        self.label_Tu_4.setMaximumSize(QtCore.QSize(150, 150))
        font = QtGui.QFont()
        font.setPointSize(20)
        font.setBold(True)
        font.setWeight(75)
        self.label_Tu_4.setFont(font)
        self.label_Tu_4.setStyleSheet("background-color: rgb(0, 255, 0);")
        self.label_Tu_4.setAlignment(QtCore.Qt.AlignCenter)
        self.label_Tu_4.setObjectName("label_Tu_4")
        self.tabWidget.addTab(self.tab_Home, "")
        self.tab_History = QtWidgets.QWidget()
        self.tab_History.setObjectName("tab_History")
        self.tabWidget.addTab(self.tab_History, "")
        self.gridLayout.addWidget(self.tabWidget, 0, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Tủ gửi đồ thông minh"))
        self.label.setText(_translate("MainWindow", "Tủ gửi đồ thông minh sử dụng trí tuệ nhân tạo"))
        self.label_Camera.setText(_translate("MainWindow", "Camera"))
        self.label_Tu_1.setText(_translate("MainWindow", "Tủ số 1"))
        self.btn_Tu_1.setText(_translate("MainWindow", "Mở khóa"))
        self.label_Tu_2.setText(_translate("MainWindow", "Tủ số 1"))
        self.btn_Tu_2.setText(_translate("MainWindow", "Mở khóa"))
        self.label_Tu_3.setText(_translate("MainWindow", "Tủ số 1"))
        self.btn_Tu_4.setText(_translate("MainWindow", "Mở khóa"))
        self.btn_Tu_3.setText(_translate("MainWindow", "Mở khóa"))
        self.label_Tu_4.setText(_translate("MainWindow", "Tủ số 1"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_Home), _translate("MainWindow", "Trang chủ"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_History), _translate("MainWindow", "Lịch sử"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())