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
        MainWindow.resize(992, 627)
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
        self.label.setGeometry(QtCore.QRect(0, 20, 971, 61))
        font = QtGui.QFont()
        font.setPointSize(25)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.btn_Tu_1 = QtWidgets.QPushButton(self.tab_Home)
        self.btn_Tu_1.setGeometry(QtCore.QRect(50, 350, 111, 51))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.btn_Tu_1.setFont(font)
        self.btn_Tu_1.setStyleSheet("background-color: rgb(170, 255, 255);\n"
"border-color: rgb(255, 0, 0);")
        self.btn_Tu_1.setObjectName("btn_Tu_1")
        self.label_Tu_1 = QtWidgets.QLabel(self.tab_Home)
        self.label_Tu_1.setGeometry(QtCore.QRect(10, 140, 201, 201))
        font = QtGui.QFont()
        font.setPointSize(20)
        font.setBold(True)
        font.setWeight(75)
        self.label_Tu_1.setFont(font)
        self.label_Tu_1.setStyleSheet("background-color: rgb(0, 255, 0);")
        self.label_Tu_1.setAlignment(QtCore.Qt.AlignCenter)
        self.label_Tu_1.setObjectName("label_Tu_1")
        self.btn_Tu_2 = QtWidgets.QPushButton(self.tab_Home)
        self.btn_Tu_2.setGeometry(QtCore.QRect(300, 350, 111, 51))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.btn_Tu_2.setFont(font)
        self.btn_Tu_2.setStyleSheet("background-color: rgb(170, 255, 255);\n"
"border-color: rgb(255, 0, 0);")
        self.btn_Tu_2.setObjectName("btn_Tu_2")
        self.label_Tu_2 = QtWidgets.QLabel(self.tab_Home)
        self.label_Tu_2.setGeometry(QtCore.QRect(260, 140, 201, 201))
        font = QtGui.QFont()
        font.setPointSize(20)
        font.setBold(True)
        font.setWeight(75)
        self.label_Tu_2.setFont(font)
        self.label_Tu_2.setStyleSheet("background-color: rgb(0, 255, 0);")
        self.label_Tu_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_Tu_2.setObjectName("label_Tu_2")
        self.btn_Tu_3 = QtWidgets.QPushButton(self.tab_Home)
        self.btn_Tu_3.setGeometry(QtCore.QRect(790, 350, 111, 51))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.btn_Tu_3.setFont(font)
        self.btn_Tu_3.setStyleSheet("background-color: rgb(170, 255, 255);\n"
"border-color: rgb(255, 0, 0);")
        self.btn_Tu_3.setObjectName("btn_Tu_3")
        self.btn_Tu_4 = QtWidgets.QPushButton(self.tab_Home)
        self.btn_Tu_4.setGeometry(QtCore.QRect(540, 350, 111, 51))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.btn_Tu_4.setFont(font)
        self.btn_Tu_4.setStyleSheet("background-color: rgb(170, 255, 255);\n"
"border-color: rgb(255, 0, 0);")
        self.btn_Tu_4.setObjectName("btn_Tu_4")
        self.label_Tu_4 = QtWidgets.QLabel(self.tab_Home)
        self.label_Tu_4.setGeometry(QtCore.QRect(750, 140, 201, 201))
        font = QtGui.QFont()
        font.setPointSize(20)
        font.setBold(True)
        font.setWeight(75)
        self.label_Tu_4.setFont(font)
        self.label_Tu_4.setStyleSheet("background-color: rgb(0, 255, 0);")
        self.label_Tu_4.setAlignment(QtCore.Qt.AlignCenter)
        self.label_Tu_4.setObjectName("label_Tu_4")
        self.label_Tu_3 = QtWidgets.QLabel(self.tab_Home)
        self.label_Tu_3.setGeometry(QtCore.QRect(510, 140, 201, 201))
        font = QtGui.QFont()
        font.setPointSize(20)
        font.setBold(True)
        font.setWeight(75)
        self.label_Tu_3.setFont(font)
        self.label_Tu_3.setStyleSheet("background-color: rgb(0, 255, 0);")
        self.label_Tu_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_Tu_3.setObjectName("label_Tu_3")
        self.label_display = QtWidgets.QLabel(self.tab_Home)
        self.label_display.setGeometry(QtCore.QRect(0, 470, 981, 111))
        font = QtGui.QFont()
        font.setPointSize(25)
        font.setBold(True)
        font.setWeight(75)
        self.label_display.setFont(font)
        self.label_display.setStyleSheet("background-color: rgb(255, 255, 0);\n"
"color: rgb(255, 0, 0);")
        self.label_display.setAlignment(QtCore.Qt.AlignCenter)
        self.label_display.setObjectName("label_display")
        self.tabWidget.addTab(self.tab_Home, "")
        self.tab_History = QtWidgets.QWidget()
        self.tab_History.setObjectName("tab_History")
        self.label_2 = QtWidgets.QLabel(self.tab_History)
        self.label_2.setGeometry(QtCore.QRect(0, 0, 971, 61))
        font = QtGui.QFont()
        font.setPointSize(25)
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.btn_view_Img = QtWidgets.QPushButton(self.tab_History)
        self.btn_view_Img.setGeometry(QtCore.QRect(90, 460, 111, 51))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.btn_view_Img.setFont(font)
        self.btn_view_Img.setStyleSheet("background-color: rgb(170, 255, 255);\n"
"border-color: rgb(255, 0, 0);")
        self.btn_view_Img.setObjectName("btn_view_Img")
        self.btn_delete = QtWidgets.QPushButton(self.tab_History)
        self.btn_delete.setGeometry(QtCore.QRect(760, 460, 111, 51))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.btn_delete.setFont(font)
        self.btn_delete.setStyleSheet("background-color: rgb(170, 255, 255);\n"
"border-color: rgb(255, 0, 0);")
        self.btn_delete.setObjectName("btn_delete")
        self.table_Data = QtWidgets.QTableWidget(self.tab_History)
        self.table_Data.setEnabled(True)
        self.table_Data.setGeometry(QtCore.QRect(90, 80, 781, 351))
        font = QtGui.QFont()
        font.setFamily("MS Shell Dlg 2")
        font.setPointSize(20)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.table_Data.setFont(font)
        self.table_Data.setStyleSheet("font: 20pt \"MS Shell Dlg 2\";\n"
"border-color: rgb(0, 0, 0);\n"
"border-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0, stop:0 rgba(0, 0, 0, 255), stop:1 rgba(255, 255, 255, 255));")
        self.table_Data.setLineWidth(1)
        self.table_Data.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustIgnored)
        self.table_Data.setAutoScroll(True)
        self.table_Data.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        self.table_Data.setTextElideMode(QtCore.Qt.ElideMiddle)
        self.table_Data.setShowGrid(True)
        self.table_Data.setGridStyle(QtCore.Qt.SolidLine)
        self.table_Data.setCornerButtonEnabled(True)
        self.table_Data.setRowCount(30)
        self.table_Data.setObjectName("table_Data")
        self.table_Data.setColumnCount(4)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignCenter)
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        item.setFont(font)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        item.setForeground(brush)
        self.table_Data.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignCenter)
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        item.setFont(font)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        item.setForeground(brush)
        self.table_Data.setHorizontalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignCenter)
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        item.setFont(font)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        item.setForeground(brush)
        self.table_Data.setHorizontalHeaderItem(2, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignCenter)
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        item.setFont(font)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        item.setForeground(brush)
        self.table_Data.setHorizontalHeaderItem(3, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignCenter)
        self.table_Data.setItem(0, 0, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignCenter)
        self.table_Data.setItem(0, 1, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignCenter)
        self.table_Data.setItem(0, 2, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_Data.setItem(1, 0, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_Data.setItem(2, 0, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_Data.setItem(3, 0, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_Data.setItem(4, 0, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_Data.setItem(5, 0, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_Data.setItem(6, 0, item)
        item = QtWidgets.QTableWidgetItem()
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        item.setFont(font)
        self.table_Data.setItem(6, 2, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_Data.setItem(7, 0, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_Data.setItem(8, 0, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_Data.setItem(9, 0, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_Data.setItem(10, 0, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_Data.setItem(11, 0, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_Data.setItem(12, 0, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_Data.setItem(13, 0, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_Data.setItem(14, 0, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_Data.setItem(15, 0, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_Data.setItem(16, 0, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_Data.setItem(17, 0, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_Data.setItem(18, 0, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_Data.setItem(19, 0, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_Data.setItem(20, 0, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_Data.setItem(21, 0, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_Data.setItem(22, 0, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_Data.setItem(23, 0, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_Data.setItem(24, 0, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_Data.setItem(25, 0, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_Data.setItem(26, 0, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_Data.setItem(27, 0, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_Data.setItem(28, 0, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_Data.setItem(29, 0, item)
        self.table_Data.horizontalHeader().setVisible(False)
        self.table_Data.horizontalHeader().setCascadingSectionResizes(False)
        self.table_Data.horizontalHeader().setDefaultSectionSize(190)
        self.table_Data.horizontalHeader().setHighlightSections(True)
        self.table_Data.horizontalHeader().setMinimumSectionSize(30)
        self.table_Data.horizontalHeader().setSortIndicatorShown(False)
        self.table_Data.horizontalHeader().setStretchLastSection(False)
        self.table_Data.verticalHeader().setVisible(False)
        self.table_Data.verticalHeader().setCascadingSectionResizes(False)
        self.table_Data.verticalHeader().setDefaultSectionSize(30)
        self.table_Data.verticalHeader().setMinimumSectionSize(30)
        self.tabWidget.addTab(self.tab_History, "")
        self.gridLayout.addWidget(self.tabWidget, 0, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(1)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Tủ gửi đồ thông minh"))
        self.label.setText(_translate("MainWindow", "Tủ gửi đồ thông minh sử dụng trí tuệ nhân tạo"))
        self.btn_Tu_1.setText(_translate("MainWindow", "Mở khóa"))
        self.label_Tu_1.setText(_translate("MainWindow", "Tủ số 1"))
        self.btn_Tu_2.setText(_translate("MainWindow", "Mở khóa"))
        self.label_Tu_2.setText(_translate("MainWindow", "Tủ số 2"))
        self.btn_Tu_3.setText(_translate("MainWindow", "Mở khóa"))
        self.btn_Tu_4.setText(_translate("MainWindow", "Mở khóa"))
        self.label_Tu_4.setText(_translate("MainWindow", "Tủ số 4"))
        self.label_Tu_3.setText(_translate("MainWindow", "Tủ số 3"))
        self.label_display.setText(_translate("MainWindow", "Console"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_Home), _translate("MainWindow", "Trang chủ"))
        self.label_2.setText(_translate("MainWindow", "Danh sách những người quên chưa lấy đồ"))
        self.btn_view_Img.setText(_translate("MainWindow", "Xem ảnh"))
        self.btn_delete.setText(_translate("MainWindow", "Xóa dữ liệu"))
        self.table_Data.setSortingEnabled(False)
        item = self.table_Data.horizontalHeaderItem(0)
        item.setText(_translate("MainWindow", "STT"))
        item = self.table_Data.horizontalHeaderItem(1)
        item.setText(_translate("MainWindow", "Số tủ"))
        item = self.table_Data.horizontalHeaderItem(2)
        item.setText(_translate("MainWindow", "Ngày gửi"))
        item = self.table_Data.horizontalHeaderItem(3)
        item.setText(_translate("MainWindow", "Giờ gửi"))
        __sortingEnabled = self.table_Data.isSortingEnabled()
        self.table_Data.setSortingEnabled(False)
        item = self.table_Data.item(0, 0)
        item.setText(_translate("MainWindow", "1"))
        item = self.table_Data.item(0, 1)
        item.setText(_translate("MainWindow", "2"))
        item = self.table_Data.item(0, 2)
        item.setText(_translate("MainWindow", "3"))
        item = self.table_Data.item(1, 0)
        item.setText(_translate("MainWindow", "2"))
        item = self.table_Data.item(2, 0)
        item.setText(_translate("MainWindow", "3"))
        item = self.table_Data.item(3, 0)
        item.setText(_translate("MainWindow", "4"))
        item = self.table_Data.item(4, 0)
        item.setText(_translate("MainWindow", "5"))
        item = self.table_Data.item(5, 0)
        item.setText(_translate("MainWindow", "6"))
        item = self.table_Data.item(6, 0)
        item.setText(_translate("MainWindow", "7"))
        item = self.table_Data.item(7, 0)
        item.setText(_translate("MainWindow", "8"))
        item = self.table_Data.item(8, 0)
        item.setText(_translate("MainWindow", "9"))
        item = self.table_Data.item(9, 0)
        item.setText(_translate("MainWindow", "10"))
        item = self.table_Data.item(10, 0)
        item.setText(_translate("MainWindow", "11"))
        item = self.table_Data.item(11, 0)
        item.setText(_translate("MainWindow", "12"))
        item = self.table_Data.item(12, 0)
        item.setText(_translate("MainWindow", "13"))
        item = self.table_Data.item(13, 0)
        item.setText(_translate("MainWindow", "14"))
        item = self.table_Data.item(14, 0)
        item.setText(_translate("MainWindow", "15"))
        item = self.table_Data.item(15, 0)
        item.setText(_translate("MainWindow", "16"))
        item = self.table_Data.item(16, 0)
        item.setText(_translate("MainWindow", "17"))
        item = self.table_Data.item(17, 0)
        item.setText(_translate("MainWindow", "18"))
        item = self.table_Data.item(18, 0)
        item.setText(_translate("MainWindow", "19"))
        item = self.table_Data.item(19, 0)
        item.setText(_translate("MainWindow", "20"))
        item = self.table_Data.item(20, 0)
        item.setText(_translate("MainWindow", "21"))
        item = self.table_Data.item(21, 0)
        item.setText(_translate("MainWindow", "22"))
        item = self.table_Data.item(22, 0)
        item.setText(_translate("MainWindow", "23"))
        item = self.table_Data.item(23, 0)
        item.setText(_translate("MainWindow", "24"))
        item = self.table_Data.item(24, 0)
        item.setText(_translate("MainWindow", "25"))
        item = self.table_Data.item(25, 0)
        item.setText(_translate("MainWindow", "26"))
        item = self.table_Data.item(26, 0)
        item.setText(_translate("MainWindow", "27"))
        item = self.table_Data.item(27, 0)
        item.setText(_translate("MainWindow", "28"))
        item = self.table_Data.item(28, 0)
        item.setText(_translate("MainWindow", "29"))
        item = self.table_Data.item(29, 0)
        item.setText(_translate("MainWindow", "30"))
        self.table_Data.setSortingEnabled(__sortingEnabled)
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_History), _translate("MainWindow", "Lịch sử"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
