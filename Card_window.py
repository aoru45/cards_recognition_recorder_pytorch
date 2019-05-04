import sys

from PyQt5.QtWidgets import QApplication,QMainWindow,QWidget,QMessageBox
from PyQt5 import QtCore
_translate = QtCore.QCoreApplication.translate
from card_recorder import *

class MyWindow(QMainWindow,Ui_MainWindow):
    def __init__(self):
        super(MyWindow,self).__init__()
        self.setupUi(self)
    def message(self):
        reply = QMessageBox.information(self,                         #使用infomation信息框  
                                        "标题",  
                                        "消息",  
                                        QMessageBox.Yes | QMessageBox.No) 

class Recorder():
    def __init__(self,window):
        self.window = window
        self.cards = {
            "card_3":[],
            "card_4":[],
            "card_5":[],
            "card_6":[],
            "card_7":[],
            "card_8":[],
            "card_9":[],
            "card_10":[],
            "card_j":[],
            "card_q":[],
            "card_k":[],
            "card_a":[],
            "card_2":[]
        }
        self.recg_cards = []
    def add(self,card):# 給number，添加一張牌
            
            number = card.split("_")[-1] # 3
            self.window.statusbar.showMessage("Card {} detected!".format(card))
            if not card in self.cards["card_{}".format(number)]:
                idx = len(self.cards["card_{}".format(number)]) + 1
                self.cards["card_{}".format(number)].append(card)
                eval("self.window.label_{}_{}.setStyleSheet('image: url(:/cards/pukeImage/{}.jpg)')".format(number,idx,card))
                eval("self.window.label_{}.setText(_translate('MainWindow', '<html><head/><body><p align=\"center\"><span style=\" font-size:12pt; font-weight:600; color:#ff0000;\">{}</span></p></body></html>'))".format(number,4-idx))
        
if __name__ == '__main__':
    '''
    主函数
    '''

    app = QApplication(sys.argv)
    mainWindow = MyWindow()
    mainWindow.show()
    recorder = Recorder(mainWindow)
    recorder.add("diamond_3")
    recorder.add("diamond_4")
    sys.exit(app.exec_())
