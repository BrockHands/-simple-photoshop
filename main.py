from email.mime import image
import math
from multiprocessing.sharedctypes import Value
import sys
from tkinter import image_names
import cv2
from math import *
import numpy as np
from PIL import Image
from cv2 import COLOR_BGR2GRAY
from PySide6.QtGui import QAction, QImage, QPixmap,QIcon
from PySide6.QtCore import Qt,QDateTime,QSize
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QMainWindow, QSlider,QLineEdit, QTabWidget,
    QHBoxLayout, QVBoxLayout, QPushButton, QFileDialog
)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        #아이콘
        self.setWindowTitle('Icon')
        self.setWindowIcon(QIcon('web.png'))
        self.setWindowTitle('SimplePS')
        #시간표시
        self.date = QDateTime.currentDateTime()


        # 메뉴바 만들기
        
      #  self.menu = self.menuBar()
      #  self.menu_file = self.menu.addMenu("편집 열기")
      #  Bright = QAction("밝기", self, triggered=self.initui)
      #  self.menu_file.addAction(Bright)
       
        self.save_img = None

        

        ##하단바 시간표시
        self.statusBar().showMessage(self.date.toString('yyyy년mm월dd일 hh:mm'))
        
        #툴바 액션 모음

        #새로고침
        Refresh = QAction(QIcon('Refresh.png'), 'Refresh    Ctrl+R', self)
        Refresh.setShortcut('Ctrl+R')
        Refresh.setStatusTip('Refresh application')
        Refresh.triggered.connect(self.clear_label)  

        #이미지 열기
        openimage  = QAction(QIcon('open.png'), 'open   Ctrl+N', self)
        openimage .setShortcut('Ctrl+N')
        openimage .setStatusTip('Open image')
        openimage .triggered.connect(self.show_file_dialog)  

        #이미지 저장
        saveimage  = QAction(QIcon('save.png'), 'save   Ctrl+S', self)
        saveimage .setShortcut('Ctrl+S')
        saveimage .setStatusTip('Save image')
        saveimage .triggered.connect(self.save_file)  

        
        

        #툴바
        self.toolbar = self.addToolBar('Exit Tool')
        self.toolbar.addAction(openimage)
        self.toolbar.addAction(saveimage)
        self.toolbar.addAction(Refresh)

      
        
       

       
        # 메인화면 레이아웃
        main_layout = QHBoxLayout()

        # 슬라이더
        self.slider = QSlider(Qt.Horizontal, self)
        self.slider.move(300, 0)
        self.slider.setRange(0, 200)
        self.slider.setSingleStep(2)
        self.slider.setTickInterval(20)
        self.slider.setTickPosition(QSlider.TicksAbove)
        self.slider.valueChanged.connect(self.value_changed)

        self.label = QLabel(self)
        self.label.setGeometry(500, 0, 100, 30)
        self.label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.label.setStyleSheet("border-radius: 5px;"
                                  "border: 1px solid gray;"
                                  "background-color: #BBDEFB")

    

        # 사이드바 메뉴버튼
        

        sidebar = QVBoxLayout()
        
        #탭설정
        tab1 = QWidget()
        tab2 = QWidget()
        tab3 = QWidget()
        tab4 = QWidget()
        tab5 = QWidget()

        tabs = QTabWidget()
        
        tabs.addTab(tab1, '화소점 처리')
        tabs.addTab(tab2, '화소영역 처리')
        tabs.addTab(tab3, '기하학처리 처리')
        tabs.addTab(tab4, '윤곽선 처리')
        tabs.addTab(tab5 , '기타')

        tab1.layout = QVBoxLayout(tab1)
        tab2.layout = QVBoxLayout(tab2)
        tab3.layout = QVBoxLayout(tab3)
        tab4.layout = QVBoxLayout(tab4)
        tab5.layout = QVBoxLayout(tab5)
        sidebar.addWidget(tabs)

      
         ##버튼
        button4 = QPushButton("좌우반전") 
        button5 = QPushButton("상하반전")
        button6 = QPushButton("흑백변경")
        button8 = QPushButton("확대")
        button9 = QPushButton("축소")
        button10 = QPushButton("블러링")
        buttonem = QPushButton("엠보싱")
        button12 = QPushButton("반전")
        button13 = QPushButton("이진화")
        button14 = QPushButton("스트레칭")
        button16 = QPushButton("소벨")
        button17 = QPushButton("윤곽선 검출(가장자리 검출 이미지)")
        button18 = QPushButton("카툰")
        button19 = QPushButton("얼굴인식")
        button20 = QPushButton("왜곡 <0~100오목, 100~볼록>")
### 
        buttonX = QPushButton("아래 부터는 슬라이드를 조정후 누르세요.")
        button11 = QPushButton("밝게 만들기")
        button7 = QPushButton("회전")
        button15 = QPushButton("리사이즈")


        button4.clicked.connect(self.flip_image)
        button5.clicked.connect(self.flip_image2)
        button6.clicked.connect(self.gray_image)
        button7.clicked.connect(self.r45_image)
        button8.clicked.connect(self.zoom_image)
        button9.clicked.connect(self.small_image)
        button10.clicked.connect(self.blurring_image)
        button11.clicked.connect(self.add_image)
        button12.clicked.connect(self.nag_image)
        button13.clicked.connect(self.bin_image)
        button14.clicked.connect(self.str_image)
        buttonem.clicked.connect(self.embossImage)
        button15.clicked.connect(self.resize_image)
        button16.clicked.connect(self.sobel_image)
        button17.clicked.connect(self.contours_image)
        button18.clicked.connect(self.cartoon_image)
        button19.clicked.connect(self.face_image)
        button20.clicked.connect(self.distortion_image)
        
        #화소점 탭
        tab1.layout.addWidget(button12)
        tab1.layout.addWidget(button13)
        tab1.layout.addWidget(button14)
        tab1.layout.addWidget(buttonX)
        tab1.layout.addWidget(button11)

        #화소영역 탭
        tab2.layout.addWidget(button10)
        tab2.layout.addWidget(buttonem)
        
        #기하학처리 
        tab3.layout.addWidget(button4)
        tab3.layout.addWidget(button5)
        tab3.layout.addWidget(button8)
        tab3.layout.addWidget(button9)
        tab3.layout.addWidget(button20)
        tab3.layout.addWidget(button8)
        tab3.layout.addWidget(button9)
        tab3.layout.addWidget(button6)
        tab3.layout.addWidget(buttonX)
        tab3.layout.addWidget(button7)
        tab3.layout.addWidget(button15)       
        
        #윤곽선 처리
        tab4.layout.addWidget(button16)
        tab4.layout.addWidget(button17)
        
        #기타

        tab5.layout.addWidget(button18)
        tab5.layout.addWidget(button19)


     
        #레이아웃
        main_layout.addLayout(sidebar)
        

        self.label1 = QLabel(self)
        self.label1.setFixedSize(640,480)
        main_layout.addWidget(self.label1)

        self.label2 = QLabel(self)
        self.label2.setFixedSize(640, 480)
        main_layout.addWidget(self.label2)

        widget = QWidget(self)
        widget.setLayout(main_layout)
        self.setCentralWidget(widget)

        
 
        
    #슬라이드 연동 정수 값 표시
    def value_changed(self, value):
        self.label.setText(str(value))

       
    # define
     
    def show_file_dialog(self):
        global pixmap,h,w,_,pixmap3
        file_name = QFileDialog.getOpenFileName(self, "이미지 열기", "./")
        self.image = cv2.imread(file_name[0],)
        h, w, _ = self.image.shape
        bytes_per_line = 3 * w

        image = QImage(
            self.image.data, w, h, bytes_per_line, QImage.Format_RGB888
        ).rgbSwapped()

        pixmap = QPixmap(image)
        self.label1.setPixmap(pixmap)
        
        
    def save_file(self):
        FileSave = QFileDialog.getSaveFileName(self, 'Save file', './',"Image files (*.jpg *.png);; XPM file (*.xpm)")
        cv2.imwrite('.jpg', self.save_img)


    def flip_image(self):
        image = cv2.flip(self.image, 1)
        self.save_img = image.copy()

        h, w, _ = image.shape
        bytes_per_line = 3 * w

        image = QImage(
            image.data, w, h, bytes_per_line, QImage.Format_RGB888
        ).rgbSwapped()
        pixmap = QPixmap(image)
        self.label2.setPixmap(pixmap)

    def flip_image2(self):
        image = cv2.flip(self.image, 0)
        self.save_img = image.copy()

        h, w, _ = image.shape
        bytes_per_line = 3 * w
        image = QImage(
            image.data, w, h, bytes_per_line, QImage.Format_RGB888
        ).rgbSwapped()
        pixmap = QPixmap(image) 
        self.label2.setPixmap(pixmap)
        
    def gray_image(self):
        image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.save_img = image.copy()

        h, w = image.shape
        bytes_per_line = w

        image = QImage(
            image.data, w, h, bytes_per_line, QImage.Format_Grayscale8
        )

        pixmap = QPixmap(image) 
        self.label2.setPixmap(pixmap)


    def r45_image(self):
        global pixmap,h,w,_
       
        angle = self.slider.value()
        center = (h/2, w/2)
        scale = 1
        matrix = cv2.getRotationMatrix2D(center, angle, scale)

        radians = math.radians(angle)
        sin = math.sin(radians)
        cos = math.cos(radians)
        bound_w = int((h * scale * abs(sin)) + (w * scale * abs(cos)))
        bound_h = int((h * scale * abs(cos)) + (w * scale * abs(sin)))

        matrix[0, 2] += ((bound_w / 2) - center[0])
        matrix[1, 2] += ((bound_h / 2) - center[1])

        image = cv2.warpAffine(self.image, matrix, (bound_h, bound_w))
        self.save_img = image.copy()

        h, w, _ = image.shape
        bytes_per_line = 3 * w
        image = QImage(
            image.data, w, h, bytes_per_line, QImage.Format_RGB888
        ).rgbSwapped()
        pixmap = QPixmap(image) 
        self.label2.setPixmap(pixmap)

    
    def small_image(self):
        image = cv2.pyrDown(self.image)
        self.save_img = image.copy()

        h, w, _ = image.shape
        bytes_per_line = 3 * w
        image = QImage(
            image.data, w, h, bytes_per_line, QImage.Format_RGB888
        ).rgbSwapped()

        pixmap = QPixmap(image) 
        self.label2.setPixmap(pixmap)

    def zoom_image(self):
        image = cv2.pyrUp(self.image)
        self.save_img = image.copy()

        h, w, _ = image.shape
        bytes_per_line = 3 * w
        image = QImage(
            image.data, w, h, bytes_per_line, QImage.Format_RGB888
        ).rgbSwapped()
        pixmap = QPixmap(image) 
        self.label2.setPixmap(pixmap)

    def blurring_image(self):
        mask = np.zeros((3, 3), np.float32)

        for i in range(3):
            for k in range(3):
                mask[i][k] = 1/9
        image = cv2.filter2D(self.image, -1, mask)
        self.save_img = image.copy()

        h, w, _ = image.shape
        bytes_per_line = 3 * w
        image = QImage(
            image.data, w, h, bytes_per_line, QImage.Format_RGB888
        ).rgbSwapped()

        pixmap = QPixmap(image) 
        self.label2.setPixmap(pixmap)

    def add_image(self):

        value = self.slider.value()
        b, g, r = cv2.split(self.image)

        b = cv2.add(b, value)
        g = cv2.add(g, value)
        r = cv2.add(r, value)

        image = cv2.merge((b, g, r))
        self.save_img = image.copy()

        h, w, _ = image.shape
        bytes_per_line = 3 * w
        image = QImage(
            image.data, w, h, bytes_per_line, QImage.Format_RGB888
        ).rgbSwapped()

        pixmap = QPixmap(image) 
        self.label2.setPixmap(pixmap)    

    def embossImage(self):
        mask = np.zeros((3, 3), np.float32)
        mask[0][0] = -1.0
        mask[0][1] = -1.0
        mask[1][0] = -1.0
        mask[1][2] = 1.0
        mask[2][1] = 1.0
        mask[2][2] = 1.0

        image = cv2.filter2D(self.image, -1, mask)
        image += 127
        self.save_img = image.copy()

        h, w, _ = image.shape
        bytes_per_line = 3 * w
        image = QImage(
            image.data, w, h, bytes_per_line, QImage.Format_RGB888
        ).rgbSwapped()

        pixmap = QPixmap(image) 
        self.label2.setPixmap(pixmap)

    def nag_image(self):
        global pixmap,h,w,_
        b, g, r = cv2.split(self.image)

        for i in range(h):
            for k in range(w):
                b[i][k] = 255 - b[i][k]
                g[i][k] = 255 - g[i][k]
                r[i][k] = 255 - r[i][k]

        image = cv2.merge((b, g, r))
        h, w, _ = image.shape
        bytes_per_line = 3 * w
        image = QImage(
            image.data, w, h, bytes_per_line, QImage.Format_RGB888
        ).rgbSwapped()
        pixmap = QPixmap(image) 
        self.label2.setPixmap(pixmap)

    def bin_image(self):
        global pixmap,h,w,_
        b, g, r = cv2.split(self.image)
        for i in range(h):
            for k in range(w):
                if b[i][k] > 127 and g[i][k] > 127 and r[i][k] > 127:
                    b[i][k] = 255
                    g[i][k] = 255
                    r[i][k] = 255
                else:
                    b[i][k] = 0
                    g[i][k] = 0
                    r[i][k] = 0

        image = cv2.merge((b, g, r))
        self.save_img = image.copy()

        h, w, _ = image.shape
        bytes_per_line = 3 * w
        image = QImage(
            image.data, w, h, bytes_per_line, QImage.Format_RGB888
        ).rgbSwapped()

        pixmap = QPixmap(image) 
        self.label2.setPixmap(pixmap)
    
    def str_image(self):
        image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(image)
        v = cv2.equalizeHist(v)

        image = cv2.merge([h, s, v])
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        self.save_img = image.copy()

        h, w, _ = image.shape
        bytes_per_line = 3 * w
        image = QImage(
            image.data, w, h, bytes_per_line, QImage.Format_RGB888
        ).rgbSwapped()
        
        pixmap = QPixmap(image) 
        self.label2.setPixmap(pixmap)

    def resize_image(self):
        global pixmap,h,w,_
        value = self.slider.value()
        image = cv2.resize(self.image , dsize=(int(h * value/50), int(w * value/50)))

        self.save_img = image.copy()
        h, w, _ = image.shape
        bytes_per_line = 3 * w
        image = QImage(
            image.data, w, h, bytes_per_line, QImage.Format_RGB888
        ).rgbSwapped()
        pixmap = QPixmap(image) 
        self.label2.setPixmap(pixmap)

    def sobel_image(self):
        global pixmap,h,w,_
        image = cv2.Sobel(self.image, cv2.CV_8U, dx=1, dy=1, ksize=3)
        
        self.save_img = image.copy()
        h, w, _ = image.shape
        bytes_per_line = 3 * w
        image = QImage(
            image.data, w, h, bytes_per_line, QImage.Format_RGB888
        ).rgbSwapped()
        pixmap = QPixmap(image) 
        self.label2.setPixmap(pixmap)

    def contours_image(self):
        
        global pixmap,h,w,_

        dst = self.image.copy()

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        gray = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        ret, binary = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)
        morp = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        image = cv2.bitwise_not(morp)

        contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(dst, contours, -1, (0, 0, 255), 3)
        for i in range(len(contours)):
            cv2.putText(dst, str(i), tuple(contours[i][0][0]), cv2.FONT_HERSHEY_COMPLEX, 1.3, (255, 0, 0), 1)
            print(i, hierarchy[0][i])
        
        image = dst.copy()


        self.save_img = image.copy()
        h, w, _ = image.shape
        bytes_per_line = 3 * w
        image = QImage(
            image.data, w, h, bytes_per_line, QImage.Format_RGB888
        ).rgbSwapped()
        pixmap = QPixmap(image) 
        self.label2.setPixmap(pixmap)
    
    def cartoon_image(self):
        image = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        image = cv2.medianBlur(image, 7)

        edges = cv2.Laplacian(image, cv2.CV_8U, ksize=5)
        ret, mask = cv2.threshold(edges, 100, 255, cv2.THRESH_BINARY_INV)
        image = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

        self.save_img = image.copy()
        h, w, _ = image.shape
        bytes_per_line = 3 * w
        image = QImage(
            image.data, w, h, bytes_per_line, QImage.Format_RGB888
        ).rgbSwapped()
        pixmap = QPixmap(image) 
        self.label2.setPixmap(pixmap)
    
    def face_image(self):
        face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
        grey = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)

        ## 얼굴 찾기
        face_rects = face_cascade.detectMultiScale(grey, 1.1, 5)
        image = self.image[:]

        for (x, y, w, h) in face_rects:
            cv2.rectangle(image, (x, y), (x + w, y + w), (0, 255, 0), 3)

        self.save_img = image.copy()
        h, w, _ = image.shape
        bytes_per_line = 3 * w
        image = QImage(
            image.data, w, h, bytes_per_line, QImage.Format_RGB888
        ).rgbSwapped()
        pixmap = QPixmap(image) 
        self.label2.setPixmap(pixmap)   
    
    def distortion_image(self):
        global h,w,_
        exp = self.slider.value() / 100  # 볼록지수 1.1~, 오목지수 0.1~1.0
        scale = 1
        mapy, mapx = np.indices((h, w), dtype=np.float32)

        mapx = 2 * mapx / (w-1) - 1
        mapy = 2 * mapy / (h-1) - 1

        r, theta = cv2.cartToPolar(mapx, mapy)  # 직교좌표를 극좌표로 변환시키는
        r[r < scale] = r[r < scale] ** exp

        mapx, mapy = cv2.polarToCart(r, theta)  # 극좌표를 직교좌표로 변환시켜주는 함수
        mapx = ((mapx +  1) * w - 1) / 2
        mapy = ((mapy +  1) * h - 1) / 2

        image = cv2.remap(self.image, mapx, mapy, cv2.INTER_LINEAR)

        self.save_img = image.copy()
        h, w, _ = image.shape
        bytes_per_line = 3 * w
        image = QImage(
            image.data, w, h, bytes_per_line, QImage.Format_RGB888
        ).rgbSwapped()
        pixmap = QPixmap(image) 
        self.label2.setPixmap(pixmap)   

    def clear_label(self):
        self.label2.clear()

if __name__ == "__main__":
    app = QApplication()
    window = MainWindow()
    window.show()
    sys.exit(app.exec())





