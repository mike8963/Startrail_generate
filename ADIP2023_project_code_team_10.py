import sys
from PyQt5.QtWidgets import QApplication, QDialog, QPushButton, QFileDialog, QGraphicsScene, QGraphicsView, QGraphicsPixmapItem, QComboBox
from PyQt5.QtGui import QPixmap, QImage, QMovie
from PyQt5.uic import loadUi
from PyQt5.QtWidgets import QLabel
from PyQt5.QtCore import Qt, QRectF
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import imageio
import time


gif_movie = None
gif_scene_item = None
mode = "circle"

def load_image(dialog):
    file_path, _ = QFileDialog.getOpenFileName(dialog, "選擇圖片", "", "圖片文件 (*.png *.jpg *.jpeg *.bmp *.gif)")  # 打開圖片文件對話框
    if file_path:  # 確保用戶選擇了文件而不是取消操作
        pixmap = QPixmap(file_path)
        # 在這裡你可以對所選圖片進行處理或顯示
        print("選擇的圖片文件:", file_path)
        image = cv2.imread(file_path)
    return image

def check_movie_state(state):
    if state == QMovie.Running:
        print("GIF animation is running")
    elif state == QMovie.NotRunning:
        print("GIF animation stopped")
    elif state == QMovie.Paused:
        print("GIF animation paused")

def show_image(dialog, scene, scene_for_gif, gif_view):
    global gif_movie, gif_scene_item, mode

    loaded_image = load_image(dialog)

    loaded_image = star_trail2(loaded_image, mode)

    # 取得圖片的高度、寬度和通道數
    height, width, _ = loaded_image.shape
    # 計算每行的位元組數量
    bytesPerLine = 3 * width
    # 建立 QImage 物件，使用 RGB888 格式
    qImg = QImage(loaded_image.data, width, height, bytesPerLine, QImage.Format_RGB888)
    # 將 QImage 轉換為 QPixmap
    pixmap = QPixmap.fromImage(qImg)
    # 清空場景
    scene.clear()  
    # 將 QPixmap 加入場景
    scene.addPixmap(pixmap)
    # 取得場景的 QGraphicsView
    graphics_view = scene.views()[0]
    # 將 QPixmap 轉換為 QGraphicsPixmapItem
    pixmap_item = QGraphicsPixmapItem(pixmap)
    # 將 QGraphicsPixmapItem 加入場景
    scene.addItem(pixmap_item)
    # 取得 QPixmap 圖片的邊界矩形
    jpg_rect = pixmap_item.boundingRect()
    # 設置場景的大小為圖片的邊界矩形
    scene.setSceneRect(jpg_rect)
    # 將 QGraphicsView 設置為符合圖片邊界矩形的大小，保持長寬比
    graphics_view.fitInView(jpg_rect, Qt.KeepAspectRatio) 


    # 清空 GIF 圖形場景
    scene_for_gif.clear()
    # GIF 檔案路徑
    gif_path = 'output.gif'
    # 創建 QMovie 物件
    gif_movie = QMovie(gif_path)
    # 取得目前的 QPixmap
    gif_pixmap = QPixmap(gif_movie.currentPixmap())
    # 建立 QGraphicsPixmapItem 並加入至場景
    gif_scene_item = QGraphicsPixmapItem(gif_pixmap)
    scene_for_gif.addItem(gif_scene_item)

    # 連接狀態改變的信號至函數 check_movie_state
    gif_movie.stateChanged.connect(check_movie_state)
    # 連接幀改變的信號至函數 update_gif_frame
    gif_movie.frameChanged.connect(update_gif_frame)
    # 啟動 GIF 播放
    gif_movie.start()

    # 取得 GIF 圖片的邊界矩形
    gif_rect = gif_scene_item.boundingRect()
    # 設置場景大小為 GIF 圖片的邊界矩形
    scene_for_gif.setSceneRect(gif_rect)
    # 將 QGraphicsView 設置為符合 GIF 圖片邊界矩形的大小，保持長寬比
    gif_view.fitInView(gif_rect, Qt.KeepAspectRatio)
    

def update_gif_frame():
    global gif_movie, gif_scene_item

    # 更新 QGraphicsPixmapItem 的内容為當前 GIF
    current_frame = gif_movie.currentPixmap()
    gif_scene_item.setPixmap(current_frame)


def star_trail2(img, mode = "circle"):
    start_SplitImage_time = time.time()
    # 前景切割
    (h, w, _) = img.shape
    
    background_mask, foreground_mask = AAA_(img)
    
    background_mask = background_mask.astype('uint8')
    foreground_mask = foreground_mask.astype('uint8')
        
    background = cv2.bitwise_and(img, img, mask = background_mask)
    foreground = cv2.bitwise_and(img, img, mask = foreground_mask)

    cv2.imwrite("foreground_img.jpg", foreground)
    cv2.imwrite("background_img.jpg", background)

    
    img_copy2 = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
    _, img_copy2 = cv2.threshold(img_copy2, 1, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(shape = cv2.MORPH_RECT, ksize = (5, 5))

    img_copy2 = cv2.erode(img_copy2, kernel,iterations=30)

    star_contours = get_star_contours(img)
    star_contours = cv2.bitwise_and(star_contours, star_contours, mask = background_mask)
    cv2.imwrite("star_contours_masked.jpg", star_contours)
 
    
    # thresholding
    _, thresh = cv2.threshold(star_contours, 70, 255, cv2.THRESH_BINARY)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    white_mask = np.zeros_like(img)

    cv2.drawContours(white_mask, contours, -1, (255, 255, 255), thickness=cv2.FILLED)
    
    c = max(contours, key=cv2.contourArea)
    
    M= cv2.moments(c)
    center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
          

    if mode == "circle":
        scaling_factor = 1.0
        decay_sf = 0
        rotate_factor = 0.1
    elif mode == "spiral":
        # 試驗中
        scaling_factor = 1.0
        #decay_sf = -0.0015
        decay_sf = 0.0009
        rotate_factor = 0.1
    elif mode == "radial":
        # 試驗中
        scaling_factor = 1.0
        decay_sf = -0.0025
        rotate_factor = 0
    else:
        pass
       
    (h, w, _) = img.shape
    masked = cv2.bitwise_and(img, white_mask)
    black_mask = 255 - white_mask

    img_list = []
    print(h * w)
    print("generating gif image...")
    if(h * w < 1100000):
        star_count = 1000
    elif(h * w > 42000000):
        star_count = 200
    else:
        star_count = 300
        
    for i in range(0, star_count):
        rotate_matrix = cv2.getRotationMatrix2D(center, i * rotate_factor, scaling_factor - i * decay_sf)
        rotate_mask = cv2.warpAffine(black_mask, rotate_matrix, (w, h))
        rotate_img = cv2.warpAffine(masked, rotate_matrix, (w, h))
        img = cv2.bitwise_and(img, img, rotate_mask)
        img = cv2.add(img, rotate_img)

        if(i%10 == 0):
            img2 = cv2.bitwise_and(img, img, mask=background_mask)
            img2 = cv2.add(img2, foreground)
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
            img_list.append(img2)
    gif_name = 'output' + '.gif'
    imageio.mimsave(gif_name, img_list, fps=15)


    img = cv2.bitwise_and(img, img, mask=background_mask)
    img = cv2.add(img, foreground)

    cv2.imwrite("img.jpg", img)
    end_SplitImage_time = time.time()
    execution_time = end_SplitImage_time - start_SplitImage_time
    print(f"星軌旋轉次數為: {star_count} 次")
    print(f"程式執行時間為: {execution_time} 秒")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def on_combobox_changed(index):
    global mode
    print(f"目前選取的索引: {index}")
    if index == 0:
        mode = 'circle'
    elif index == 2:
        mode = 'spiral'
    elif index == 1:
        mode = 'radial'
    else:
        pass
    

def AAA_(img):
    kernel = cv2.getStructuringElement(shape = cv2.MORPH_RECT, ksize = (5, 5))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel1 = np.ones((5, 5), np.float32) / 25
    gray = cv2.filter2D(gray, -1, kernel1)


    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    #sobelx = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
    #sobely = cv2.Scharr(gray, cv2.CV_64F, 0, 1)

    sobelx = np.uint8(np.absolute(sobelx))
    sobely = np.uint8(np.absolute(sobely))

    output = cv2.bitwise_or(sobelx, sobely)#找尋邊界

    sobelx_horizontal = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    output_2 = np.uint8(np.absolute(sobelx_horizontal))

    output = cv2.add(output_2,output)

    #output = cv2.add(output_2,output)

    #output = cv2.add(output_2,output)

    #output = cv2.add(output_2,output)

    

    cv2.imwrite("img_before_mask.jpg", output) 
    #laplacian = cv2.Laplacian(sobel_combined, cv2.CV_64F)#高通
    #laplacian = np.uint8(np.absolute(laplacian))
    
    #output = cv2.dilate(output, kernel,iterations=1)
    #output = cv2.GaussianBlur(output, (5, 5), 0)#低通
    
    _, output = cv2.threshold(output, 60, 255, cv2.THRESH_BINARY)
   
    #output = cv2.erode(output, kernel,iterations=1)
    output = cv2.dilate(output, kernel,iterations=5)
    output = cv2.erode(output, kernel,iterations=5)
    
    
    

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(output )
    max_label = 1
    max_area = stats[max_label, cv2.CC_STAT_AREA]
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] > max_area:
            max_label = label
            max_area = stats[label, cv2.CC_STAT_AREA]
    
    background_mask = np.zeros_like(labels)
    foreground_mask = np.zeros_like(labels)
    background_mask[labels == max_label] = 255

    #output = cv2.adaptiveThreshold(output, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

    print(np.shape(background_mask))
    #抓取前景部分
    for i in range(0, np.shape(background_mask)[1]):
        first_number = 0
        for j in range(0, np.shape(background_mask)[0]):
            if(background_mask[j][i] == 255 or first_number == 1):
                first_number = 1
                background_mask[j][i] = 255
            else:
                background_mask[j][i] = 0

    #顏色反轉
    for i in range(0, np.shape(background_mask)[1]):
        for j in range(0, np.shape(background_mask)[0]):
            if(background_mask[j][i] == 255):
                background_mask[j][i] = 0
            else:
                background_mask[j][i] = 255
    cv2.imwrite("img_mask.jpg", background_mask) 
    foreground_mask = 255 - background_mask
    return background_mask, foreground_mask
    
def AAA(img):
    kernel = cv2.getStructuringElement(shape = cv2.MORPH_RECT, ksize = (5, 5))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.power(gray / 255.0, 0.6)
    gray = np.uint8(gray * 255)

    kernel1 = np.ones((5, 5), np.float32) / 25
    kernel2 = np.ones((5, 5), np.float32) / 25


    gray1 = cv2.filter2D(gray, -1, kernel2)
    gray = cv2.filter2D(gray, -1, kernel1)

    


    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    #sobelx = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
    #sobely = cv2.Scharr(gray, cv2.CV_64F, 0, 1)

    sobelx = np.uint8(np.absolute(sobelx))
    sobely = np.uint8(np.absolute(sobely))

    output = cv2.bitwise_or(sobelx, sobely)#找尋邊界

    sobelx_horizontal = cv2.Sobel(gray1, cv2.CV_64F, 0, 1, ksize=3)
    output_2 = np.uint8(np.absolute(sobelx_horizontal))

    #output = cv2.add(output_2,output)

    #output = cv2.add(output_2,output)

    #output = cv2.add(output_2,output)

    #output = cv2.add(output_2,output)

    

    cv2.imwrite("img_before_mask.jpg", output_2) 
    #laplacian = cv2.Laplacian(sobel_combined, cv2.CV_64F)#高通
    #laplacian = np.uint8(np.absolute(laplacian))
    
    #output = cv2.dilate(output, kernel,iterations=1)
    #output = cv2.GaussianBlur(output, (5, 5), 0)#低通
    
    _, output = cv2.threshold(output, 60, 255, cv2.THRESH_BINARY)
   
    #output = cv2.erode(output, kernel,iterations=1)
    output = cv2.dilate(output, kernel,iterations=5)
    output = cv2.erode(output, kernel,iterations=5)
    
    
    

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(output )
    max_label = 1
    max_area = stats[max_label, cv2.CC_STAT_AREA]
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] > max_area:
            max_label = label
            max_area = stats[label, cv2.CC_STAT_AREA]
    
    output_ = np.zeros_like(labels)
    output_[labels == max_label] = 255

    #output = cv2.adaptiveThreshold(output, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

    print(np.shape(output_))
    #抓取前景部分
    for i in range(0, np.shape(output_)[1]):
        first_number = 0
        for j in range(0, np.shape(output_)[0]):
            if(output_[j][i] == 255 or first_number == 1):
                first_number = 1
                output_[j][i] = 255
            else:
                output_[j][i] = 0

    #顏色反轉
    for i in range(0, np.shape(output_)[1]):
        for j in range(0, np.shape(output_)[0]):
            if(output_[j][i] == 255):
                output_[j][i] = 0
            else:
                output_[j][i] = 255
    cv2.imwrite("img_mask.jpg", output_) 
    return output_
        
def get_star_contours(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    kernel = cv2.getStructuringElement(shape = cv2.MORPH_RECT, ksize = (5, 5))
    erode = cv2.erode(img, kernel, iterations = 5)
    dilate = cv2.dilate(erode, kernel, iterations = 5)
    img = img - dilate
    return img       
  
def main():
    app = QApplication(sys.argv)
    dialog = QDialog()
    ui_file_path = 'test2.ui'
    loadUi(ui_file_path, dialog)

    graphics_view = dialog.findChild(QGraphicsView, 'png')  # 找到 QGraphicsView 组件
    scene = QGraphicsScene()  # 創建一個 QGraphicsScene
    graphics_view.setScene(scene)

    gif_view = dialog.findChild(QGraphicsView, 'gif')  # 找到名為 'gif' 的 QGraphicsView 组件
    scene_for_gif = QGraphicsScene()  # 創建一個 QGraphicsScene 用於顯示 GIF
    gif_view.setScene(scene_for_gif)

    combo_box = dialog.findChild(QComboBox, 'comboBox_2')
    combo_box.currentIndexChanged.connect(on_combobox_changed)

    # 獲取名為 pushButton 的按鈕，並連接到 load_image 函數
    push_button = dialog.findChild(QPushButton, 'pushButton')
     # 將 dialog 對象傳遞到 load_image 函數
    push_button.clicked.connect(lambda: show_image(dialog, scene, scene_for_gif, gif_view)) 

    dialog.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()