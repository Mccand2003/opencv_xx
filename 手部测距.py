import cv2
from cvzone.HandTrackingModule import HandDetector
import math
import numpy as np
import cvzone

# 窗口设置
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# 手部检测
detector = HandDetector(detectionCon=0.8, maxHands=1)

# X为食指根部跟小指根部之间的距离，Y此时手机摄像头的距离
x = [300, 245, 200, 170, 145, 130, 112, 103, 93, 87, 80, 75, 70, 67, 62, 59, 57]
y = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
# 创建一个二次函数，对xy的曲线进行拟合，从而得到a，b，c的值
coff = np.polyfit(x, y, 2)  # y = Ax^2 + Bx + C


while True:
    ret, img = cap.read()
    # 获取手部位置，不进行绘制
    hands = detector.findHands(img, draw=False)

    if hands:
        lmList = hands[0]['lmList']
        x, y, w, h = hands[0]['bbox']

        # 分别获取食指根部和小指根部的坐标(x,y)
        x1, y1, _ = lmList[5]
        x2, y2, _ = lmList[17]

        # 获取两位置之间的距离，并代入方程，从而得到摄像头跟手间的距离
        distance = int(math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2))
        A, B, C = coff
        distance_cm = A * distance ** 2 + B * distance + C

        # 绘制手部矩形及距离
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 3)
        cvzone.putTextRect(img, f'{int(distance_cm)} cm', (x + 5, y - 10))

    cv2.imshow("img", img)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break



