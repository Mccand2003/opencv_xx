import math
import random
import cvzone
import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector

# 设置显示窗口大小为h:w=720：1280
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# 手部检测，maxHands=1即只检测一个手
detector = HandDetector(detectionCon=0.5, maxHands=1)

# 设置游戏参数，path_food为食物图片
class SnakeGameClass:
    def __init__(self, path_food):
        self.points = []  # 蛇的所有点，绘制后形成蛇的形状
        self.lengths = []  # 每个点之间的距离
        self.current_length = 0  # 蛇的总长度
        self.allowed_length = 150  # 允许的最长长度
        self.previous_head = 0, 0  # 之前点的坐标
        self.imgFood = cv2.imread(path_food, cv2.IMREAD_UNCHANGED)  # 读取食物图片
        self.h_food, self.w_food, _ = self.imgFood.shape    # 图片尺寸（h, w, c）
        self.food_point = 0, 0   # 食物位置初始化
        self.random_food_location()     # 随机生成位置
        self.score = 0      # 游戏分数
        self.gameOver = False   # 游戏是否结束

    def random_food_location(self):  # 随机生成食物位置,坐标区间设置
        self.food_point = random.randint(200, 900), random.randint(150, 550)


    def update(self, img_main, current_head):

        if self.gameOver:
            # 参数分别为图片，绘制字符串，绘制字符串的大小，文本比例，文本粗细，文本间隔
            # 默认紫底白字，可以通过设置colorT改变文本颜色,colorR改变底色
            cvzone.putTextRect(img_main, "Game Over", [300, 400],
                               scale=7, thickness=5, offset=20)
            cvzone.putTextRect(img_main, f'Your Score: {self.score}', [300, 550],
                               scale=7, thickness=5, offset=20)
        else:
            px, py = self.previous_head
            cx, cy = current_head

            self.points.append([cx, cy])
            # 获取前一个点和新的点的实际距离即欧几里的距离，并更新点的参数
            distance = math.hypot(cx - px, cy - py)
            # 将距离存储进lengths里
            self.lengths.append(distance)
            self.current_length += distance
            self.previous_head = cx, cy

            # 当需要绘制的长度超过允许最大长度时，则删除索引最后的点和位置，造成蛇实时移动的效果
            if self.current_length > self.allowed_length:
                for i, length in enumerate(self.lengths):
                    self.current_length -= length

                    # 删除元素
                    self.lengths.pop(i)
                    self.points.pop(i)
                    if self.current_length < self.allowed_length:
                        break

            # 如果蛇吃到了食物,即手指的坐标接近食物的坐标
            rx, ry = self.food_point
            if rx - self.w_food // 2 < cx < rx + self.w_food // 2 and ry - self.h_food // 2 < cy < ry + self.h_food // 2:
                self.random_food_location()

                # 吃下食物后，蛇身长度增加，分数加一
                self.allowed_length += 50
                self.score += 1
                print(self.score)

            # 绘制出蛇的形状
            if self.points:
                for i, point in enumerate(self.points):
                    if i != 0:
                        # 制出蛇身的每一个点
                        cv2.line(img_main, self.points[i - 1], self.points[i], (0, 0, 255), 20)
                # 绘制蛇头
                cv2.circle(img_main, self.points[-1], 20, (0, 255, 0), cv2.FILLED)

            # 绘制食物
            img_main = cvzone.overlayPNG(img_main, self.imgFood,
                                         (rx - self.w_food // 2, ry - self.h_food // 2))

            # 分数实时显示
            cvzone.putTextRect(img_main, f'Score: {self.score}', [50, 80],
                               scale=3, thickness=3, offset=10)

            # 得到由点组成的多边形，如果蛇头代表的点与多边形距离过近，则得到蛇头与蛇身相碰，游戏结束并对参数进行初始化
            pts = np.array(self.points[:-2], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(img_main, [pts], False, (0, 255, 0), 3)
            minDist = cv2.pointPolygonTest(pts, (cx, cy), True)
            if -1 <= minDist <= 1:
                print("Hit")
                # 参数初始化
                self.gameOver = True
                self.points = []
                self.lengths = []
                self.current_length = 0
                self.allowed_length = 150
                self.previous_head = 0, 0
                self.random_food_location()

        return img_main

# 设置游戏，Dount.png为食物图片
game = SnakeGameClass("Donut.png")

while True:
    ret, img = cap.read()
    img = cv2.flip(img, 1)  # 将图片进行翻转，1为水平翻转
    hands, img = detector.findHands(img, flipType=False)    # 分别返回手部轮廓及绘制好后的图片,如果查找不到手部，则只返回图片

    if hands:   # 手部追踪会绘制21个点，其中八号为食指尖端的点，即利用这个点来进行游戏
        lmList = hands[0]['lmList']
        point_index = lmList[8][0:2]    # 只取（x, y）
        img = game.update(img, point_index)

    # 图像显示
    cv2.imshow("Image", img)
    key = cv2.waitKey(1)

    # 按r重新开始游戏
    if key == ord('r'):
        game.gameOver = False


# cvzone库写了很多OpenCV用的东西，函数有参考价值