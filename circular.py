import cv2

import numpy as np


# 将多个窗口聚集到一个上面
def stackImages(scale, img_array):

    # 根据输入的元组元素个数决定长和宽
    rows = len(img_array)
    cols = len(img_array[0])

    # 判断元组中元素是否为一个列表
    rowsAvailable = isinstance(img_array[0], list)

    # 分别获取原单个窗口的宽和高
    width = img_array[0][0].shape[1]
    height = img_array[0][0].shape[0]

    if rowsAvailable:
        # 可以理解为二维数组进行遍历，在对应的位置放上元组中的元素所代表的窗口
        for x in range(0, rows):
            for y in range(0, cols):

                if img_array[x][y].shape[:2] == img_array[0][0].shape[:2]:
                    img_array[x][y] = cv2.resize(img_array[x][y], (0, 0), None, scale, scale)
                else:
                    img_array[x][y] = cv2.resize(img_array[x][y], (img_array[0][0].shape[1], img_array[0][0].shape[0]),
                                                 None, scale, scale)
                if len(img_array[x][y].shape) == 2:
                    img_array[x][y] = cv2.cvtColor(img_array[x][y], cv2.COLOR_GRAY2BGR)

        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(img_array[x])

        ver = np.vstack(hor)

    else:

        for x in range(0, rows):
            if img_array[x].shape[:2] == img_array[0].shape[:2]:
                img_array[x] = cv2.resize(img_array[x], (0, 0), None, scale, scale)
            else:
                img_array[x] = cv2.resize(img_array[x], (img_array[0].shape[1], img_array[0].shape[0]), None, scale, scale)
            if len(img_array[x].shape) == 2:
                img_array[x] = cv2.cvtColor(img_array[x], cv2.COLOR_GRAY2BGR)


        hor = np.hstack(img_array)
        ver = hor

    return ver



def empty(a):
    pass

# 通过软件连接手机摄像头
cam_url='rtsp://admin:admin@192.168.43.1:8554/live'
cap = cv2.VideoCapture(0)


# 创造一个显示窗口，并对显示窗口进行缩放
cv2.namedWindow("TrackBars")
cv2.resizeWindow("TrackBars", 640, 240)

# 创造滑条并对hsv三通道数值初始化
cv2.createTrackbar("Hue Min", "TrackBars", 32, 179, empty)
cv2.createTrackbar("Hue Max", "TrackBars", 127, 179, empty)
cv2.createTrackbar("Sat Min", "TrackBars", 110, 255, empty)
cv2.createTrackbar("Sat Max", "TrackBars", 255, 255, empty)
cv2.createTrackbar("Val Min", "TrackBars", 133, 255, empty)
cv2.createTrackbar("Val Max", "TrackBars", 255, 255, empty)



while True:

    ret, img = cap.read()
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 设置HSV3通道的变化条，通过变化条的调整，可以观察对图像的影响，同时提取了颜色
    h_min = cv2.getTrackbarPos("Hue Min", "TrackBars")
    h_max = cv2.getTrackbarPos("Hue Max", "TrackBars")
    s_min = cv2.getTrackbarPos("Sat Min", "TrackBars")
    s_max = cv2.getTrackbarPos("Sat Max", "TrackBars")
    v_min = cv2.getTrackbarPos("Val Min", "TrackBars")
    v_max = cv2.getTrackbarPos("Val Max", "TrackBars")

    # 通过设置掩膜及按位与操作来观察特定的颜色区域
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    mask = cv2.inRange(img_hsv, lower, upper)
    img_result = cv2.bitwise_and(img, img, mask=mask)



    try:
        # 形态学处理
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur_img = cv2.GaussianBlur(img_gray, (3, 3), 0)

        # 利用霍夫变换查找圆的位置
        circles = cv2.HoughCircles(image=blur_img, method=cv2.HOUGH_GRADIENT,
                                   dp=1.2,

                                   minDist=100, # 两个圆之间圆心的最小距离.如果太小的，多个相邻的圆可能被错误地检测成了一个重合的圆。反之，这参数设置太大，某些圆就不能被检测出来。

                                   param1=100,

                                   param2=120,

                                   minRadius=0, # 圆半径的最小值

                                   maxRadius=2000) # 圆半径的最大值

        circles = np.uint16(np.around(circles))



        for i in circles[0, :]:
            # 画圆
            cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 255), 5)
            # 画圆心
            cv2.circle(img, (i[0], i[1]), 2, (0, 255, 255), 3)
            print('圆心坐标为（%.2f,%.2f）' % (i[0], i[1]))
            a = str(i[0])
            b = str(i[1])
            str = "(" + a + " ," + b + ")"
            # 绘制圆心坐标
            img = cv2.putText(img, str , (i[0], i[1]), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2, )

    except:
        print('无法识别到圆')


    # 分别将灰度图，hsv通道图，原图，掩模，按位与之后的结果，原图显示在同一个窗口上
    imgStack = stackImages(0.6, ([img_gray, img_hsv, img], [mask, img_result, img]))

    # 图像显示及关闭
    cv2.imshow("Stack Images",
               np.hstack([img_result, img]))# 将两个窗口同时显示在一个屏幕上

    if cv2.waitKey(1) & 0xff == ord('q'):
        break
