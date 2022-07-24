import cv2
import numpy as np

# 存储BGR颜色在HSV空间的范围，red有两部分范围
color_dist = {'red1': {'Lower': np.array([0, 43, 46]), 'Upper': np.array([10, 255, 255])},
              'red2': {'Lower': np.array([156, 43, 46]), 'Upper': np.array([180, 255, 255])},
              'blue': {'Lower': np.array([100, 43, 46]), 'Upper': np.array([124, 255, 255])},
              'green': {'Lower': np.array([35, 43, 46]), 'Upper': np.array([77, 255, 255])},
              }

# 要识别的颜色
color_recognition_list = ['red', 'green', 'blue']

# 分别存储矩形数据，中心点x和y坐标和识别的颜色
boxes_list = []
center_x_list = []
center_y_list = []
target_color_list = []

# 图像显示函数
def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey()
    cv2.destroyAllWindows()

# 对图片进行灰度化及形态学操作后寻找轮廓，及中心点的xy坐标并存储起来
def find_target(img, color=None):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_ero = cv2.erode(img_gray, (3, 3))
    cv_show('img_ero', img_ero)
    comtours, hierarchy = cv2.findContours(img_ero,
                                           cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_NONE)
    for c in comtours:

        # 对于面积过小的区域视为干扰，忽略
        if cv2.contourArea(c) < 1000:
            continue
        else:
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            boxes_list.append(box)
            M = cv2.moments(c)
            # 计算中心点的x、y坐标
            center_x_list.append(int(M['m10'] / M['m00']))
            center_y_list.append(int(M['m01'] / M['m00']))
            target_color_list.append(color)

# 寻找颜色区域函数
def select_color_img(target_color, img):
    for i in target_color:
        # 本函数只考虑了寻找bgr三色
        if i == 'green' or i == 'blue':
            # 寻找颜色对应范围内的像素区域作为掩模，再和原图进行按位与操作，从而得到对应颜色区域
            mask = cv2.inRange(img, color_dist[i]['Lower'], color_dist[i]['Upper'])
            cv_show('mask', mask)
            color = cv2.bitwise_and(img, img, mask=mask)
            cv_show('color', color)
            # 将得到的颜色区域进行寻找轮廓，从而得到中心点坐标等数据
            find_target(color, i)
        else:
            # 红色在hsv色彩空间中有两部分范围，分别进行操作之后，将得的图片进行相加，从而得到一个完整的红色区域
            mask1 = cv2.inRange(img, color_dist['red1']['Lower'], color_dist['red1']['Upper'])
            red1 = cv2.bitwise_and(img, img , mask=mask1)
            mask2 = cv2.inRange(img, color_dist['red2']['Lower'], color_dist['red2']['Upper'])
            red2 = cv2.bitwise_and(img, img, mask=mask2)
            color = cv2.add(red1, red2)
            cv_show('color', color)
            find_target(color, i)
    return color

# 图像预处理
img = cv2.imread('001.jpg')
img = cv2.resize(img, (800, 600))
img_blur = cv2.GaussianBlur(img, (3, 3), 0, 0)
img_hsv = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HSV)

# 寻找特定颜色区域
img_sel = select_color_img(color_recognition_list, img_hsv)

for i in range(0, len(boxes_list)):
    # 绘制矩形框
    cv2.drawContours(img, [np.int0(boxes_list[i])], -1, (0, 255, 255), 2)

    # 绘制中心点
    cv2.circle(img, (center_x_list[i], center_y_list[i]), 7, 128, -1)

    # 把坐标转化为字符串
    str1 = '(' + target_color_list[i] + ':' + str(center_x_list[i]) + ',' + str(center_y_list[i]) + ')'

    # 绘制坐标及颜色
    cv2.putText(img, str1, (center_x_list[i] - 50, center_y_list[i] + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2,
                cv2.LINE_AA)

cv_show("img+", img)
# 针对实际运用情形是可以调整寻找颜色的范围，因为一种颜色的范围实际上非常大，在实际运用中容易找到不需要的区域