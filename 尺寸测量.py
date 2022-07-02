import cv2
import numpy as np

# 自定义函数
def cv2_show(name, img) :
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized

if __name__ == '__main__':

    # cv2.namedWindow("camera", 1)
    # 开启ip摄像头
    # video = "http://admin:admin@192.168.43.1:8081/"
    cap = cv2.VideoCapture('010.mp4')
    # retv = cv2.VideoCapture.isOpened(video)
# 图像处理
    while True:
        ret, img = cap.read()

        # 图像压缩，灰度化，二值化，高斯滤波
        img_resize = resize(img, 500)
        img_gray = cv2.cvtColor(img_resize, cv2.COLOR_BGR2GRAY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        blur = cv2.GaussianBlur(img_gray, (3, 3), 1, 1)

        # sobel算子，canny边缘检测，闭操作
        sobelx = cv2.Sobel(blur, cv2.CV_64F, 0, 1)
        sobely = cv2.Sobel(blur, cv2.CV_64F, 1, 0)
        sobel = cv2.addWeighted(sobely, 0.6, sobelx, 0.3, 2)
        dst = cv2.convertScaleAbs(sobel)
        canny = cv2.Canny(dst, 50, 127)
        close_1 = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, kernel)
        close_2 = cv2.morphologyEx(close_1, cv2.MORPH_CLOSE, kernel)

        # 创建list存储w和h
        w_list = []
        h_list = []

        #寻找轮廓
        contours, hierarchy = cv2.findContours(close_2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 遍历轮廓
        for (i, c) in enumerate(contours):
            rect = cv2.minAreaRect(c)
            area = cv2.contourArea(c)
            print(area)

            # 过滤小面积轮廓
            if area > 400:
                ((x, y), (w, h), o) = rect
                points = cv2.boxPoints(rect)
                points = np.int0(points)
                (xr, yr, wr, hr) = points
                high = h
                width = w
                w_list.append(width )
                h_list.append(high )

                # 显示特定轮廓
                if area < 9000:
                    img_resize = cv2.drawContours(img_resize, [points], 0, [255, 255, 0], 1)
        w_list.sort()
        h_list.sort()

        # 计算w和h
        dim_h = h_list[0] / h_list[-1] * 29.7
        dim_w = w_list[0] / w_list[-1] * 21.0

        # 将w和h显示
        cv2.putText(img_resize, "{:.1f}cm".format(dim_w),
                    (int(15), int(50)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (255, 255, 255), 1)
        cv2.putText(img_resize, "{:.1f}cm".format(dim_h),
                    (int(50), int(15)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (255, 255, 255), 1)
        # cv2_show('0', img_resize)
        key = cv2.waitKey(100)
        cv2.imshow('mcc', img_resize)
        if key == 27:
            # esc键断开连接
            print("esc break...")
            break

    # 释放内存
    cap.release()
    cv2.destroyAllWindows()