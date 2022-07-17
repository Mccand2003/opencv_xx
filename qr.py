import cv2
import pyzbar.pyzbar as pyzbar
import numpy as np


def decodeDisplay(image):
    # 对图像进行解码
    barcodes = pyzbar.decode(image)
    # 存放矩形数据（x, y, w, h）
    rects_list = []
    # 存放多边形点数据
    polygon_points_list = []
    # 存放二维码信息
    QR_info = []

    # 这里循环，因为画面中可能有多个二维码
    for barcode in barcodes:

        # 画出图像中条形码的边界框
        (x, y, w, h) = barcode.rect
        rects_list.append((x, y, w, h))
        polygon_points = barcode.polygon

        # 获得一个矩阵,存储点的信息
        extract_polygon_points = np.zeros((4, 2), dtype=np.int)
        for idx, points in enumerate(polygon_points):
            point_x, point_y = points
            extract_polygon_points[idx] = [point_x, point_y]

        # 将矩阵转化为opencv中常用的（x, 1, y）,reshape成 (4,1 2)也是可以的
        extract_polygon_points = extract_polygon_points.reshape((-1, 1, 2))
        polygon_points_list.append(extract_polygon_points)

        # 条形码数据为字节对象，所以如果我们想在输出图像上画出来，就需要先将它转换成字符串
        barcodeData = barcode.data.decode("utf-8")
        barcodeType = barcode.type

        # 绘出图像上条形码的数据和条形码类型
        text = "{} ({})".format(barcodeData, barcodeType)
        QR_info.append(text)
        cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    .5, (0, 0, 125), 2)

        # 向终端打印条形码数据和条形码类型
        print("[INFO] Found {} barcode: {}".format(barcodeType, barcodeData))

    # 返回图片和list
    return image, rects_list, polygon_points_list, QR_info

# 使用Ip摄像头连接,使用rtsp地址进行连接，@后接rtsp地址
cam_url='rtsp://admin:admin@192.168.43.1:8554/live'

def detect():
    cap = cv2.VideoCapture(cam_url)

    while True:

        # 读取当前帧
        ret, frame = cap.read()

        # 转换为灰度图是为了检测到二维码
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        im, rects_list, polygon_points_list, QR_info = decodeDisplay(gray)

        # 把检测到的二维码的信息绘制到BGR彩色图像上
        for data in zip(rects_list, polygon_points_list, QR_info):
            print(f"data: {data}")
            x, y, w, h = data[0]
            polygon_points = data[1]
            text = data[2]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.polylines(frame, [polygon_points],
                          isClosed=True,
                          color=(255, 0, 0),
                          thickness=2,
                          lineType=cv2.LINE_AA)
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        .5, (0, 255, 0), 2)

        # 显示处理后BGR图像
        cv2.imshow("frame", frame)

        # 按q键退出画面显示
        k = cv2.waitKey(1)
        if k == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    detect()

