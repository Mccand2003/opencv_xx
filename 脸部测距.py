import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector

# 设置摄像头及面部检测器
cap = cv2.VideoCapture(0)
cap.set(3, 1080)
cap.set(4, 720)
detector = FaceMeshDetector(maxFaces=1)

while True:
    ret, img = cap.read()
    # 通过面部检测器获取轮廓和图形
    img, faces = detector.findFaceMesh(img, draw=False)

    if faces:
        face = faces[0]
        # 分别得到左右眼的位置
        point_left = face[145]
        point_right = face[374]

        # 得到左右眼之间的宽度
        w, _ = detector.findDistance(point_left, point_right)

        # 男生左右眼之间平均宽度6.3cm，以此为参照物结合焦距，从而得到摄像头离人脸的距离
        W = 6.3       
        f = 840
        d = (W * f) / w
        print(d)

        # 参数分别为图片，绘制字符串，绘制字符串的大小，文本比例
        # 默认紫底白字，可以通过设置colorT改变文本颜色,colorR改变底色
        cvzone.putTextRect(img, f'Depth: {int(d)}cm',
                           (face[10][0] - 100, face[10][1] - 50),
                           scale=2)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
