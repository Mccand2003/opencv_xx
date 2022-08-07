import cv2
import mediapipe as mp
import time


class FaceDetector():
    def __init__(self, min_detection_con=0.6):

        # 定义面部检测器
        self.faceDetection = mp.solutions.face_detection.FaceDetection(min_detection_con)

    def find_faces(self, img, draw=True):

        # 转换图片为rgb图片
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(img_rgb)

        bboxs = []

        # 在检测到的人脸中进行遍历，将人脸坐标以及人脸得分添加列表中
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                bbox_c = detection.location_data.relative_bounding_box
                img_h, img_w, _ = img.shape
                bbox = int(bbox_c.xmin * img_w), int(bbox_c.ymin * img_h), \
                       int(bbox_c.width * img_w), int(bbox_c.height * img_h)    # (x, y, w, h)
                bboxs.append([id, bbox, detection.score])

                # 绘制人脸区域矩形和人脸得分，并在四角加厚美观，
                if draw:
                    l = 25
                    x, y, w, h = bbox
                    x1, y1 = x + w, y + h
                    cv2.rectangle(img, bbox, (255, 0, 255), 1)
                    cv2.line(img, (x, y), (x + l, y), (255, 0, 255), 6)
                    cv2.line(img, (x, y), (x, y + l), (255, 0, 255), 6)
                    cv2.line(img, (x1, y), (x1 - l, y), (255, 0, 255), 6)
                    cv2.line(img, (x1, y), (x1, y + l), (255, 0, 255), 6)
                    cv2.line(img, (x, y1), (x + l, y1), (255, 0, 255), 6)
                    cv2.line(img, (x, y1), (x, y1 - l), (255, 0, 255), 6)
                    cv2.line(img, (x1, y1), (x1 - l, y1), (255, 0, 255), 6)
                    cv2.line(img, (x1, y1), (x1, y1 - l), (255, 0, 255), 6)
                    cv2.putText(img, f'{int(detection.score[0] * 100)}%',
                                (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_PLAIN,
                                2, (255, 0, 255), 2)
        return img, bboxs



# 主函数
def main():

    # 摄像头等的数据初始化
    cap = cv2.VideoCapture(0)
    first_time = 0
    detector = FaceDetector()

    while True:
        ret, img = cap.read()
        img, bboxs = detector.find_faces(img)

        # 通过time.time()获取程序运行时间，即获得两幅图片间隔的时间，用一秒除以它，即得帧率,并将其显示出来
        final_ime = time.time()
        fps = 1 / (final_ime - first_time)
        first_time = final_ime
        cv2.putText(img, f'FPS: {int(fps)}', (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (200, 255, 20), 2)

        cv2.imshow("img", img)
        if cv2.waitKey(1) & 0xff == ord('q'):
            break

# 函数运行
if __name__ == "__main__":
    main()