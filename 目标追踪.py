import cv2

cap = cv2.VideoCapture(0)
tracker = cv2.TrackerCSRT_create()
ret, img = cap.read()
boxes = cv2.selectROI('img', img, False)
tracker.init(img, boxes)


def drawbox(img, boxes):
    (x, y, w, h) = int(boxes[0]), int(boxes[1]), int(boxes[2]), int(boxes[3])
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)

while True:
    timer = cv2.getTickCount()
    ret, img = cap.read()
    ret, boxes = tracker.update(img)
    if ret:
        drawbox(img, boxes)

    cv2.imshow('img', img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):
        break
cap.release()
cv2.destroyAllWindows()


