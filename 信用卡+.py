import numpy as np
import cv2

# 自定义函数
def cv2_show(name, img) :
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def sort_contours(cnts, method="left-to-right"):
    reverse = False
    i = 0
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    # 用一个最小的矩形，把找到的形状包起来x,y,h,w
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))

    return cnts, boundingBoxes

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

img = cv2.imread('mcc.jpg')
ref = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ref = cv2.threshold(ref, 10, 255, cv2.THRESH_BINARY_INV)[1]

# 计算轮廓
refCnts, hierarchy = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img, refCnts, -1, (0, 0, 255), 3)
print(np.array(refCnts, dtype=object).shape)
refCnts = sort_contours(refCnts, method="left-to-right")[0] # 排序，从左到右，从上到下
digits = {}

# 遍历每一个轮廓
for (i, c) in enumerate(refCnts):
# 计算外接矩形并且resize成合适大小
    (x, y, w, h) = cv2.boundingRect(c)
    roi = ref[y + 10:y + h - 10, x:x + h - 38]
    roi = cv2.resize(roi, (58, 88))
    # 每一个数字对应每一个模板
    digits[i] = roi

# 读取图片并灰度化，然后裁剪
img = cv2.imread('img.png')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
img_resize = resize(img_gray, width=300)
img_resize2 = resize(img, width=300)

# 自定义卷积核
retval = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))

# 进行一次礼帽操作，再进行二次闭操作和一次阈值处理
tophat = cv2.morphologyEx(img_resize, cv2.MORPH_TOPHAT, retval)
close_1 = cv2.morphologyEx(tophat, cv2.MORPH_CLOSE, retval)
ret, threshold = cv2.threshold(close_1, 127, 255, cv2.THRESH_OTSU)
close_2 = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, retval)

# 查找并绘制轮廓
contours, hieraychy = cv2.findContours(close_2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
x, y, w, h = cv2.boundingRect(contours[0])
brcnt = np.array([[[x, y]], [[x+w, y]], [[x+w, y+h]], [[x, y+h]]])
draw = cv2.drawContours(img_resize, [brcnt], -1, (0, 0, 255), 2)
locs = []

for (i, c) in enumerate(contours) :
    (x, y, w, h) = cv2.boundingRect(c)
    cv2.rectangle(img_resize, (x-2, y-2), (x+w, y+h), (0, 255, 200), 1)
    ar = w / float(h)

    # 选择合适的区域，根据实际任务来，这里的基本都是四个数字一组
    if ar > 2.0 and ar < 4.0:
        if (w > 40 and w < 60) and (h > 10 and h < 20):
            # 符合的留下来
            locs.append((x -2 , y -2, w+2 , h +2))

# 将符合的轮廓从左到右排序
locs = sorted(locs, key=lambda x:x[0])
output = []

# 遍历每一个轮廓中的数字
for (i, (gX, gY, gW, gH)) in enumerate(locs):
    groupOutput = []
    # 根据坐标提取每一个组
    group = img_resize[gY - 5:gY + gH + 5, gX - 5:gX + gW + 5]
    # 预处理
    group = cv2.threshold(group, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    # 计算每一组的轮廓
    digitCnts, hierarchy = cv2.findContours(group.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    digitCnts = sort_contours(digitCnts, method="left-to-right")[0]

    # 计算每一组中的每一个数值
    for c in digitCnts :
        # 找到当前数值的轮廓，resize成合适的的大小
        (x, y, w, h) = cv2.boundingRect(c)
        roi = group[y - 1:y + h + 1, x - 1:x + w + 1]
        roi = cv2.resize(roi, (58, 88))
        # cv2_show('roi', roi)

        # 计算匹配得分
        scores = []

        # 在模板中计算每一个得分
        for (digit, digitROI) in digits.items() :
            # 模板匹配
            result = cv2.matchTemplate(roi, digitROI, cv2.TM_CCOEFF)
            (_, score, _, _) = cv2.minMaxLoc(result)
            scores.append(score)

        # 得到最合适的数字
        groupOutput.append(str(np.argmax(scores)))
    # 画出来
    cv2.rectangle(img_resize2, (gX - 5, gY - 5), (gX + gW + 5, gY + gH + 5), (0, 0, 255), 1)
    cv2.putText(img_resize2, "".join(groupOutput), (gX, gY - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 200, 205), 1)

# 得到结果
    output.extend(groupOutput)

# 打印结果
print("Credit Card : {}".format("".join(output)))
cv2.imshow("Image", img_resize2)
cv2.waitKey()
