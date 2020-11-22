
count_triangle = 0
count_rectangle = 0
count_circle = 0

                            #python 3.7 version
import cv2                  #opencv 4.3.0.36 version
import numpy as np          #numpy 1.19.0 version
frameWidth = 640
frameHeight = 480
cap = cv2.VideoCapture("RealLastNoMore.avi")
cap.set(3, frameWidth)
cap.set(4, frameHeight)

MIN_MATCH = 5

# ORB 검출기 생성

detector = cv2.ORB_create(1000)

# Flann 추출기 생성

FLANN_INDEX_LSH = 6

index_params= dict(algorithm = FLANN_INDEX_LSH,

                   table_number = 6,

                   key_size = 12,

                   multi_probe_level = 1)

search_params=dict(checks=32)

matcher = cv2.FlannBasedMatcher(index_params, search_params)

# 카메라 캡쳐 연결 및 객체 저장
ret, img = cap.read()
rect = cv2.selectROI('Select Window', img, fromCenter=False, showCrosshair=True)
cv2.destroyWindow('Select Window')
imgStack=img
ret, frame = cap.read()
dst = frame.copy()


x, y, w, h = rect
cv2.rectangle(img=frame, pt1=(x, y), pt2=(x + w, y + h), color=(0, 0, 255), thickness=2)
dst = dst[y:y + h, x:x + w]
img1=dst
#############################################
ret, img = cap.read()
#img=cv2.resize(img, dsize=(640, 480), interpolation=cv2.INTER_AREA)
rect = cv2.selectROI('Select Window', img, fromCenter=False, showCrosshair=True)
cv2.destroyWindow('Select Window')

ret, frame = cap.read()
dst = frame.copy()
x, y, w, h = rect
cv2.rectangle(img=frame, pt1=(x, y), pt2=(x + w, y + h), color=(0, 0, 255), thickness=2)
dst = dst[y:y + h, x:x + w]
img3=dst
###########################################
ret, img = cap.read()
rect = cv2.selectROI('Select Window', img, fromCenter=False, showCrosshair=True)
cv2.destroyWindow('Select Window')

ret, frame = cap.read()
dst = frame.copy()
x, y, w, h = rect
cv2.rectangle(img=frame, pt1=(x, y), pt2=(x + w, y + h), color=(0, 0, 255), thickness=2)
dst = dst[y:y + h, x:x + w]
img4=dst
##############################################
ret, img = cap.read()
rect = cv2.selectROI('Select Window', img, fromCenter=False, showCrosshair=True)
cv2.destroyWindow('Select Window')

ret, frame = cap.read()
dst = frame.copy()
x, y, w, h = rect
cv2.rectangle(img=frame, pt1=(x, y), pt2=(x + w, y + h), color=(0, 0, 255), thickness=2)
dst = dst[y:y + h, x:x + w]
img5=dst
################################################
def stackImages(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]),
                                                None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    return ver

# 이 부분이 도형 컨투어, 정보를 나타내는 부분입니다
def getContours(img):
    Area = 1000
    global count_triangle
    global count_rectangle
    global count_circle
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > Area:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            #삼각형 검출
            if len(approx) == 3:
                count_triangle += 1
                return 1
            #사각형 검출
            elif 4 <= len(approx) <= 6:
                count_rectangle += 1
                return 2
            #원 검출
            elif len(approx) >= 8:
                count_circle += 1
                return 3
count=0
avg_x=0
avg_y=0
#ROi 이미지와 본이미지를 비교하며 특징점 검출
def detect(img1,img2,avg_x,avg_y):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    global imgStack
    global count
    # 키포인트와 디스크립터 추출

    kp1, desc1 = detector.detectAndCompute(gray1, None)

    kp2, desc2 = detector.detectAndCompute(gray2, None)

    # k=2로 knnMatch

    matches = matcher.knnMatch(desc1, desc2, 2)

    # 이웃 거리의 75%로 좋은 매칭점 추출

    ratio = 0.75

    good_matches = [m[0] for m in matches \
 \
                    if len(m) == 2 and m[0].distance < m[1].distance * ratio]


    # 좋은 매칭점 최소 갯수 이상 인 경우

    if len(good_matches) > MIN_MATCH:

        # 좋은 매칭점으로 원본과 대상 영상의 좌표 구하기

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches])

        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches])

        mtrx, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        if mask.sum() > MIN_MATCH:  # 정상치 매칭점 최소 갯수 이상 인 경우

            # 특징점들의 중심좌표 추출
            arr_x = []
            arr_y = []
            for i, (m, n) in enumerate(dst_pts):
                arr_x.append(m)
                arr_y.append(n)
                img2 = cv2.circle(img2, (m, n), 1, (255, 0, 0), -1)
            avg_x.append(sum(arr_x, 0.0) / len(arr_x))
            avg_y.append(sum(arr_y, 0.0) / len(arr_y))

            imgContour = img2
            imgBlur = cv2.GaussianBlur(img1, (7, 7), 1)
            imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)
            imgCanny = cv2.Canny(imgGray, 255, 0)
            kernel = np.ones((7, 7))
            imgDil = cv2.morphologyEx(imgCanny, cv2.MORPH_CLOSE, kernel)

            # 블러,캐니,클로즈 전처리 후 컨투어
            col=getContours(imgDil)
            #삼각형일 경우
            if(col==1):
                imgContour = cv2.circle(imgContour, (int(avg_x[count]), int(avg_y[count])), 10, (255, 0, 0), -1)
            #사각형일 경우
            elif(col==2):
                imgContour = cv2.circle(imgContour, (int(avg_x[count]), int(avg_y[count])), 10, (0, 255, 0), -1)
            #원일 경우
            elif(col==3):
                imgContour = cv2.circle(imgContour, (int(avg_x[count]), int(avg_y[count])), 10, (0, 0, 255), -1)
            #삼각형, 사각형, 원이 아닐경우
            else:
                imgContour = cv2.circle(imgContour, (int(avg_x[count]), int(avg_y[count])), 10, (255, 255, 255), -1)
            imgStack = stackImages(0.8, ([img11, imgContour]))
            count +=1


while cap.isOpened():
    success, img = cap.read()
    if success == False:
        break;
    img11=img.copy()
    avg_x=[]
    avg_y=[]
    count=0
    img2=img
    #객체 4개와 영상의 특징점 검출
    detect(img1,img2,avg_x,avg_y)
    detect(img3,img2,avg_x,avg_y)
    detect(img4,img2,avg_x,avg_y)
    detect(img5,img2,avg_x,avg_y)
    #영상에 포함되어 있는 삼각형,사각형,원의 개수 표시
    cv2.putText(imgStack, "triangle :" + str(count_triangle), (860, 30), cv2.FONT_HERSHEY_COMPLEX, .7,
                (255, 0, 0), 2)
    cv2.putText(imgStack, "rectangle :" + str(count_rectangle), (860, 50), cv2.FONT_HERSHEY_COMPLEX, .7,
                (0, 255, 0), 2)
    cv2.putText(imgStack, "circle :" + str(count_circle), (860, 70), cv2.FONT_HERSHEY_COMPLEX, .7,
                (0, 0, 255), 2)

    # 결과 출력
    cv2.imshow("Result", imgStack)
    count_triangle = 0
    count_rectangle = 0
    count_circle = 0
    if cv2.waitKey(1) & 0xFF == ord('q'):
       break
cap.release()
