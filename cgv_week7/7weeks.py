import cv2
import keyboard as kb
import numpy as np

print("스페이스바를 누르시면 시작됩니다.")
kb.wait("space bar")

# 이미지 삽입
print("준비된 이미지 입력")
basic_img = cv2.imread('basic_car.jpg', cv2.IMREAD_COLOR)
cv2.imshow('basic_car', basic_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

src = [[704, 483], [1000, 452], [996, 518], [701, 554]]

src_np = np.array(src, dtype=np.float32)

# 각 좌표 정보를 가지고 너비와 높이의 최댓값을 구하여 template의 비율을 측정하였다.
width = max(np.linalg.norm(src_np[0] - src_np[1]), np.linalg.norm(src_np[2] - src_np[3]))
height = max(np.linalg.norm(src_np[0] - src_np[3]), np.linalg.norm(src_np[1] - src_np[2]))

print("번호판 가로세로 비율을 구하였습니다.")
print("가로비 : ", width)
print("세로비 : ", height)
print("계속 진행하시려면 스페이스바를 눌러주세요.")
kb.wait("space bar")

#번호판의 가로 세로 10% 더 크게 조정
width = width*(1.1)
height = height*(1.1)
print("번호판 가로 세로가 10% 늘어났습니다.")
print("가로비 : ", width)
print("세로비 : ", height)

print("계속 진행하시려면 스페이스바를 눌러주세요.")
kb.wait("space bar")

# 이 template의 좌표 정보(x', y')를 저장한다.
dst_np = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)

# 투영행렬 계산
M = cv2.getPerspectiveTransform(src=src_np, dst=dst_np)

print("투영행렬을 구하였습니다.")
print("계속 진행하시려면 스페이스바를 눌러주세요.")
kb.wait("space bar")

# 꼭짓점 좌표
img_circle = basic_img.copy()
Color = (0, 255, 0)
cv2.circle(img_circle, (704, 483), 3, Color, thickness=-1, lineType=cv2.LINE_AA)
cv2.circle(img_circle, (1000, 452), 3, Color, thickness=-1, lineType=cv2.LINE_AA)
cv2.circle(img_circle,  (996, 518), 3, Color, thickness=-1, lineType=cv2.LINE_AA)
cv2.circle(img_circle, (701, 554), 3, Color, thickness=-1, lineType=cv2.LINE_AA)

# 변환 결과 출력
img_normal = cv2.warpPerspective(basic_img, M=M, dsize=(int(width), int(height)))
print("번호판을 디스플레이 하였습니다.")
cv2.imshow('img_circle', img_circle)
cv2.imshow('result', img_normal)

print("계속 진행하시려면 디스플레이를 클릭하고 스페이스바를 눌러주세요.")
cv2.waitKey(0)
cv2.destroyAllWindows()

# 1~2번 끝

#가우시안 블러처리를 하여 노이즈 제거
img_gray = cv2.cvtColor(img_normal, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(img_gray, (3, 3), 0)
print("가우시안 블러처리를 하여 노이즈를 제거하였습니다.")
print("계속 진행하시려면 스페이스바를 눌러주세요.")
kb.wait("space bar")

# 이진화 (binary, otsu, adpative_gaussian)
disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)) #이진화
ret, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
ret, binary = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)
adaptive_gaussian = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
cv2.imshow("otsu", otsu)
cv2.imshow("binary", binary)
cv2.imshow("adaptive", adaptive_gaussian)
cv2.waitKey(0)
cv2.destroyAllWindows()
print("계속 진행하시려면 디스플레이를 클릭 후 스페이스바를 눌러주세요.")

print("이진화를 진행하였습니다. 이진화 결과가 otsu가 가장좋아 이후 otsu 이진화로 진행하였습니다")
print("계속 진행하시려면 스페이스바를 눌러주세요.")
kb.wait("space bar")

# SE 생성
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
print("SE를 생성하였습니다.")
print("계속 진행하시려면 스페이스바를 눌러주세요.")
kb.wait("space bar")

# 모폴리지 연산
otsu_not = cv2.bitwise_not(otsu)
dst_erode = cv2.erode(otsu_not, kernel, iterations=1)
dst_dilate = cv2.dilate(otsu_not,kernel,iterations=1)
dst_open = cv2.morphologyEx(otsu_not, cv2.MORPH_OPEN, kernel)
dst_close = cv2.morphologyEx(otsu_not, cv2.MORPH_CLOSE, kernel)
print("모폴리지 연산 완료.")

cv2.imshow('Before', img_normal)
cv2.imshow('After_erode', dst_erode)
cv2.imshow('After_dilate', dst_dilate)
cv2.imshow('After_morph-erosion-dilation', dst_open)
cv2.imshow('After_morph-dilation-erosion', dst_close)

cv2.waitKey(0)
cv2.destroyAllWindows()


