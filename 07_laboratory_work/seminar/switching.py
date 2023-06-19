import cv2
import numpy as np

# Загрузка изображений
img1 = cv2.imread('data/Univ1.jpg')
img2 = cv2.imread('data/Univ2.jpg')

img1 = cv2.imread('data/mai/01.jpg')
img2 = cv2.imread('data/mai/02.jpg')
img3 = cv2.imread('data/mai/03.jpg')
img4 = cv2.imread('data/mai/04.jpg')
img5 = cv2.imread('data/mai/05.jpg')
img6 = cv2.imread('data/mai/06.jpg')
img7 = cv2.imread('data/mai/07.jpg')
img8 = cv2.imread('data/mai/08.jpg')
img9 = cv2.imread('data/mai/09.jpg')
img10 = cv2.imread('data/mai/10.jpg')

# Преобразование изображений в оттенки серого
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Инициализация объекта SIFT (определение ключевых точек)
sift = cv2.xfeatures2d.SIFT_create()

# Нахождение ключевых точек и дескрипторов на обоих изображениях
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

# Нахождение соответствий между дескрипторами на двух изображениях
matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
matches = matcher.knnMatch(des1, des2, k=2)

# Фильтрация соответствий с помощью метрики Lowe's ratio test
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

# Нахождение матрицы гомографии
src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

# Преобразование изображений
panorama = cv2.warpPerspective(img1, H, (img1.shape[1] + img2.shape[1], img2.shape[0]))
panorama[0:img2.shape[0], 0:img2.shape[1]] = img2

# Отображение результата
cv2.imshow('Panorama', panorama)
cv2.waitKey()
cv2.destroyAllWindows()