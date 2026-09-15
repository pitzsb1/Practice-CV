import cv2 as cv

img=cv.imread('/Users/pitzsb1/computer-vision-practice/ch3/rose.png')
patch=img[250:350,170:270,:]

def draw(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        # 왼쪽 클릭 → 빨간색 직사각형
        cv.rectangle(img, (x, y), (x+200, y+200), (0, 0, 255), 2)

    cv.imshow('Drawing', img)

cv.namedWindow('Drawing')
cv.imshow('Drawing', img)

cv.setMouseCallback('Drawing', draw)

while True:
    if cv.waitKey(1) == ord('q'):
        cv.destroyAllWindows()
        break