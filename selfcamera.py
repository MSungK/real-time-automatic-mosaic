import cv2

def capture():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()

    cv2.imwrite('my_image.jpg',frame)
    cap.release()

if __name__ == '__main__':
    capture()
