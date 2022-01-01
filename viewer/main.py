import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Mam nadzieję, że sztuczna inteligencja jest przydatna

def getFaces(image):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 2
    thickness = 2

    heatmap_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(heatmap_gray, 1.1, 4)
    image_with_rectangles = np.copy(heatmap_gray)
    
    for (x, y, w, h) in faces:
        image_with_rectangles = cv2.rectangle(image_with_rectangles, (x, y), (x + w, y + h), (255, 0, 0), 3)
        image_with_rectangles = cv2.putText(image_with_rectangles, "Not a Dog",
                                            (x + w, y + h), font,
                                            fontScale, (255, 0, 0), thickness, cv2.LINE_AA)
    return image_with_rectangles


def recVid():
    #webcam fix to correct dimensions
    cap = cv2.VideoCapture(0)
    cap.set(3, 1920)
    cap.set(4, 1080)


    fourcc = 0x7634706d
    out = cv2.VideoWriter(os.path.join("data", 'faces.mp4'), fourcc, 20.0, (1920, 1080))

    while (True):
        ret, frame = cap.read()
        out.write(getFaces(frame))
        cv2.imshow('frame', frame)
        c = cv2.waitKey(1)
        if c & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


def main():
    # picture
    pth = os.path.join("data", "dom.png")
    image = getFaces(cv2.imread(pth))
    plt.imshow(image)
    plt.show()

    # video
    #recVid()
    print("Kurwa!")


if __name__ == '__main__':
    main()
