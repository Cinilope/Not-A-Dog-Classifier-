import os
import numpy as np
import cv2

# Mam nadzieję, że sztuczna inteligencja jest przydatna

CASCADE_PATH = os.path.join(os.path.dirname(__file__),
                            'haarcascade_frontalface_default.xml')


def getFaces(image):
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
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

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        processed = getFaces(frame)
        out.write(processed)
        cv2.imshow('frame', processed)
        c = cv2.waitKey(1)
        if c & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


def main():
    # Run real-time face detection on webcam feed
    recVid()


if __name__ == '__main__':
    main()
