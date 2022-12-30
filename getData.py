import cv2
import os


def getData(name):
    img_counter = 0
    path = "E:/Project/PBL4/Dataset/FaceData/raw/" + name
    os.makedirs(path)
    cam = cv2.VideoCapture(0)
    while True:
        ret, frame = cam.read()
        if not ret:
            print("failed to grab frame")
            break
        cv2.putText(frame, f"{img_counter}", (100, 100),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0), 2)
        cv2.imshow("Take a photo", frame)
        img_name = path + "/image_{}.jpg".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1
        if img_counter == 200:
            break
        if cv2.waitKey(20) & 0xFF == ord('q'):
            print("Escape hit, closing...")
            break

    cam.release()
    cv2.destroyAllWindows()

