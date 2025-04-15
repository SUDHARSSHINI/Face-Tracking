import cv2
from cvzone.FaceDetectionModule import FaceDetector
import numpy as np

# Initialize webcam
cap = cv2.VideoCapture(0)
ws, hs = 1280, 720  # Width and height
cap.set(3, ws)
cap.set(4, hs)

if not cap.isOpened():
    print("Camera couldn't Access!!!")
    exit()

# Initialize Face Detector
detector = FaceDetector()

while True:
    success, img = cap.read()
    img, bboxs = detector.findFaces(img, draw=False)

    if bboxs:
        # Get face center coordinates
        fx, fy = bboxs[0]["center"]

        # Convert coordinate to "servo-like" range (180° scale)
        servoX = np.interp(fx, [0, ws], [180, 0])
        servoY = np.interp(fy, [0, hs], [180, 0])

        # Ensure values are within 0-180 range
        servoX = np.clip(servoX, 0, 180)
        servoY = np.clip(servoY, 0, 180)

        # Draw tracking indicators
        cv2.circle(img, (fx, fy), 80, (0, 0, 255), 2)
        cv2.putText(img, f"({fx}, {fy})", (fx+15, fy-15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        cv2.line(img, (0, fy), (ws, fy), (0, 0, 0), 2)  # X-axis line
        cv2.line(img, (fx, hs), (fx, 0), (0, 0, 0), 2)  # Y-axis line
        cv2.circle(img, (fx, fy), 15, (0, 0, 255), cv2.FILLED)
        cv2.putText(img, "TARGET LOCKED", (850, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    else:
        # Draw default target position
        cv2.putText(img, "NO TARGET", (880, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)
        cv2.circle(img, (640, 360), 80, (0, 0, 255), 2)
        cv2.circle(img, (640, 360), 15, (0, 0, 255), cv2.FILLED)
        cv2.line(img, (0, 360), (ws, 360), (0, 0, 0), 2)
        cv2.line(img, (640, hs), (640, 0), (0, 0, 0), 2)

    # Display video feed
    cv2.imshow("Face Tracking", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break

cap.release()
cv2.destroyAllWindows()
