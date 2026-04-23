import cv2
import mediapipe as mp
import numpy as np

mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.7)

mask = cv2.imread("mascara.png", cv2.IMREAD_UNCHANGED)

def overlay_mask(frame, mask, x, y, w, h):
    mask_resized = cv2.resize(mask, (w, h))

    mask_rgb = mask_resized[:, :, :3]
    mask_alpha = mask_resized[:, :, 3] / 255.0

    roi = frame[y:y+h, x:x+w]

    for c in range(3):
        roi[:, :, c] = (1 - mask_alpha) * roi[:, :, c] + mask_alpha * mask_rgb[:, :, c]

    frame[y:y+h, x:x+w] = roi

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)

    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape

            x = int(bboxC.xmin * iw) - 20
            y = int(bboxC.ymin * ih) - 40
            w = int(bboxC.width * iw) + 40
            h = int(bboxC.height * ih) + 40

            overlay_mask(frame, mask, x, y, w, h)

    cv2.imshow("Mascara Animada", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
