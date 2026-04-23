import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

def reconocer_letra(hand_landmarks, frame):
    h, w, _ = frame.shape
    dedos = [(int(hand_landmarks.landmark[i].x * w), int(hand_landmarks.landmark[i].y * h)) for i in range(21)]

    pulgar, indice, medio, anular, menique = dedos[4], dedos[8], dedos[12], dedos[16], dedos[20]

    for i, (x, y) in enumerate(dedos):
        cv2.circle(frame, (x, y), 5, (233, 23, 0), -1)
        cv2.putText(frame, str(i), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    cv2.putText(frame, f'({int(pulgar[0])}, {int(pulgar[1])})', (pulgar[0], pulgar[1] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (245, 0, 0), 2, cv2.LINE_AA)

    cv2.line(frame, (int(pulgar[0]), int(pulgar[1])), (int(indice[0]), int(indice[1])), (244, 34, 12), 2)

    distancia_pulgar_indice = np.linalg.norm(np.array(pulgar) - np.array(indice))
    distancia_indice_medio = np.linalg.norm(np.array(indice) - np.array(medio))

    cv2.putText(frame, f'({distancia_pulgar_indice})', (pulgar[0]-40, pulgar[1] - 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.circle(frame, (pulgar[0]+20, pulgar[1]+20), int(distancia_indice_medio), (34, 234, 65), -1)

    if distancia_pulgar_indice < 30 and distancia_indice_medio > 50:
        return "A"
    elif indice[1] < medio[1] and medio[1] < anular[1] and anular[1] < menique[1]:
        return "B"
    elif distancia_pulgar_indice > 50 and distancia_indice_medio > 50:
        return "C"

    return "Desconocido"

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            letra_detectada = reconocer_letra(hand_landmarks, frame)

            cv2.putText(frame, f"Letra: {letra_detectada}", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("Reconocimiento de Letras", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
