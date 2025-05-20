import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

def calcular_flexiones(landmarks):
    dedos = []
    dedos_indices = [
        (4, 3, 2),   # Pulgar
        (8, 7, 6),   # Índice
        (12,11,10),  # Medio
        (16,15,14),  # Anular
        (20,19,18)   # Meñique
    ]
    for fingertip, pip, mcp in dedos_indices:
        dist_tip_pip = np.linalg.norm(np.array([
            landmarks[fingertip].x - landmarks[pip].x,
            landmarks[fingertip].y - landmarks[pip].y
        ]))
        dist_pip_mcp = np.linalg.norm(np.array([
            landmarks[pip].x - landmarks[mcp].x,
            landmarks[pip].y - landmarks[mcp].y
        ]))
        flexion = dist_tip_pip / (dist_pip_mcp + 1e-6)
        dedos.append(flexion)
    return dedos

def detectar_letra(flexiones):
    # Aquí ajustamos rangos reales más amplios
    if flexiones[0] < 1.3 and all(f > 1.7 for f in flexiones[1:]):
        return "A"
    if flexiones[0] > 1.5 and all(1.0 < f < 1.6 for f in flexiones[1:]):
        return "B"
    return "?"

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resultado = hands.process(frame_rgb)

    letra_detectada = "..."

    if resultado.multi_hand_landmarks:
        for mano in resultado.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, mano, mp_hands.HAND_CONNECTIONS)
            flexiones = calcular_flexiones(mano.landmark)
            letra_detectada = detectar_letra(flexiones)

            # Mostrar flexiones en pantalla
            for i, f in enumerate(flexiones):
                cv2.putText(frame, f"Dedo {i}: {f:.2f}", (10, 100 + i * 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.putText(frame, f"Letra detectada: {letra_detectada}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Traductor LSM - Vision", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
