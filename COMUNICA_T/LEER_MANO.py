import cv2
import mediapipe as mp
import numpy as np

# Inicializar MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# Función para calcular la "flexión" de cada dedo
def obtener_flexiones(landmarks):
    tips = [4, 8, 12, 16, 20]  # Pulgar, Índice, Medio, Anular, Meñique
    pip_joints = [2, 6, 10, 14, 18]
    flexiones = []

    for tip, pip in zip(tips, pip_joints):
        distancia = np.linalg.norm(np.array([
            landmarks[tip].x - landmarks[pip].x,
            landmarks[tip].y - landmarks[pip].y,
            landmarks[tip].z - landmarks[pip].z
        ]))
        flexiones.append(distancia)

    return flexiones

# Función para clasificar una letra en base a las flexiones
def detectar_letra(flexiones):
    p, i, m, a, e = flexiones

    flexionado = lambda x: x < 0.12
    extendido = lambda x: x > 0.135
    medio = lambda x: 0.12 <= x <= 0.135

    print(f"Flexiones: {flexiones}")

    # Letra A
    if extendido(p) and all(flexionado(d) for d in [i, m, a, e]):
        print("Letra detectada: A")
        return 'A'

    # Letra B
    if flexionado(p) and all(extendido(d) or medio(d) for d in [i, m, a, e]):
        print("Letra detectada: B")
        return 'B'

    # Letra C
    if all(medio(d) for d in [p, i, m, a, e]):
        print("Letra detectada: C")
        return 'C'

    # Letra D
    if extendido(i) and all(flexionado(d) for d in [p, m, a, e]):
        print("Letra detectada: D")
        return 'D'

    print("Letra detectada: ?")
    return '?'

# Captura de cámara
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    letra = '?'
    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
            flexiones = obtener_flexiones(handLms.landmark)
            print("Flexiones:", flexiones)
            letra = detectar_letra(flexiones)

    # Mostrar resultado
    cv2.rectangle(frame, (0, 0), (100, 80), (0, 0, 0), -1)
    cv2.putText(frame, letra, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
    cv2.imshow("Reconocimiento de señas LSM", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
