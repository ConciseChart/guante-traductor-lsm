import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import math

# Inicialización de MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Estructuras para ambas manos
manos = {
    "left": {
        "trayectoria_indice": deque(maxlen=20),
        "trayectoria_menique": deque(maxlen=20),
        "trayectoria_indice_k": deque(maxlen=10),  # Para seguir movimiento de K
        "trayectoria_medio_k": deque(maxlen=10),   # Para seguir movimiento de K
        "letra": "...",
        "cooldown": 0,
        "j_detectada": False,
        "z_detectada": False
    },
    "right": {
        "trayectoria_indice": deque(maxlen=20),
        "trayectoria_menique": deque(maxlen=20),
        "trayectoria_indice_k": deque(maxlen=10),  # Para seguir movimiento de K
        "trayectoria_medio_k": deque(maxlen=10),   # Para seguir movimiento de K
        "letra": "...",
        "cooldown": 0,
        "j_detectada": False,
        "z_detectada": False
    }
}


# Función corregida para detección de dedos (manos frontales)
def dedos_extendidos(landmarks, hand_type):
    dedos = []

    # Lógica corregida para el pulgar (manos frontales)
    if hand_type == "left":
        # Pulgar izquierdo: x aumenta hacia la derecha (mano frontal)
        dedos.append(1 if landmarks[4].x > landmarks[3].x else 0)
    else:
        # Pulgar derecho: x aumenta hacia la izquierda (mano frontal)
        dedos.append(1 if landmarks[4].x < landmarks[3].x else 0)

    # Resto de dedos (y-axis)
    for tip in [8, 12, 16, 20]:
        base = tip - 2
        dedos.append(1 if landmarks[tip].y < landmarks[base].y else 0)
    return dedos


# Función de curvatura (igual)
def detectar_curvatura(trayectoria):
    if len(trayectoria) < 10:
        return False
    puntos = np.array(trayectoria)
    distancia_directa = np.linalg.norm(puntos[-1] - puntos[0])
    distancia_total = np.sum(np.linalg.norm(np.diff(puntos, axis=0), axis=1))
    return distancia_total / (distancia_directa + 1e-6) > 1.8 and distancia_total > 0.1


def calcular_angulo(p1, p2, p3):
    try:
        v1 = np.array([p1.x - p2.x, p1.y - p2.y])
        v2 = np.array([p3.x - p2.x, p3.y - p2.y])

        # Verificar que los vectores no sean cero
        if np.all(v1 == 0) or np.all(v2 == 0):
            return 0

        cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * (np.linalg.norm(v2) + 1e-6))
        angulo = np.arccos(np.clip(cos_theta, -1.0, 1.0))
        return np.degrees(angulo)
    except Exception as e:
        print(f"Error en calcular_angulo: {e}")
        return 0


def detectar_letra(landmarks, hand_type):
    try:
        mano = manos[hand_type]

        if mano["cooldown"] > 0:
            mano["cooldown"] -= 1
            return None

        # Detectar dedos extendidos
        dedos = dedos_extendidos(landmarks, hand_type)
        print(f"Dedos detectados ({hand_type}): {dedos}")

        # Obtener orientación de la mano
        y_base = landmarks[0].y  # Coordenada Y de la muñeca
        y_medio = landmarks[12].y  # Coordenada Y del dedo medio
        umbral_orientacion = 0.10  # Umbral para determinar la orientación

        # Detectar letra N
        if dedos == [0, 0, 0, 1, 1]:
            if y_medio > y_base + umbral_orientacion:  # Mano orientada hacia abajo
                print("Se detectó N")
                return "N"

        # Detectar letra M
        if dedos == [0, 0, 0, 0, 0]:
            if y_medio > y_base + umbral_orientacion:  # Mano orientada hacia abajo
                print("Se detectó M")
                return "M"

        # Detectar letra W
        if dedos == [0, 1, 1, 1, 0]:
            if y_medio < y_base - umbral_orientacion:  # Mano orientada hacia arriba
                print("Se detectó W")
                return "W"

        # ... resto del código de detección de otras letras ...

        dedos = dedos_extendidos(landmarks, hand_type)

        # A
        if dedos == [1, 0, 0, 0, 0]:
            return "A"

        # B
        if dedos == [0, 1, 1, 1, 1]:
            return "B"

        # C
        if all(0 <= d <= 1 for d in dedos):
            angulos_dedos = [
                calcular_angulo(landmarks[5], landmarks[6], landmarks[8]),
                calcular_angulo(landmarks[9], landmarks[10], landmarks[12]),
                calcular_angulo(landmarks[13], landmarks[14], landmarks[16]),
                calcular_angulo(landmarks[17], landmarks[18], landmarks[20])
            ]

            alturas_dedos = [landmarks[8].y, landmarks[12].y, landmarks[16].y, landmarks[20].y]
            altura_promedio = sum(alturas_dedos) / len(alturas_dedos)

            if all(70 < angulo < 150 for angulo in angulos_dedos):
                if hand_type == "left":
                    if (landmarks[4].x > landmarks[8].x and
                            landmarks[4].y > landmarks[8].y - 0.1 and
                            landmarks[4].y < landmarks[8].y + 0.2):
                        if (abs(landmarks[8].y - altura_promedio) < 0.1 and
                                abs(landmarks[12].y - altura_promedio) < 0.1 and
                                abs(landmarks[16].y - altura_promedio) < 0.1 and
                                abs(landmarks[20].y - altura_promedio) < 0.1):
                            return "C"
                else:
                    if (landmarks[4].x < landmarks[8].x and
                            landmarks[4].y > landmarks[8].y - 0.1 and
                            landmarks[4].y < landmarks[8].y + 0.2):
                        if (abs(landmarks[8].y - altura_promedio) < 0.1 and
                                abs(landmarks[12].y - altura_promedio) < 0.1 and
                                abs(landmarks[16].y - altura_promedio) < 0.1 and
                                abs(landmarks[20].y - altura_promedio) < 0.1):
                            return "C"

        # O
        if all(0 <= d <= 1 for d in dedos):
            angulos_dedos = [
                calcular_angulo(landmarks[5], landmarks[6], landmarks[8]),
                calcular_angulo(landmarks[9], landmarks[10], landmarks[12]),
                calcular_angulo(landmarks[13], landmarks[14], landmarks[16]),
                calcular_angulo(landmarks[17], landmarks[18], landmarks[20])
            ]

            alturas_dedos = [landmarks[8].y, landmarks[12].y, landmarks[16].y, landmarks[20].y]
            altura_promedio = sum(alturas_dedos) / len(alturas_dedos)
            
            # Distancia entre el pulgar y el índice para la O
            dist_pulgar_indice = np.linalg.norm([
                landmarks[4].x - landmarks[8].x,
                landmarks[4].y - landmarks[8].y
            ])

            if all(50 < angulo < 120 for angulo in angulos_dedos):  # Ángulos más cerrados que la C
                if hand_type == "left":
                    if (landmarks[4].x > landmarks[8].x and
                            landmarks[4].y > landmarks[8].y - 0.05 and
                            landmarks[4].y < landmarks[8].y + 0.1 and
                            dist_pulgar_indice < 0.05):  # Distancia más cercana para la O
                        if (abs(landmarks[8].y - altura_promedio) < 0.08 and
                                abs(landmarks[12].y - altura_promedio) < 0.08 and
                                abs(landmarks[16].y - altura_promedio) < 0.08 and
                                abs(landmarks[20].y - altura_promedio) < 0.08):
                            return "O"
                else:
                    if (landmarks[4].x < landmarks[8].x and
                            landmarks[4].y > landmarks[8].y - 0.05 and
                            landmarks[4].y < landmarks[8].y + 0.1 and
                            dist_pulgar_indice < 0.05):  # Distancia más cercana para la O
                        if (abs(landmarks[8].y - altura_promedio) < 0.08 and
                                abs(landmarks[12].y - altura_promedio) < 0.08 and
                                abs(landmarks[16].y - altura_promedio) < 0.08 and
                                abs(landmarks[20].y - altura_promedio) < 0.08):
                            return "O"

        # E
        if dedos == [0, 0, 0, 0, 0]:
            return "E"

        # F
        if dedos == [0, 0, 1, 1, 1]:
            return "F"

        # G y H
        if dedos == [1, 1, 0, 0, 0] or dedos == [1, 1, 1, 0, 0]:
            indice_vertical = landmarks[8].y - landmarks[5].y
            indice_horizontal = landmarks[8].x - landmarks[5].x

            if abs(indice_horizontal) > abs(indice_vertical):
                if hand_type == "left":
                    if indice_horizontal > 0:
                        if dedos == [1, 1, 1, 0, 0]:
                            return "H"
                        return "G"
                else:
                    if indice_horizontal < 0:

                        if dedos == [1, 1, 1, 0, 0]:
                            return "H"
                        return "G"

        # I y J
        if dedos == [0, 0, 0, 0, 1]:
            # Obtener la trayectoria del meñique
            trayectoria = manos[hand_type]["trayectoria_menique"]

            # Verificar si hay un movimiento en forma de J
            if len(trayectoria) >= 10:
                # Calcular el movimiento total en X e Y
                inicio = np.array(trayectoria[0])
                fin = np.array(trayectoria[-1])
                # Calcular diferencias en X e Y
                diff_x = abs(fin[0] - inicio[0])
                diff_y = fin[1] - inicio[1]

                # Si hay suficiente movimiento horizontal y vertical hacia abajo
                if diff_y > 0.05 and diff_x > 0.02:
                    mano["j_detectada"] = True
                    return "J"

            # Si no hay movimiento significativo, es una I
            mano["j_detectada"] = False
            return "I"

        # K y P (tienen configuración similar de dedos)
        if dedos == [1, 1, 1, 0, 0]:
            # Actualizar trayectorias de los dedos para K
            mano["trayectoria_indice_k"].append((landmarks[8].x, landmarks[8].y))
            mano["trayectoria_medio_k"].append((landmarks[12].x, landmarks[12].y))

            # Verificar movimiento para K
            movimiento_detectado = False
            if len(mano["trayectoria_indice_k"]) >= 5 and len(mano["trayectoria_medio_k"]) >= 5:
                # Calcular el movimiento de los dedos
                mov_indice = np.std([p[1] for p in mano["trayectoria_indice_k"]])
                mov_medio = np.std([p[1] for p in mano["trayectoria_medio_k"]])

                # Si hay suficiente movimiento en ambos dedos
                if mov_indice > 0.01 and mov_medio > 0.01:
                    movimiento_detectado = True

            # Verificar la posición del pulgar
            if hand_type == "left":
                pulgar_correcto = landmarks[4].x > landmarks[8].x and landmarks[4].x < landmarks[12].x
            else:
                pulgar_correcto = landmarks[4].x < landmarks[8].x and landmarks[4].x > landmarks[12].x

            if pulgar_correcto:
                # Si hay movimiento, es K
                if movimiento_detectado:
                    return "K"
                # Si no hay movimiento y los dedos están en la posición correcta, es P
                else:
                    # Verificar que el índice y medio estén extendidos y juntos
                    dist_indice_medio = np.linalg.norm([
                        landmarks[8].x - landmarks[12].x,
                        landmarks[8].y - landmarks[12].y
                    ])
                    # Si los dedos están suficientemente juntos
                    if dist_indice_medio < 0.05:
                        return "P"

        # Inicializamos la trayectoria si no existe
        if "trayectoria_k" not in manos[hand_type]:
            manos[hand_type]["trayectoria_k"] = []

        # Configuración para K y P (dedos índice y medio extendidos, anular y meñique flexionados)
        if dedos == [1, 1, 1, 0, 0]:
            # Obtener el punto medio entre los dedos índice y medio
            punto_medio = np.array([
                (landmarks[8].x + landmarks[12].x) / 2,
                (landmarks[8].y + landmarks[12].y) / 2
            ])

            # Acceder a los datos de la mano correspondiente
            mano = manos[hand_type]
            mano["trayectoria_k"].append(punto_medio)

            # Asegurarnos de que tenemos suficientes puntos en la trayectoria para evaluar
            if len(mano["trayectoria_k"]) >= 5:
                inicio = np.array(mano["trayectoria_k"][0])
                fin = np.array(mano["trayectoria_k"][-1])

                # Calcular la diferencia en el movimiento
                diff_x = abs(fin[0] - inicio[0])
                diff_y = abs(fin[1] - inicio[1])

                # 1. Detectar P: si no hay movimiento significativo en X o Y
                if diff_x < 0.01 and diff_y < 0.01:  # Está completamente estática
                    mano["trayectoria_k"].clear()  # Limpiar trayectoria
                    return "P"

                # 2. Detectar K: si hay movimiento horizontal significativo
                if diff_x > 0.05:  # Movimiento claro en X (horizontal)
                    mano["trayectoria_k"].clear()  # Limpiar trayectoria
                    return "K"

            # En caso de no poder distinguir, retornamos None
            return None
        # Inicializamos la trayectoria si no existe
        if "trayectoria_z" not in manos[hand_type]:
            manos[hand_type]["trayectoria_z"] = []
        # I y J
        if dedos == [0, 1, 0, 0, 0]:
            # Obtener la trayectoria del meñique
            trayectoria = manos[hand_type]["trayectoria_menique"]

            # Verificar si hay un movimiento en forma de J
            if len(trayectoria) >= 10:
                # Calcular el movimiento total en X e Y
                inicio = np.array(trayectoria[0])
                fin = np.array(trayectoria[-1])
                # Calcular diferencias en X e Y
                diff_x = abs(fin[0] - inicio[0])
                diff_y = fin[1] - inicio[1]

                # Si hay suficiente movimiento horizontal y vertical hacia abajo
                if diff_y > 0.05 and diff_x > 0.02:
                    mano["j_detectada"] = True
                    return "Z"

            # Si no hay movimiento significativo, es una I
            mano["j_detectada"] = False
            return "D"

            # Si no cumple las condiciones para detección, retorna None
            print("No se cumplen las condiciones para detectar Z.")  # Log no detectado
            return None
        if dedos == [1, 0, 1, 1, 1]:
            return "Q"
        # L
        if dedos == [1, 1, 0, 0, 0]:
            indice_vertical = landmarks[8].y - landmarks[5].y
            indice_horizontal = landmarks[8].x - landmarks[5].x
            if abs(indice_vertical) > abs(indice_horizontal) and indice_vertical < 0:
                return "L"

        # M y W (misma configuración de dedos extendidos)
        if dedos == [1, 1, 1, 1, 1]:
            # Calcular relación posición dedo medio vs base de la mano
            y_medio = landmarks[12].y
            y_base = landmarks[0].y

            # Umbral para detección de orientación (ajustar según necesidad)
            umbral_orientacion = 0.15  # 15% del tamaño de la mano

            # W: dedos hacia arriba (palma frontal)
            if y_medio < y_base - umbral_orientacion:
                return "W"
            # M: dedos hacia abajo (palma hacia adentro)
            elif y_medio > y_base + umbral_orientacion:
                return "M"

        # U y V
        if dedos == [0, 1, 1, 0, 0]:
            dist_dedos = np.linalg.norm([
                landmarks[8].x - landmarks[12].x,
                landmarks[8].y - landmarks[12].y
            ])
            if dist_dedos < 0.05:
                return "U"
            return "V"

        # W
        if dedos == [0, 1, 1, 1, 0]:
            return "W"

        if dedos == [1, 1, 0, 0, 1]:
            return "X"
        # Y
        if dedos == [1, 0, 0, 0, 1] or dedos == [1, 1, 0, 0, 1]:
            return "Y"

        return None

    except Exception as e:
        print(f"Error en detectar_letra:", e)
        return None


# Captura de video
cap = cv2.VideoCapture(0)

# En el bucle principal, agregar manejo de errores
while True:
    try:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resultado = hands.process(frame_rgb)

        # En el bucle principal, después de procesar los landmarks
        if resultado.multi_hand_landmarks:
            for hand_idx, mano in enumerate(resultado.multi_hand_landmarks):
                try:
                    handedness = resultado.multi_handedness[hand_idx].classification[0].label.lower()
                    landmarks = mano.landmark
                    dedos = dedos_extendidos(landmarks, handedness)

                    # Calcular orientación del índice para debug
                    if dedos == [1, 1, 0, 0, 0]:  # Posible L o G
                        orientacion = math.degrees(math.atan2(
                            landmarks[8].y - landmarks[5].y,
                            landmarks[8].x - landmarks[5].x
                        ))

                        # Mostrar información de debug
                        y_base = 150
                        cv2.putText(frame, f"Mano: {handedness}", (10, y_base + hand_idx * 90),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        cv2.putText(frame, f"Dedos: {dedos}", (10, y_base + 20 + hand_idx * 90),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        cv2.putText(frame, f"Orientación: {int(orientacion)}°",
                                    (10, y_base + 40 + hand_idx * 90),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                        # Dibujar línea de orientación del índice
                        pt1 = (int(landmarks[5].x * frame.shape[1]), int(landmarks[5].y * frame.shape[0]))
                        pt2 = (int(landmarks[8].x * frame.shape[1]), int(landmarks[8].y * frame.shape[0]))
                        cv2.line(frame, pt1, pt2, (0, 255, 255), 2)

                    # Agregar en la sección de debug del bucle principal
                    if dedos == [0, 1, 1, 1, 0]:  # Para M y W
                        # Mostrar información de debug
                        indice_y = landmarks[8].y - landmarks[5].y
                        medio_y = landmarks[12].y - landmarks[9].y
                        anular_y = landmarks[16].y - landmarks[13].y

                        cv2.putText(frame, f"Indice Y: {indice_y:.2f}", (10, y_base + 60 + hand_idx * 90),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        cv2.putText(frame, f"Medio Y: {medio_y:.2f}", (10, y_base + 80 + hand_idx * 90),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        cv2.putText(frame, f"Anular Y: {anular_y:.2f}", (10, y_base + 100 + hand_idx * 90),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                except Exception as e:
                    print(f"Error en visualización: {e}")

        if resultado.multi_hand_landmarks:
            for hand_idx, mano in enumerate(resultado.multi_hand_landmarks):
                try:
                    handedness = resultado.multi_handedness[hand_idx].classification[0].label.lower()
                    if handedness not in ["left", "right"]:
                        continue

                    mp_drawing.draw_landmarks(frame, mano, mp_hands.HAND_CONNECTIONS)
                    landmarks = mano.landmark

                    # Actualizar trayectorias de forma segura
                    try:
                        x_indice = landmarks[8].x
                        y_indice = landmarks[8].y
                        x_menique = landmarks[20].x
                        y_menique = landmarks[20].y

                        manos[handedness]["trayectoria_indice"].append(np.array([x_indice, y_indice]))
                        manos[handedness]["trayectoria_menique"].append(np.array([x_menique, y_menique]))

                        letra = detectar_letra(landmarks, handedness)
                        if letra:
                            manos[handedness]["letra"] = letra
                            manos[handedness]["cooldown"] = 20
                    except Exception as e:
                        print(f"Error al procesar landmarks: {e}")
                        continue
                except Exception as e:
                    print(f"Error al procesar mano: {e}")
                    continue

        # Actualizar cooldowns y mostrar texto
        for hand_type in ["left", "right"]:
            if manos[hand_type]["cooldown"] > 0:
                manos[hand_type]["cooldown"] -= 1

        cv2.putText(frame, f"Izquierda: {manos['left']['letra']}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Derecha: {manos['right']['letra']}", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Traductor LSM A-Z (2 Manos)", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break
    except Exception as e:
        print(f"Error en el bucle principal: {e}")
        continue

# Bucle principal donde procesamos cada cuadro de video
while True:
    # Aquí va el procesamiento de cada frame, detección de landmarks, etc.
    # Suponiendo que ya tienes el frame procesado y landmarks detectados...

    if dedos == [1, 1, 1, 1, 1]:  # Cuando detecte que todos los dedos están extendidos
        # Dibujar línea de referencia para depuración
        base_x = int(landmarks[0].x * frame.shape[1])  # Coordenada X del landmark 0 (muñeca)
        base_y = int(landmarks[0].y * frame.shape[0])  # Coordenada Y del landmark 0 (muñeca)
        medio_y = int(landmarks[12].y * frame.shape[0])  # Coordenada Y del landmark 12 (punta del dedo medio)

        # Línea base (línea horizontal en la muñeca)
        cv2.line(frame, (base_x - 50, base_y), (base_x + 50, base_y), (200, 200, 200), 2)

        # Línea del dedo medio (línea horizontal en el punto del dedo medio)
        cv2.line(frame, (base_x - 50, medio_y), (base_x + 50, medio_y), (0, 255, 255), 2)

        # Texto de depuración que muestra la diferencia en Y
        cv2.putText(frame, f"Diferencia Y: {medio_y - base_y}px",  # Diferencia en píxeles
                    (base_x - 40, base_y - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Mostrar el frame procesado
    cv2.imshow("Frame", frame)

    # Salir al presionar la tecla 'q'

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Liberar recursos al finalizar
cap.release()
cv2.destroyAllWindows()