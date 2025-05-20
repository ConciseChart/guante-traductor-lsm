import serial
import time

# Configura tu puerto y velocidad (ajústalo si es necesario)
puerto = serial.Serial('COM3', 9600)
time.sleep(2)

# Función para normalizar los valores (ajusta si es necesario)
def normalizar(valores):
    return [(v - 200) / (900 - 200) for v in valores]  # suponiendo rango 200-900

# Función para verificar si los valores están en un rango
def en_rango(valores, min_vals, max_vals):
    return all(min_v <= v <= max_v for v, min_v, max_v in zip(valores, min_vals, max_vals))

# Rangos para la letra A (dedos flexionados)
min_a = [0.05, 0.05, 0.05, 0.05, 0.02]
max_a = [0.18, 0.10, 0.09, 0.08, 0.07]

# Rangos para la letra B (pulgar flexionado, demás extendidos)
min_b = [0.13, 0.13, 0.14, 0.13, 0.11]
max_b = [0.18, 0.17, 0.18, 0.15, 0.14]

# Función principal de detección
def detectar_letra(flexiones):
    if en_rango(flexiones, min_a, max_a):
        print("Letra detectada: A")
        return 'A'
    elif en_rango(flexiones, min_b, max_b):
        print("Letra detectada: B")
        return 'B'
    else:
        print("Letra detectada: ?")
        return '?'

# Lectura continua desde el puerto serial
try:
    while True:
        linea = puerto.readline().decode('utf-8').strip()
        if linea:
            try:
                # Suponiendo que Arduino envía: "512,523,487,489,500"
                datos = list(map(int, linea.split(',')))
                flexiones = normalizar(datos)
                print("Flexiones:", flexiones)
                detectar_letra(flexiones)
            except ValueError:
                print("Error al procesar datos:", linea)
except KeyboardInterrupt:
    print("Lectura detenida por el usuario.")
    puerto.close()
