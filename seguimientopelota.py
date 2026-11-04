import cv2
import numpy as np

matiz_objetivo = 0
rango = 5
color = (0, 255, 255)
grosor = 4
lista_puntos = []
tamanio_lista = 50
dibujar = False

# Extremos del aro seleccionados con el boton derecho
puntos_aro = []

# Colores de las regiones del aro
color_aro = (0, 0, 255)
color_zona_superior = (255, 0, 0)
color_zona_inferior = (0, 255, 0)

def selecciona_color(evento, x, y, flags, frame):
    global matiz_objetivo, lista_puntos, dibujar, puntos_aro

    if evento == cv2.EVENT_LBUTTONDOWN:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        matiz_objetivo = h[y, x]
        dibujar = True
        lista_puntos = []

    # Dos clics derechos marcan los extremos del aro
    elif evento == cv2.EVENT_RBUTTONDOWN:
        if len(puntos_aro) == 2:
            puntos_aro = []
        puntos_aro.append((x, y))

def selecciona_rango(valor):
    global rango
    rango = valor

def filtra_matiz(frame):
    frame_suavizado = cv2.blur(frame, (10, 10))
    hsv = cv2.cvtColor(frame_suavizado, cv2.COLOR_BGR2HSV)

    color_inferior = np.array([matiz_objetivo - rango, 150, 0])
    color_superior = np.array([matiz_objetivo + rango, 255, 255])

    mascara = cv2.inRange(hsv, color_inferior, color_superior)
    mascara = cv2.dilate(mascara, None, iterations=4)

    contornos, _ = cv2.findContours(mascara, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    return contornos

def obten_regiones_aro(puntos_aro, ancho_frame, alto_frame):
    if len(puntos_aro) != 2:
        return None

    (x1, y1), (x2, y2) = puntos_aro

    x_izq = min(x1, x2)
    x_der = max(x1, x2)
    y_aro = int((y1 + y2) / 2)

    ancho_aro = x_der - x_izq
    if ancho_aro <= 0:
        return None

    # La altura de cada region depende del ancho del aro
    altura_zona = max(20, int(ancho_aro * 0.6))
    margen_lateral = max(10, int(ancho_aro * 0.15))

    x_inicio = max(0, x_izq - margen_lateral)
    x_fin = min(ancho_frame - 1, x_der + margen_lateral)

    zona_superior = (
        x_inicio,
        max(0, y_aro - altura_zona),
        x_fin,
        max(0, y_aro)
    )

    zona_inferior = (
        x_inicio,
        min(alto_frame - 1, y_aro),
        x_fin,
        min(alto_frame - 1, y_aro + altura_zona)
    )

    return zona_superior, zona_inferior, (x_izq, y_aro), (x_der, y_aro)

def dibuja_regiones_aro(frame, puntos_aro):
    alto_frame, ancho_frame = frame.shape[:2]
    resultado = obten_regiones_aro(puntos_aro, ancho_frame, alto_frame)

    for punto in puntos_aro:
        cv2.circle(frame, punto, 5, color_aro, -1)

    if resultado is None:
        return frame

    zona_superior, zona_inferior, punto_izq, punto_der = resultado

    # Dibujar la linea del aro
    cv2.line(frame, punto_izq, punto_der, color_aro, 3)

    overlay = frame.copy()

    x1s, y1s, x2s, y2s = zona_superior
    x1i, y1i, x2i, y2i = zona_inferior

    cv2.rectangle(overlay, (x1s, y1s), (x2s, y2s), color_zona_superior, -1)
    cv2.rectangle(overlay, (x1i, y1i), (x2i, y2i), color_zona_inferior, -1)

    cv2.addWeighted(overlay, 0.20, frame, 0.80, 0, frame)

    cv2.rectangle(frame, (x1s, y1s), (x2s, y2s), color_zona_superior, 2)
    cv2.rectangle(frame, (x1i, y1i), (x2i, y2i), color_zona_inferior, 2)

    cv2.putText(frame, 'Zona superior', (x1s, max(20, y1s - 8)),
                cv2.FONT_HERSHEY_COMPLEX, 0.6, color_zona_superior, 1)

    cv2.putText(frame, 'Zona inferior', (x1i, min(alto_frame - 10, y2i + 20)),
                cv2.FONT_HERSHEY_COMPLEX, 0.6, color_zona_inferior, 1)

    return frame

cv2.namedWindow('webcam')
cv2.createTrackbar('Rango', 'webcam', 5, 30, selecciona_rango)

camara = cv2.VideoCapture(3)

if not camara.isOpened():
    print('No es posible abrir la camara')
    exit()

while True:
    ret, frame = camara.read()
    if not ret:
        print('No es posible obtener la imagen')
        break

    # Volteamos antes de detectar y seleccionar para que coincidan las coordenadas
    frame = cv2.flip(frame, 1)

    contornos = filtra_matiz(frame)

    if len(contornos) > 0 and dibujar:
        contorno_max = max(contornos, key=cv2.contourArea)
        (x, y), radio = cv2.minEnclosingCircle(contorno_max)
        x = int(x)
        y = int(y)
        radio = int(radio)

        if len(lista_puntos) >= tamanio_lista:
            lista_puntos.pop(0)

        lista_puntos.append((x, y))

        cv2.circle(frame, (x, y), radio, color, 2)
        cv2.circle(frame, (x, y), 3, color, -1)

        if len(lista_puntos) >= 2:
            punto_prev = lista_puntos[0]
            for punto in lista_puntos:
                cv2.line(frame, punto, punto_prev, color, grosor)
                punto_prev = punto

    frame = dibuja_regiones_aro(frame, puntos_aro)

    cv2.imshow('webcam', frame)
    cv2.setMouseCallback('webcam', selecciona_color, frame)

    if cv2.waitKey(10) == ord('q'):
        break

camara.release()
cv2.destroyAllWindows()
