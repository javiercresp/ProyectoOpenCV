import cv2
import numpy as np
import math
import time

# Seguimiento del balon (adaptado del ejercicio de clase)
matiz_objetivo = 0
rango = 5
color = (0, 255, 255)
grosor = 4
lista_puntos = []
tamanio_lista = 50
dibujar = False

# Selecciones manuales
puntos_aro = []
modo_seleccion_zona = False
puntos_zona_tiro = []
zona_tiro = None

# Regiones visibles (colores BGR)
color_aro = (0, 0, 255)
color_zona_superior = (255, 0, 0)
color_zona_inferior = (0, 255, 0)
color_zona_tiro = (255, 255, 0)

# Contadores y estados
canastas = 0
tiros = 0
estado_aro = 0  # 0: espera arriba, 1: espera abajo, 2: ya contada
fotogramas_espera = 0
fotogramas_fuera = 0
posicion_anterior = None
fotogramas_dentro = 0
balon_preparado = False
intento_activo = False
fotogramas_perdido = 0
min_fotogramas_dentro = 3
max_fotogramas_perdido = 2
mensaje_tiro = 0
mensaje_canasta = 0

# Mediciones del tiro actual
inicio_lanzamiento = None
entrada_zona_superior = None
y_minima_lanzamiento = None
puntos_salida = []
NUM_PUNTOS_ANGULO = 6  # Primeras seis posiciones tras salir de la zona

# Resultados del ultimo tiro encestado
ultimo_numero_tiro = None
ultimo_tiempo = None
ultima_altura_px = None
ultimo_angulo = None


def reinicia_medicion():
    global inicio_lanzamiento, entrada_zona_superior, y_minima_lanzamiento, puntos_salida
    inicio_lanzamiento = None
    entrada_zona_superior = None
    y_minima_lanzamiento = None
    puntos_salida = []


def reinicia_estado_tiro():
    global estado_aro, posicion_anterior, fotogramas_espera, fotogramas_fuera
    estado_aro = 0
    posicion_anterior = None
    fotogramas_espera = 0
    fotogramas_fuera = 0


def selecciona_color(evento, x, y, flags, frame):
    global matiz_objetivo, lista_puntos, dibujar, puntos_aro
    global modo_seleccion_zona, puntos_zona_tiro, zona_tiro
    global fotogramas_dentro, balon_preparado, intento_activo, fotogramas_perdido

    if evento == cv2.EVENT_LBUTTONDOWN and modo_seleccion_zona:
        puntos_zona_tiro.append((x, y))
        if len(puntos_zona_tiro) == 2:
            (x1, y1), (x2, y2) = puntos_zona_tiro
            zona_tiro = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
            modo_seleccion_zona = False
            fotogramas_dentro = 0
            balon_preparado = False
            intento_activo = False
            fotogramas_perdido = 0
            reinicia_medicion()
            reinicia_estado_tiro()
            print('Zona de tiro seleccionada:', zona_tiro)

    elif evento == cv2.EVENT_LBUTTONDOWN:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        matiz_objetivo = int(h[y, x])
        dibujar = True
        lista_puntos = []
        fotogramas_dentro = 0
        balon_preparado = False
        intento_activo = False
        fotogramas_perdido = 0
        reinicia_medicion()
        reinicia_estado_tiro()

    elif evento == cv2.EVENT_RBUTTONDOWN:
        if len(puntos_aro) == 2:
            puntos_aro = []
        puntos_aro.append((x, y))
        intento_activo = False
        balon_preparado = False
        fotogramas_dentro = 0
        reinicia_medicion()
        reinicia_estado_tiro()


def selecciona_rango(valor):
    global rango
    rango = valor


def filtra_matiz(frame):
    frame_suavizado = cv2.blur(frame, (10, 10))
    hsv = cv2.cvtColor(frame_suavizado, cv2.COLOR_BGR2HSV)
    # El canal H de OpenCV esta entre 0 y 179.
    h_min = max(0, matiz_objetivo - rango)
    h_max = min(179, matiz_objetivo + rango)
    mascara = cv2.inRange(hsv, np.array([h_min, 150, 0]),
                          np.array([h_max, 255, 255]))
    mascara = cv2.dilate(mascara, None, iterations=4)
    contornos, _ = cv2.findContours(mascara, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    return contornos


def obten_regiones_aro(puntos, ancho_frame, alto_frame):
    if len(puntos) != 2:
        return None
    (x1, y1), (x2, y2) = puntos
    izquierda, derecha = min(x1, x2), max(x1, x2)
    y_aro = int((y1 + y2) / 2)
    ancho_aro = derecha - izquierda
    if ancho_aro <= 0:
        return None
    altura = max(20, int(ancho_aro * 0.6))
    margen = max(10, int(ancho_aro * 0.15))
    xi, xd = max(0, izquierda - margen), min(ancho_frame - 1, derecha + margen)
    zona_superior = (xi, max(0, y_aro - altura), xd, y_aro)
    zona_inferior = (xi, y_aro, xd, min(alto_frame - 1, y_aro + altura))
    return zona_superior, zona_inferior, (izquierda, y_aro), (derecha, y_aro)


def dentro_zona(centro, zona):
    x, y = centro
    x1, y1, x2, y2 = zona
    return x1 <= x <= x2 and y1 <= y <= y2


def dibuja_regiones_aro(frame, regiones):
    for punto in puntos_aro:
        cv2.circle(frame, punto, 5, color_aro, -1)
    if regiones is None:
        return
    superior, inferior, punto_izq, punto_der = regiones
    copia = frame.copy()
    x1, y1, x2, y2 = superior
    cv2.rectangle(copia, (x1, y1), (x2, y2), color_zona_superior, -1)
    x1, y1, x2, y2 = inferior
    cv2.rectangle(copia, (x1, y1), (x2, y2), color_zona_inferior, -1)
    cv2.addWeighted(copia, 0.20, frame, 0.80, 0, frame)
    for zona, color_region in [(superior, color_zona_superior),
                               (inferior, color_zona_inferior)]:
        x1, y1, x2, y2 = zona
        cv2.rectangle(frame, (x1, y1), (x2, y2), color_region, 2)
    cv2.line(frame, punto_izq, punto_der, color_aro, 3)


def dibuja_zona_tiro(frame):
    if zona_tiro is not None:
        x1, y1, x2, y2 = zona_tiro
        copia = frame.copy()
        cv2.rectangle(copia, (x1, y1), (x2, y2), color_zona_tiro, -1)
        cv2.addWeighted(copia, 0.15, frame, 0.85, 0, frame)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color_zona_tiro, 2)
        cv2.putText(frame, 'Zona de tiro', (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_COMPLEX, 0.6, color_zona_tiro, 2)
    if modo_seleccion_zona:
        cv2.putText(frame, 'Marca 2 esquinas con clic izquierdo', (15, 120),
                    cv2.FONT_HERSHEY_COMPLEX, 0.6, color_zona_tiro, 2)
        for punto in puntos_zona_tiro:
            cv2.circle(frame, punto, 5, color_zona_tiro, -1)


def calcula_angulo(puntos):
    # Con varios fotogramas evitamos calcularlo sobre un unico salto de 1 frame.
    if len(puntos) < NUM_PUNTOS_ANGULO:
        return None
    (x1, y1) = puntos[0]
    (x2, y2) = puntos[NUM_PUNTOS_ANGULO - 1]
    dx = x2 - x1
    dy = y1 - y2  # En la imagen Y crece hacia abajo.
    if abs(dx) < 2 or dy <= 0:
        return None  # Sin avance horizontal claro o sin ascenso.
    return math.degrees(math.atan2(dy, abs(dx)))


def comprueba_tiro(centro):
    global tiros, fotogramas_dentro, balon_preparado, intento_activo
    global fotogramas_perdido, mensaje_tiro, inicio_lanzamiento
    if zona_tiro is None:
        return
    if centro is None:
        fotogramas_perdido += 1
        if fotogramas_perdido > max_fotogramas_perdido:
            balon_preparado = False
            fotogramas_dentro = 0
        return
    if fotogramas_perdido > max_fotogramas_perdido:
        balon_preparado = False
        fotogramas_dentro = 0
    fotogramas_perdido = 0

    if dentro_zona(centro, zona_tiro):
        fotogramas_dentro += 1
        if fotogramas_dentro >= min_fotogramas_dentro:
            balon_preparado = True
            # Se prepara un nuevo tiro, desechando el intento anterior.
            if intento_activo:
                intento_activo = False
                reinicia_medicion()
                reinicia_estado_tiro()
    else:
        if balon_preparado:
            tiros += 1
            mensaje_tiro = 30
            intento_activo = True
            reinicia_medicion()
            reinicia_estado_tiro()
            inicio_lanzamiento = time.perf_counter()
            print('TIRO DETECTADO. Total:', tiros)
        balon_preparado = False
        fotogramas_dentro = 0


def actualiza_mediciones(centro):
    global y_minima_lanzamiento, puntos_salida
    if centro is None or not intento_activo:
        return
    if y_minima_lanzamiento is None or centro[1] < y_minima_lanzamiento:
        y_minima_lanzamiento = centro[1]
    if len(puntos_salida) < NUM_PUNTOS_ANGULO:
        puntos_salida.append(centro)


def comprueba_canasta(centro, regiones):
    global estado_aro, posicion_anterior, fotogramas_espera, fotogramas_fuera
    global canastas, mensaje_canasta, intento_activo, entrada_zona_superior
    global ultimo_numero_tiro, ultimo_tiempo, ultima_altura_px, ultimo_angulo

    if not intento_activo or regiones is None:
        reinicia_estado_tiro()
        return
    if centro is None:
        posicion_anterior = None
        if estado_aro == 1:
            fotogramas_espera += 1
            if fotogramas_espera > 20:
                reinicia_estado_tiro()
        return

    superior, inferior, punto_izq, punto_der = regiones
    arriba = dentro_zona(centro, superior)
    abajo = dentro_zona(centro, inferior)
    bajando = posicion_anterior is not None and centro[1] > posicion_anterior[1]

    if arriba and entrada_zona_superior is None and inicio_lanzamiento is not None:
        entrada_zona_superior = time.perf_counter()

    if estado_aro == 0:
        if arriba and bajando:
            estado_aro = 1
            fotogramas_espera = 0
    elif estado_aro == 1:
        fotogramas_espera += 1
        if abajo and bajando:
            # Medimos exclusivamente los lanzamientos que terminan en canasta.
            y_aro = (punto_izq[1] + punto_der[1]) / 2
            ultimo_tiempo = (entrada_zona_superior - inicio_lanzamiento
                             if inicio_lanzamiento is not None and entrada_zona_superior is not None
                             else None)
            ultima_altura_px = (y_aro - y_minima_lanzamiento
                                if y_minima_lanzamiento is not None else None)
            ultimo_angulo = calcula_angulo(puntos_salida)
            ultimo_numero_tiro = tiros
            canastas += 1
            mensaje_canasta = 35
            estado_aro = 2
            intento_activo = False
            print('CANASTA! Total:', canastas,
                  '| Tiempo:', ('%.2f s' % ultimo_tiempo) if ultimo_tiempo is not None else 'N/D',
                  '| Altura:', ('%.0f px' % ultima_altura_px) if ultima_altura_px is not None else 'N/D',
                  '| Angulo:', ('%.1f grados' % ultimo_angulo) if ultimo_angulo is not None else 'N/D')
            reinicia_medicion()
        elif fotogramas_espera > 20:
            reinicia_estado_tiro()
    elif estado_aro == 2:
        if not arriba and not abajo:
            fotogramas_fuera += 1
        else:
            fotogramas_fuera = 0
        if fotogramas_fuera >= 5:
            reinicia_estado_tiro()

    posicion_anterior = centro


cv2.namedWindow('webcam', cv2.WINDOW_NORMAL)
cv2.resizeWindow('webcam', 1280, 720)
cv2.createTrackbar('Rango', 'webcam', rango, 30, selecciona_rango)

camara = cv2.VideoCapture(3)
camara.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camara.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not camara.isOpened():
    print('No es posible abrir la camara')
    raise SystemExit

while True:
    ret, frame = camara.read()
    if not ret:
        print('No es posible obtener la imagen')
        break

    # Voltear antes de detectar para que coincida con los clics del usuario.
    frame = cv2.flip(frame, 1)
    contornos = filtra_matiz(frame)
    centro_balon = None
    if contornos and dibujar:
        contorno_max = max(contornos, key=cv2.contourArea)
        (x, y), radio = cv2.minEnclosingCircle(contorno_max)
        centro_balon = (int(x), int(y))
        if len(lista_puntos) >= tamanio_lista:
            lista_puntos.pop(0)
        lista_puntos.append(centro_balon)
        cv2.circle(frame, centro_balon, int(radio), color, 2)
        cv2.circle(frame, centro_balon, 3, color, -1)
        if len(lista_puntos) >= 2:
            for anterior, actual in zip(lista_puntos, lista_puntos[1:]):
                cv2.line(frame, anterior, actual, color, grosor)

    alto_frame, ancho_frame = frame.shape[:2]
    regiones = obten_regiones_aro(puntos_aro, ancho_frame, alto_frame)
    comprueba_tiro(centro_balon)
    actualiza_mediciones(centro_balon)
    comprueba_canasta(centro_balon, regiones)

    dibuja_regiones_aro(frame, regiones)
    dibuja_zona_tiro(frame)

    porcentaje = 100 * canastas / tiros if tiros > 0 else 0
    cv2.putText(frame, 'Tiros: ' + str(tiros), (15, 35),
                cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, 'Canastas: ' + str(canastas), (15, 65),
                cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, 'Acierto: %.1f%%' % porcentaje, (15, 95),
                cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 2)

    if mensaje_tiro > 0:
        cv2.putText(frame, 'TIRO!', (15, 135), cv2.FONT_HERSHEY_COMPLEX,
                    0.8, (0, 255, 255), 2)
        mensaje_tiro -= 1
    if mensaje_canasta > 0:
        cv2.putText(frame, 'CANASTA!', (15, 175), cv2.FONT_HERSHEY_COMPLEX,
                    1.1, (0, 255, 0), 3)
        mensaje_canasta -= 1

    if ultimo_numero_tiro is not None:
        cv2.putText(frame, 'Ultima canasta - tiro: ' + str(ultimo_numero_tiro),
                    (15, 215), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 2)
        tiempo_txt = ('%.2f s' % ultimo_tiempo) if ultimo_tiempo is not None else 'N/D'
        altura_txt = ('%.0f px' % ultima_altura_px) if ultima_altura_px is not None else 'N/D'
        angulo_txt = ('%.1f grados' % ultimo_angulo) if ultimo_angulo is not None else 'N/D'
        cv2.putText(frame, 'Tiempo hasta zona superior: ' + tiempo_txt,
                    (15, 245), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, 'Altura sobre aro: ' + altura_txt,
                    (15, 275), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, 'Angulo salida aprox.: ' + angulo_txt,
                    (15, 305), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow('webcam', frame)
    cv2.setMouseCallback('webcam', selecciona_color, frame)

    tecla = cv2.waitKey(10) & 0xFF
    if tecla == ord('q'):
        break
    elif tecla == ord('z'):
        modo_seleccion_zona = True
        puntos_zona_tiro = []
        zona_tiro = None
        fotogramas_dentro = 0
        balon_preparado = False
        intento_activo = False
        fotogramas_perdido = 0
        reinicia_medicion()
        reinicia_estado_tiro()
        print('Marca las dos esquinas de la zona de tiro')

camara.release()
cv2.destroyAllWindows()
