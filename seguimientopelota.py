import cv2
import numpy as np
import time

matiz_objetivo = 0
rango = 5
color = (0, 255, 255)
grosor = 4
lista_puntos = []
tamanio_lista = 50
dibujar = False

# Extremos del aro seleccionados con el boton derecho
puntos_aro = []

# Zona de lanzamiento: se elige pulsando Z y despues con dos clics izquierdos
modo_seleccion_zona = False
puntos_zona_tiro = []
zona_tiro = None
color_zona_tiro = (255, 255, 0)

# Contador de lanzamientos: primero debe verse el balon dentro de la zona
# durante varios fotogramas, y luego salir de ella.
tiros = 0
fotogramas_dentro = 0
balon_preparado = False
intento_activo = False
fotogramas_perdido = 0
min_fotogramas_dentro = 3
max_fotogramas_perdido = 2
mensaje_tiro = 0

# Estado del detector: 0=esperando, 1=paso por arriba, 2=canasta contada
estado_tiro = 0
canastas = 0
posicion_anterior = None
fotogramas_espera = 0
fotogramas_fuera = 0
mensaje_canasta = 0

# Medidas del lanzamiento: se muestran solo cuando se confirma canasta.
# Tiempo = desde que sale de la zona de tiro hasta la primera entrada
# en la zona superior del aro (NO es necesariamente el vuelo completo).
# Altura = diferencia vertical en pixeles respecto al aro (NO metros).
inicio_lanzamiento = None
entrada_zona_superior = None
y_minima_lanzamiento = None
ultimo_tiempo = None
ultima_altura_px = None
ultimo_numero_tiro = None

def reinicia_medicion():
    global inicio_lanzamiento, entrada_zona_superior, y_minima_lanzamiento
    inicio_lanzamiento = None
    entrada_zona_superior = None
    y_minima_lanzamiento = None

def actualiza_altura(centro):
    global y_minima_lanzamiento
    if not intento_activo or centro is None:
        return
    if y_minima_lanzamiento is None or centro[1] < y_minima_lanzamiento:
        y_minima_lanzamiento = centro[1]

# Colores de las regiones del aro
color_aro = (0, 0, 255)
color_zona_superior = (255, 0, 0)
color_zona_inferior = (0, 255, 0)

def selecciona_color(evento, x, y, flags, frame):
    global matiz_objetivo, lista_puntos, dibujar, puntos_aro
    global estado_tiro, posicion_anterior, fotogramas_espera, fotogramas_fuera
    global modo_seleccion_zona, puntos_zona_tiro, zona_tiro
    global fotogramas_dentro, balon_preparado, intento_activo, fotogramas_perdido
    global ultimo_tiempo, ultima_altura_px, ultimo_numero_tiro

    if evento == cv2.EVENT_LBUTTONDOWN and modo_seleccion_zona:
        puntos_zona_tiro.append((x, y))

        if len(puntos_zona_tiro) == 2:
            (x1, y1), (x2, y2) = puntos_zona_tiro
            zona_tiro = (min(x1, x2), min(y1, y2),
                         max(x1, x2), max(y1, y2))
            modo_seleccion_zona = False
            fotogramas_dentro = 0
            balon_preparado = False
            intento_activo = False
            fotogramas_perdido = 0
            reinicia_medicion()
            print('Zona de lanzamiento seleccionada:', zona_tiro)

    elif evento == cv2.EVENT_LBUTTONDOWN:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        matiz_objetivo = h[y, x]
        dibujar = True
        lista_puntos = []
        estado_tiro = 0
        posicion_anterior = None
        fotogramas_espera = 0
        fotogramas_fuera = 0
        fotogramas_dentro = 0
        balon_preparado = False
        intento_activo = False
        fotogramas_perdido = 0
        reinicia_medicion()

    # Dos clics derechos marcan los extremos del aro
    elif evento == cv2.EVENT_RBUTTONDOWN:
        if len(puntos_aro) == 2:
            puntos_aro = []
        puntos_aro.append((x, y))
        estado_tiro = 0
        posicion_anterior = None
        fotogramas_espera = 0
        fotogramas_fuera = 0
        reinicia_medicion()

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

def dibuja_zona_tiro(frame):
    # Zona de lanzamiento ya seleccionada
    if zona_tiro is not None:
        x1, y1, x2, y2 = zona_tiro
        capa = frame.copy()
        cv2.rectangle(capa, (x1, y1), (x2, y2), color_zona_tiro, -1)
        cv2.addWeighted(capa, 0.15, frame, 0.85, 0, frame)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color_zona_tiro, 2)
        cv2.putText(frame, 'Zona de tiro', (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_COMPLEX, 0.6, color_zona_tiro, 2)

    # Mientras se marcan las dos esquinas, mostramos los puntos
    if modo_seleccion_zona:
        cv2.putText(frame, 'Selecciona 2 esquinas con clic izquierdo',
                    (15, 120), cv2.FONT_HERSHEY_COMPLEX, 0.6,
                    color_zona_tiro, 2)
        for punto in puntos_zona_tiro:
            cv2.circle(frame, punto, 5, color_zona_tiro, -1)

    return frame

def dentro_zona(centro, zona):
    x, y = centro
    x1, y1, x2, y2 = zona
    return x1 <= x <= x2 and y1 <= y <= y2

def comprueba_tiro(centro):
    global tiros, fotogramas_dentro, balon_preparado
    global intento_activo, fotogramas_perdido, mensaje_tiro
    global estado_tiro, posicion_anterior, fotogramas_espera, fotogramas_fuera
    global inicio_lanzamiento

    if zona_tiro is None:
        return

    if centro is None:
        # No contamos una salida si simplemente desaparece la deteccion.
        fotogramas_perdido += 1
        if fotogramas_perdido > max_fotogramas_perdido:
            fotogramas_dentro = 0
            balon_preparado = False
        return

    # Si venimos de una perdida larga no asumimos que ha salido de la zona.
    if fotogramas_perdido > max_fotogramas_perdido:
        fotogramas_dentro = 0
        balon_preparado = False
    fotogramas_perdido = 0

    dentro = dentro_zona(centro, zona_tiro)

    if dentro:
        fotogramas_dentro += 1
        if fotogramas_dentro >= min_fotogramas_dentro:
            balon_preparado = True
            # Ha vuelto a la zona: ya puede empezar un nuevo lanzamiento.
            intento_activo = False
            reinicia_medicion()
    else:
        if balon_preparado:
            tiros += 1
            mensaje_tiro = 30
            intento_activo = True
            reinicia_medicion()
            inicio_lanzamiento = time.perf_counter()
            actualiza_altura(centro)
            # Reiniciar el detector del aro para asociar la siguiente
            # posible canasta al lanzamiento que acaba de comenzar.
            estado_tiro = 0
            posicion_anterior = None
            fotogramas_espera = 0
            fotogramas_fuera = 0
            print('TIRO DETECTADO. Total:', tiros)
        balon_preparado = False
        fotogramas_dentro = 0

def comprueba_canasta(centro, regiones):
    global estado_tiro, canastas, posicion_anterior
    global fotogramas_espera, fotogramas_fuera, mensaje_canasta, intento_activo
    global entrada_zona_superior, ultimo_tiempo, ultima_altura_px, ultimo_numero_tiro

    # No se puede sumar una canasta sin un tiro registrado previamente.
    if not intento_activo:
        posicion_anterior = None
        estado_tiro = 0
        return

    if centro is None or regiones is None:
        # No avanzamos la deteccion si se pierde el balon.
        posicion_anterior = None
        if estado_tiro == 1:
            fotogramas_espera += 1
            if fotogramas_espera > 20:
                estado_tiro = 0
        return

    zona_superior, zona_inferior, _, _ = regiones
    arriba = dentro_zona(centro, zona_superior)
    abajo = dentro_zona(centro, zona_inferior)

    # Guardamos el primer contacto con la zona superior tras el tiro.
    if arriba and entrada_zona_superior is None and inicio_lanzamiento is not None:
        entrada_zona_superior = time.perf_counter()

    #  aumenta cuando el balon baja.
    bajando = posicion_anterior is not None and centro[1] > posicion_anterior[1]

    if estado_tiro == 0:
        if arriba and bajando:
            estado_tiro = 1
            fotogramas_espera = 0

    elif estado_tiro == 1:
        fotogramas_espera += 1
        if abajo and bajando:
            # Solo validamos las medidas cuando se confirma el enceste.
            _, _, punto_izq, punto_der = regiones
            y_aro = (punto_izq[1] + punto_der[1]) / 2

            if inicio_lanzamiento is not None and entrada_zona_superior is not None:
                ultimo_tiempo = entrada_zona_superior - inicio_lanzamiento
            else:
                ultimo_tiempo = None

            if y_minima_lanzamiento is not None:
                ultima_altura_px = y_aro - y_minima_lanzamiento
            else:
                ultima_altura_px = None

            ultimo_numero_tiro = tiros
            print('TIRO', ultimo_numero_tiro,
                  '| Tiempo hasta zona superior:', 
                  ('%.2f s' % ultimo_tiempo) if ultimo_tiempo is not None else 'no disponible',
                  '| Altura sobre aro:',
                  ('%.0f px' % ultima_altura_px) if ultima_altura_px is not None else 'no disponible')
            canastas += 1
            intento_activo = False  # Una canasta como maximo por tiro.
            reinicia_medicion()
            mensaje_canasta = 35  # Mostrar el aviso durante 35 fotogramas
            estado_tiro = 2
            fotogramas_fuera = 0
            print('CANASTA! Total:', canastas)
        elif fotogramas_espera > 20:
            # Descartamos un paso por arriba que no llega abajo.
            estado_tiro = 0

    elif estado_tiro == 2:
        # Evita contar mas de una vez mientras el balon sigue en las zonas.
        if not arriba and not abajo:
            fotogramas_fuera += 1
        else:
            fotogramas_fuera = 0
        if fotogramas_fuera >= 5:
            estado_tiro = 0

    posicion_anterior = centro

cv2.namedWindow('webcam', cv2.WINDOW_NORMAL)
cv2.resizeWindow('webcam', 1280, 720)
cv2.createTrackbar('Rango', 'webcam', 5, 30, selecciona_rango)

camara = cv2.VideoCapture(3)
camara.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camara.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

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
    centro_balon = None

    if len(contornos) > 0 and dibujar:
        contorno_max = max(contornos, key=cv2.contourArea)
        (x, y), radio = cv2.minEnclosingCircle(contorno_max)
        x = int(x)
        y = int(y)
        radio = int(radio)
        centro_balon = (x, y)

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

    alto_frame, ancho_frame = frame.shape[:2]
    regiones = obten_regiones_aro(puntos_aro, ancho_frame, alto_frame)
    comprueba_tiro(centro_balon)
    actualiza_altura(centro_balon)
    comprueba_canasta(centro_balon, regiones)

    frame = dibuja_regiones_aro(frame, puntos_aro)
    frame = dibuja_zona_tiro(frame)

    porcentaje = 100 * canastas / tiros if tiros > 0 else 0
    cv2.putText(frame, 'Tiros: ' + str(tiros), (15, 35),
                cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, 'Canastas: ' + str(canastas), (15, 65),
                cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, 'Acierto: ' + format(porcentaje, '.1f') + '%', (15, 95),
                cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 2)

    if mensaje_tiro > 0:
        cv2.putText(frame, 'TIRO!', (15, 135),
                    cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
        mensaje_tiro -= 1

    if mensaje_canasta > 0:
        cv2.putText(frame, 'CANASTA!', (15, 175),
                    cv2.FONT_HERSHEY_COMPLEX, 1.1, (0, 255, 0), 3)
        mensaje_canasta -= 1

    # Mostrar las medidas del ultimo tiro que termino en canasta.
    if ultimo_numero_tiro is not None:
        cv2.putText(frame, 'Ultima canasta - tiro: ' + str(ultimo_numero_tiro),
                    (15, 215), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 2)
        texto_tiempo = ('%.2f s' % ultimo_tiempo) if ultimo_tiempo is not None else 'no disponible'
        texto_altura = ('%.0f px' % ultima_altura_px) if ultima_altura_px is not None else 'no disponible'
        cv2.putText(frame, 'Tiempo a zona superior: ' + texto_tiempo,
                    (15, 245), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, 'Altura max. sobre aro: ' + texto_altura,
                    (15, 275), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow('webcam', frame)
    cv2.setMouseCallback('webcam', selecciona_color, frame)

    tecla = cv2.waitKey(10) & 0xFF
    if tecla == ord('q'):
        break
    elif tecla == ord('z'):
        modo_seleccion_zona = True
        puntos_zona_tiro = []
        zona_tiro = None  # La siguiente seleccion sustituye a la anterior
        fotogramas_dentro = 0
        balon_preparado = False
        intento_activo = False
        fotogramas_perdido = 0
        estado_tiro = 0
        reinicia_medicion()
        print('Marca las dos esquinas de la zona de lanzamiento')

camara.release()
cv2.destroyAllWindows()
