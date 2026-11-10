# Análisis de tiros de baloncesto con OpenCV 🏀

## Descripción

Proyecto desarrollado para la asignatura de Imagen Digital.

El objetivo es crear una aplicación de visión por computador que permita analizar automáticamente lanzamientos de baloncesto mediante técnicas clásicas de procesamiento de imágenes con OpenCV, sin utilizar Inteligencia Artificial ni Machine Learning.

La aplicación está pensada para analizar series de lanzamientos durante entrenamientos, obteniendo estadísticas sin necesidad de contabilizar los tiros manualmente.

## Funcionalidades

- Detección y seguimiento del balón mediante segmentación por color HSV.
- Selección manual del aro utilizando el ratón.
- Definición de dos regiones de detección alrededor del aro.
- Selección manual de una zona de lanzamiento.
- Detección de tiros cuando el balón sale de la zona seleccionada.
- Detección de posibles canastas mediante el paso descendente del balón por las regiones del aro.
- Cálculo del número de tiros, canastas y porcentaje de acierto.
- Cálculo aproximado del tiempo hasta la zona superior del aro y de la altura máxima respecto al aro, para los lanzamientos encestados.

## Principales funciones de OpenCV utilizadas

| Función | Utilidad en el proyecto |
|---|---|
| `cv2.VideoCapture()` | Captura los fotogramas de la cámara. |
| `cv2.cvtColor()` | Convierte las imágenes del espacio BGR a HSV para facilitar la detección del balón por su color. |
| `cv2.blur()` | Aplica un suavizado para reducir el ruido de la imagen. |
| `cv2.inRange()` | Genera una máscara binaria con los píxeles que pertenecen al rango HSV seleccionado. |
| `cv2.dilate()` | Amplía las regiones detectadas en la máscara, ayudando a unir fragmentos. |
| `cv2.findContours()` | Localiza los contornos de las regiones que cumplen el filtro de color. |
| `cv2.contourArea()` | Calcula el área de los contornos para seleccionar el objeto detectado de mayor tamaño. |
| `cv2.minEnclosingCircle()` | Obtiene el centro y el radio del círculo que rodea el contorno del balón. |
| `cv2.setMouseCallback()` | Permite seleccionar el color del balón, los extremos del aro y la zona de lanzamiento mediante el ratón. |
| `cv2.circle()` | Dibuja el contorno y el centro de la pelota. |
| `cv2.line()` | Dibuja la trayectoria del balón y la posición del aro. |
| `cv2.rectangle()` | Representa las regiones de detección del aro y la zona de lanzamiento. |
| `cv2.addWeighted()` | Permite dibujar regiones semitransparentes sobre el vídeo. |
| `cv2.putText()` | Muestra los contadores, el porcentaje y las mediciones. |

## Funcionamiento

1. Se inicia la captura de vídeo mediante una cámara fija.
2. El usuario selecciona el color de la pelota con el ratón.
3. Se seleccionan manualmente los dos extremos del aro.
4. Se define una zona de lanzamiento.
5. El programa sigue la posición del balón en cada fotograma.
6. Cuando el balón sale de la zona de lanzamiento, se contabiliza un tiro.
7. Si atraviesa las regiones del aro en orden descendente, se contabiliza una posible canasta.
8. Se actualizan las estadísticas mostradas en pantalla.

## Controles

| Control | Acción |
|---|---|
| Clic izquierdo | Seleccionar el color del balón. |
| Clic derecho (2 veces) | Seleccionar los extremos del aro. |
| Tecla `z` | Activar la selección de la zona de lanzamiento. |
| Clic izquierdo (2 veces tras pulsar `z`) | Seleccionar las esquinas de la zona. |
| Tecla `q` | Salir. |

## Limitaciones

- Se necesita una cámara fija para mantener las regiones seleccionadas.
- La detección depende de las condiciones de iluminación y del color del entorno.
- El balón puede confundirse con otros objetos de tonalidades similares.
- La detección de canastas es aproximada y puede producir falsos positivos por la perspectiva.
- La altura máxima se expresa en píxeles respecto al aro; no representa directamente una altura real en centímetros o metros.
- El tiempo medido corresponde al intervalo entre la salida de la zona de lanzamiento y la llegada a la región superior del aro.

## Posibles mejoras

- Mejorar el seguimiento para evitar confusiones con otros objetos.
- Reducir los falsos positivos en la detección de canastas.
- Calibrar la cámara para estimar distancias reales.
- Almacenar las estadísticas de cada tiro y comparar diferentes sesiones de entrenamiento.
- Analizar grandes series de lanzamientos para estudiar la evolución del porcentaje de acierto.