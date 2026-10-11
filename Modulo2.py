#Modulo 2: Segmentacion y morfologia matematica de la pupila
#Modulo 2.1: Algoritmos de segmentacion
import cv2
import numpy as np
import matplotlib.pyplot as plt

VIDEO_PATH = r"D:\video_ojo_oli.mp4"   # cambiá por tu video
FRAME_IDX = 0                    # frame que querés usar

# 1) Leer un frame del video
TIEMPO_S = 11.0   # segundo que querés usar

cap = cv2.VideoCapture(VIDEO_PATH)
cap.set(cv2.CAP_PROP_POS_MSEC, TIEMPO_S * 1000)   # va en milisegundos
ok, frame_bgr = cap.read()
cap.release()
assert ok, "No se pudo leer el frame, revisá VIDEO_PATH o TIEMPO_S"

# 2) Elegir la ROI con el mouse (arrastrá un rectángulo alrededor del ojo y Enter)
x, y, w, h = cv2.selectROI("Elegí la ROI", frame_bgr)
cv2.destroyAllWindows()
roi_bgr = frame_bgr[y:y+h, x:x+w]

# 3) Pasar a gris y suavizar
gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# 4) Otsu (INV: la pupila es oscura, así queda en blanco = 255)
umbral, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
umbral = 40
print(f"Umbral de Otsu: {umbral}")

# 5) Visualizar
fig, ax = plt.subplots(1, 4, figsize=(16, 4))
ax[0].imshow(cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)); ax[0].set_title("ROI")
ax[1].imshow(blur, cmap="gray"); ax[1].set_title("Gris + blur")
ax[2].hist(blur.ravel(), bins=256); ax[2].axvline(umbral, color="r"); ax[2].set_title("Histograma + umbral")
ax[3].imshow(mask, cmap="gray"); ax[3].set_title("Máscara Otsu")
for a in (ax[0], ax[1], ax[3]): a.axis("off")
plt.show()

#K means

cv2.setRNGSeed(0)   # para que los resultados sean reproducibles

# 1) Pasar a Lab y suavizar un poco
lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB)
lab = cv2.GaussianBlur(lab, (5, 5), 0)

# 2) Armar la matriz de datos: una fila por píxel, 3 columnas (L, a, b)
datos = lab.reshape(-1, 3).astype(np.float32)

# 3) Correr K-Means para cada K
criterio = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.5)
fig, ax = plt.subplots(2, 4, figsize=(16, 7))

for i, K in enumerate([2, 3, 4, 5]):
    _, etiquetas, centros = cv2.kmeans(datos, K, None, criterio, 10, cv2.KMEANS_PP_CENTERS)
    etiquetas = etiquetas.reshape(roi_bgr.shape[:2])

    # El cluster de la pupila es el de menor L (el más oscuro)
    k_pupila = np.argmin(centros[:, 0])
    mask_k = (etiquetas == k_pupila).astype(np.uint8) * 255

    ax[0, i].imshow(etiquetas, cmap="viridis"); ax[0, i].set_title(f"Clusters K={K}")
    ax[1, i].imshow(mask_k, cmap="gray");       ax[1, i].set_title(f"Máscara pupila K={K}")
    ax[0, i].axis("off"); ax[1, i].axis("off")

plt.tight_layout()
plt.show()

#Snakes

# --- Elegir el blob de la pupila a partir de K-Means (K=4) ---
K = 4
_, etiquetas, centros = cv2.kmeans(datos, K, None, criterio, 10, cv2.KMEANS_PP_CENTERS)
etiquetas = etiquetas.reshape(roi_bgr.shape[:2])
k_pupila = np.argmin(centros[:, 0])
mask_k = (etiquetas == k_pupila).astype(np.uint8) * 255

n, lab_cc, stats, cents = cv2.connectedComponentsWithStats(mask_k)
h, w = mask_k.shape
centro_roi = np.array([w / 2, h / 2])

mejor, mejor_score = None, -1
for i in range(1, n):
    area = stats[i, cv2.CC_STAT_AREA]
    dist = np.linalg.norm(cents[i] - centro_roi)
    score = area / (1 + dist)
    if score > mejor_score:
        mejor, mejor_score = i, score

mask_pupila = (lab_cc == mejor).astype(np.uint8) * 255

from skimage.segmentation import active_contour
from skimage.filters import gaussian

# 1) Sacar el reflejo del flash (píxeles muy brillantes) para que no engañe a la snake
gris = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
flash = (gris > 200).astype(np.uint8) * 255
flash = cv2.dilate(flash, np.ones((5, 5), np.uint8))
gris_sin_flash = cv2.inpaint(gris, flash, 3, cv2.INPAINT_TELEA)

# 2) Suavizar y normalizar a [0, 1]
img = gaussian(gris_sin_flash, sigma=2, preserve_range=True) / 255.0

# 3) Contorno inicial: un círculo alrededor de la pupila
M = cv2.moments(mask_pupila)
cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
radio_pupila = np.sqrt(mask_pupila.astype(bool).sum() / np.pi)
radio_ini = radio_pupila * 1.5      # un poco más grande que la pupila

t = np.linspace(0, 2 * np.pi, 200)
init = np.column_stack([cy + radio_ini * np.sin(t),    # fila (y)
                        cx + radio_ini * np.cos(t)])   # columna (x)

# 4) Correr la snake
snake = active_contour(img, init,
                       alpha=0.015,    # elasticidad: más alto = se contrae más
                       beta=10,        # rigidez: más alto = más suave/redonda
                       gamma=0.001,    # tamaño del paso
                       w_line=0,
                       w_edge=1,       # atracción a bordes
                       max_num_iter=2500)

# 5) Máscara binaria a partir de la snake (para usarla después en Dice/IoU)
pts = np.round(snake[:, ::-1]).astype(np.int32)   # (fila, col) -> (x, y) para OpenCV
mask_snake = np.zeros_like(mask_pupila)
cv2.fillPoly(mask_snake, [pts], 255)

# 6) Visualizar
fig, ax = plt.subplots(1, 3, figsize=(14, 5))
ax[0].imshow(cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB))
ax[0].plot(init[:, 1], init[:, 0], "--r", lw=1.5)
ax[0].set_title("Contorno inicial (rojo)")
ax[1].imshow(cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB))
ax[1].plot(snake[:, 1], snake[:, 0], "-g", lw=2)
ax[1].set_title("Contorno final (verde)")
ax[2].imshow(mask_snake, cmap="gray"); ax[2].set_title("Máscara de la snake")
for a in ax: a.axis("off")
plt.show()

#Watershed

from skimage.segmentation import watershed
from skimage.filters import sobel

# 1) Sacar el flash (igual que en la snake; lo repetimos para no depender de otro bloque)
gris = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
flash = (gris > 200).astype(np.uint8) * 255
flash = cv2.dilate(flash, np.ones((5, 5), np.uint8))
gris_sin_flash = cv2.inpaint(gris, flash, 3, cv2.INPAINT_TELEA)

# 2) Imagen de gradiente (el "relieve")
suave = cv2.GaussianBlur(gris_sin_flash, (5, 5), 0)
gradiente = sobel(suave.astype(float) / 255.0)

# 3) Marcadores, con tamaños relativos al tamaño de la pupila
radio = np.sqrt((mask_pupila > 0).sum() / np.pi)
r_chico = max(1, int(0.3 * radio))
r_grande = max(2, int(0.6 * radio))
k_chico = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r_chico + 1, 2 * r_chico + 1))
k_grande = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r_grande + 1, 2 * r_grande + 1))

seguro_pupila = cv2.erode(mask_pupila, k_chico)
zona_dudosa = cv2.dilate(mask_pupila, k_grande)

marcadores = np.zeros(gris.shape, dtype=np.int32)
marcadores[zona_dudosa == 0] = 1       # 1 = fondo seguro
marcadores[seguro_pupila > 0] = 2      # 2 = pupila segura

# 4) Inundación
etiquetas_ws = watershed(gradiente, marcadores)
mask_ws = (etiquetas_ws == 2).astype(np.uint8) * 255

# 5) Visualizar
fig, ax = plt.subplots(1, 4, figsize=(16, 4))
ax[0].imshow(gradiente, cmap="gray");  ax[0].set_title("Gradiente (relieve)")
ax[1].imshow(marcadores, cmap="viridis"); ax[1].set_title("Marcadores")
ax[2].imshow(mask_ws, cmap="gray");    ax[2].set_title("Máscara Watershed")
ax[3].imshow(cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB))
ax[3].contour(mask_ws, colors="lime", linewidths=1)
ax[3].set_title("Contorno sobre la ROI")
for a in ax: a.axis("off")
plt.show()

#Modulo 2.2: Morfologia matematica para limpieza de mascara
