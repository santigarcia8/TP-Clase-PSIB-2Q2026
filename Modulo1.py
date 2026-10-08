import cv2
import numpy as np
import matplotlib.pyplot as plt


#Modulo 1: Fundamentos, manejo de espacios de color y mejoramiento digital
#Modulo 1.1: Carga, muestreo y espacios de color
# ==========================================
# 1. Carga de video, extracción de frame y ROI
# ==========================================
video_path = 'video_ojo.mp4'  # Cambiar por la ruta de tu video
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise FileNotFoundError(f"No se pudo abrir el archivo de video: {video_path}")

# Leer el primer frame representativo (o avanza al frame deseado)
ret, frame = cap.read()
cap.release()

if not ret:
    raise ValueError("No se pudo extraer ningún frame del video.")

# Definición de la Región de Interés (ROI)[cite: 2]
# Puedes definir coordenadas fijas [y1:y2, x1:x2] o usar cv2.selectROI
# Coordenadas de ejemplo (ajustar según tu video):
y1, y2 = 100, 400
x1, x2 = 150, 450
roi_bgr = frame[y1:y2, x1:x2]
roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
roi_gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

# ==========================================
# 2. Dimensiones y Profundidad[cite: 2]
# ==========================================
alto, ancho = roi_bgr.shape[:2]
canales = roi_bgr.shape[2] if len(roi_bgr.shape) == 3 else 1
profundidad_bits = roi_bgr.dtype.itemsize * 8  # Bits por canal (ej. uint8 -> 8 bits)

print("--- DIMENSIONES Y PROFUNDIDAD DE LA ROI ---")
print(f"Dimensiones (Ancho x Alto): {ancho} x {alto} píxeles")
print(f"Cantidad de canales: {canales}")
print(f"Profundidad de bits por canal: {profundidad_bits} bits ({2**profundidad_bits} niveles por canal)\n")

# ==========================================
# 3. Resolución Espacial (Submuestreo)[cite: 2]
# ==========================================
escalas = [1.0, 0.50, 0.25, 0.05]
submuestreadas = []

for escala in escalas:
    ancho_s = int(ancho * escala)
    alto_s = int(alto * escala)
    # Submuestreo mediante interpolación por vecino más cercano
    sub_img = cv2.resize(roi_rgb, (ancho_s, alto_s), interpolation=cv2.INTER_NEAREST)
    submuestreadas.append((escala, sub_img))

# Visualización de Submuestreo
plt.figure(figsize=(12, 6))
for i, (escala, img) in enumerate(submuestreadas):
    plt.subplot(1, 4, i + 1)
    plt.imshow(img)
    plt.title(f"Escala {int(escala*100)}%\n({img.shape[1]}x{img.shape[0]})")
    plt.axis('off')
plt.suptitle('Resolución Espacial - Submuestreo de la ROI[cite: 2]', fontsize=14)
plt.tight_layout()
plt.show()

# ==========================================
# 4. Cuantización (Profundidad de bits)[cite: 2]
# ==========================================
def reducir_bits(imagen_gray, bits_objetivo):
    niveles = 2 ** bits_objetivo
    factor_paso = 256 // niveles
    # Reducción de escala cuantizada y reescalado a rango [0, 255] para visualización
    cuantizada = (imagen_gray // factor_paso) * (255 // (niveles - 1))
    return cuantizada.astype(np.uint8)

gray_8bit = roi_gray
gray_4bit = reducir_bits(roi_gray, 4)
gray_2bit = reducir_bits(roi_gray, 2)

# Visualización de Cuantización
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(gray_8bit, cmap='gray')
plt.title('Original (8 bits - 256 niveles)')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(gray_4bit, cmap='gray')
plt.title('Cuantizada a 4 bits (16 niveles)')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(gray_2bit, cmap='gray')
plt.title('Cuantizada a 2 bits (4 niveles)')
plt.axis('off')

plt.suptitle('Cuantización (Profundidad de Bit)[cite: 2]', fontsize=14)
plt.tight_layout()
plt.show()

# ==========================================
# 5. Espacios de Color (RGB y HSV)[cite: 2]
# ==========================================
# 5a. Canales RGB por separado en escala de grises[cite: 2]
R_channel = roi_rgb[:, :, 0]
G_channel = roi_rgb[:, :, 1]
B_channel = roi_rgb[:, :, 2]

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(R_channel, cmap='gray')
plt.title('Canal Rojo (R)')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(G_channel, cmap='gray')
plt.title('Canal Verde (G)')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(B_channel, cmap='gray')
plt.title('Canal Azul (B)')
plt.axis('off')

plt.suptitle('Canales del Espacio RGB en Escala de Grises[cite: 2]', fontsize=14)
plt.tight_layout()
plt.show()

# 5b. Conversión a HSV y separación de componentes[cite: 2]
roi_hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
H_channel, S_channel, V_channel = cv2.split(roi_hsv)

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(H_channel, cmap='hsv')
plt.title('Matiz (Hue - H)')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(S_channel, cmap='gray')
plt.title('Saturación (Saturation - S)')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(V_channel, cmap='gray')
plt.title('Valor (Value - V)')
plt.axis('off')

plt.suptitle('Componentes del Espacio HSV[cite: 2]', fontsize=14)
plt.tight_layout()
plt.show()


#Modulo 1.2: Calidad, transformaciones de intensidad y filtrado espacial

#Modulo 1.3: Deteccion de bordes y dominio frecuencial

