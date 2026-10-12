import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr_sk
from skimage.metrics import structural_similarity as ssim_sk

#Definicion de las diferentes funciones que seran utilizadas a lo largo del modulo

def cargar_frame_y_roi(video_path, num_frame=0, roi_coords=None):
    cap = cv2.VideoCapture(video_path) #Abre el archivo desde la ruta establecida en main
    cap.set(cv2.CAP_PROP_POS_FRAMES, num_frame) #Usa el frame deseado, evitando leer todos los frames previos
    ret, frame_bgr = cap.read() #Devuelve true si leyo bien y la matriz de la imagen
    cap.release() #Cierra el archivo de video para liberar la memoria del sistema
    
    if not ret or frame_bgr is None:
        raise ValueError("No se pudo leer el frame del video.")
        
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB) #Invierte los canales a RGB para que Matplotlib visualice los colores reales
    
    if roi_coords is not None:
        ymin, ymax, xmin, xmax = roi_coords
        roi_rgb = frame_rgb[ymin:ymax, xmin:xmax]
    else:
        roi_rgb = frame_rgb
        
    return frame_rgb, roi_rgb

def analizar_resolucion_espacial(roi_rgb):
   #Evalua el efecto de la resolucion espacial y el submuestreo sobre la imagen, para visualizar la perdida de detalle y el efecto de pixelado
    h_orig, w_orig = roi_rgb.shape[:2] #Obtiene las dimensiones originales 

    #Submuestreo (Reducción espacial)
    #Se va reduciendo la cantidad de pixeles de la imagen usando los factores fx y fy
    #se usa la funncion interpolation porque promedia los pixeles del area subyacente, evitando artefactos de alineacion
    roi_50 = cv2.resize(roi_rgb, (0, 0), fx=0.50, fy=0.50, interpolation=cv2.INTER_AREA)
    roi_25 = cv2.resize(roi_rgb, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    roi_05 = cv2.resize(roi_rgb, (0, 0), fx=0.05, fy=0.05, interpolation=cv2.INTER_AREA)

    #Re-escalado al tamaño original para comparación visual directa
    #interpolation inter_nearest usa el metodo del vecino mas cercano, fuerza la aparicion del efecto de pixelado
    vis_50 = cv2.resize(roi_50, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
    vis_25 = cv2.resize(roi_25, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
    vis_05 = cv2.resize(roi_05, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)

    return (roi_50, roi_25, roi_05), (vis_50, vis_25, vis_05) #devuelve las imagenes reducidas con sus dimensiones reales pequeñas y las amplificadas al tamaño original

def reducir_cuantizacion(imagen, bits):
    #Simula la reduccion de la resolucion de cuantizacion o profundidad de bits por pixel
    niveles = 2 ** bits #Calculo de niveles disponibles
    paso = 256 // niveles #calculo el ancho del intervalo para determinar el tamaño de cada nivel de cuantizacion
    return ((imagen // paso) * paso).astype(np.uint8)


def analizar_espacios_color(roi_rgb):
    #Separa la imagen de la ROI en sus canales primarios RGB y realiza la conversion a HSV
    
    #Descompongo la matriz RGB en tres matrices independientes en escalas de grises correspondientes a las intensidades de cada uno
    r, g, b = cv2.split(roi_rgb)
    
    #Conversión a HSV y separación de componentes
    roi_hsv = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(roi_hsv)
    
    return (r, g, b), (h, s, v), roi_hsv
#Analizar estos canales permite determinar cual ofrece mayor contraste entre la pupila y el iris


def modificar_brillo_contraste(imagen_gris, alpha=1.5, beta=50):
   #Hace una transformacion lineal global de intensidad sobre la imagen en escala de grises para ajustar el brillo y el contraste proporcionalmente
   #Alfa es el multiplicador de la pendiente (contraste) y beta es una constante de desplazamiento (brillo)
    return cv2.convertScaleAbs(imagen_gris, alpha=alpha, beta=beta) #ejecuta la funcion lineal a cada pixel



def escalado_rango_interes(imagen_gris, f1=20, f2=120):
    #Hace un estiramiento del contraste por tramos
    
    img_float = imagen_gris.astype(np.float32)
    
    # Escalado lineal: (f - f1) / (f2 - f1) * 255
    escalada = (img_float - f1) * (255.0 / (f2 - f1)) #Los pixeles con valor f1 pasaran a ser 0 (negro) y los que tengan f2 seran 255 (blanco)
    
    # Saturación de valores fuera de rango [0, 255]
    return np.clip(escalada, 0, 255).astype(np.uint8)
#Fijar f1 y f2 en el rango oscuro donde estan la pupila y el iris resalta con maxima nitidez la frontera entre las estructuras

def aplicar_ecualizaciones(imagen_gris):
  
    # Ecualización Global
    eq_global = cv2.equalizeHist(imagen_gris)
    
    # Ecualización Local Adaptativa (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq_clahe = clahe.apply(imagen_gris)
    
    return eq_global, eq_clahe


def agregar_ruidos(imagen_gris):
   #Simula los distintos tipos de ruidos clasicos en el procesamiento de imagenes
   #El objetivo es generar un conjunto de pruebas apra ver que tan bien responden los distintos filtros espaciales
   
   #Preparacion y normalizacion
    h, w = imagen_gris.shape
    img_float = imagen_gris.astype(np.float32) / 255.0

    #Ruido Gaussiano (media=0, std=0.08)
    gauss = np.random.normal(0, 0.08, (h, w))
    ruido_gauss = np.clip(img_float + gauss, 0, 1)
    
    #Ruido Impulsivo / Sal y Pimienta (probabilidad = 5%)
    ruido_sp = img_float.copy()
    prob = 0.05
    #Sal (Píxeles blancos = 1.0)
    ruido_sp[np.random.rand(h, w) < (prob / 2)] = 1.0
    #Pimienta (Píxeles negros = 0.0)
    ruido_sp[np.random.rand(h, w) < (prob / 2)] = 0.0

    #Ruido Multiplicativo / Speckle
    speckle = np.random.normal(0, 0.15, (h, w))
    ruido_speckle = np.clip(img_float + img_float * speckle, 0, 1)

    return (
        (ruido_gauss * 255).astype(np.uint8),
        (ruido_sp * 255).astype(np.uint8),
        (ruido_speckle * 255).astype(np.uint8)
    )


def aplicar_filtros_espaciales(imagen_ruidosa):
    #Aplica tres filtros espaciales de suavizado sobre una imagen con ruido
    #Para comparar la capacidad de mantener bordes y eliminar ruido de cada tecnica
    
    f_media = cv2.blur(imagen_ruidosa, (5, 5))
    f_gauss = cv2.GaussianBlur(imagen_ruidosa, (5, 5), sigmaX=1.0)
    f_mediana = cv2.medianBlur(imagen_ruidosa, 5)
    
    return f_media, f_gauss, f_mediana


def detectar_bordes(imagen_gris):
    #Detecta bordes sobre una imagen en escala de grises comparando cinco operadores clasicos
    
    #Robert
    k_rob1 = np.array([[1, 0], [0, -1]], dtype=np.float32)
    k_rob2 = np.array([[0, 1], [-1, 0]], dtype=np.float32)
    gx_r = cv2.filter2D(imagen_gris, cv2.CV_32F, k_rob1)
    gy_r = cv2.filter2D(imagen_gris, cv2.CV_32F, k_rob2)
    roberts = cv2.normalize(cv2.magnitude(gx_r, gy_r), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    #Prewitt
    k_prw1 = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
    k_prw2 = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
    gx_p = cv2.filter2D(imagen_gris, cv2.CV_32F, k_prw1)
    gy_p = cv2.filter2D(imagen_gris, cv2.CV_32F, k_prw2)
    prewitt = cv2.normalize(cv2.magnitude(gx_p, gy_p), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    #Sobel
    gx_s = cv2.Sobel(imagen_gris, cv2.CV_32F, 1, 0, ksize=3)
    gy_s = cv2.Sobel(imagen_gris, cv2.CV_32F, 0, 1, ksize=3)
    sobel = cv2.normalize(cv2.magnitude(gx_s, gy_s), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    #Laplaciano
    laplacian = cv2.Laplacian(imagen_gris, cv2.CV_32F, ksize=3)
    laplacian = cv2.normalize(np.abs(laplacian), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    #Canny (Con histéresis)
    canny = cv2.Canny(imagen_gris, 40, 120)

    return roberts, prewitt, sobel, laplacian, canny


def calcular_fourier_y_reconstruccion(img_a, img_b):
    #Analiza la transformada discreta de fourier bidimensional
    #Queremos descomponer la señal espectral en su epectro de maginutd y fase para luego hacer reconstrucciones parciales con la transformada inversa
    
    # FFT Imagen A
    f_a = np.fft.fftshift(np.fft.fft2(img_a.astype(np.float32))) #Calcula la transformada rapida de fourier en 2D ytraslada la componente de frecuencia cero al centro de la matriz
    mag_a = np.abs(f_a) #obtengo el espectro de magnitud
    phase_a = np.angle(f_a) #obtengo el espectro de fase

    # FFT Imagen B (para reconstrucción cruzada)
    f_b = np.fft.fftshift(np.fft.fft2(img_b.astype(np.float32)))
    mag_b = np.abs(f_b)
    phase_b = np.angle(f_b)

    # Reconstrucción Solo Magnitud (Fase constante = 0)
    rec_mag = np.real(np.fft.ifft2(np.fft.ifftshift(mag_a * np.exp(1j * 0))))
    rec_mag = cv2.normalize(rec_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Reconstrucción Solo Fase (Magnitud constante = 1)
    rec_phase = np.real(np.fft.ifft2(np.fft.ifftshift(1.0 * np.exp(1j * phase_a))))
    rec_phase = cv2.normalize(rec_phase, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Reconstrucción Cruzada: Magnitud de Imagen A + Fase de Imagen B
    rec_cross = np.real(np.fft.ifft2(np.fft.ifftshift(mag_a * np.exp(1j * phase_b))))
    rec_cross = cv2.normalize(rec_cross, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return mag_a, phase_a, rec_mag, rec_phase, rec_cross


def calcular_psd(imagen_gris):
    #Calcula la Densidad Espectral de Potencia (PSD = |F|^2) en escala logarítmica.
    f = np.fft.fftshift(np.fft.fft2(imagen_gris.astype(np.float32)))
    psd = np.abs(f) ** 2
    psd_log = 10 * np.log10(psd + 1e-5)
    return psd_log


def aplicar_filtros_frecuenciales(imagen_gris, d0=30):
    #Hace el filtrado en el dominio de la frecuencia usando filtros pasa bajos y pasa altos gaussianos
    
    #Construccion del plano de distancias frecuenciales
    h, w = imagen_gris.shape
    cy, cx = h // 2, w // 2 #determina la coordenada del centro espectral
    y, x = np.ogrid[:h, :w]
    dist_sq = (x - cx) ** 2 + (y - cy) ** 2

    # Filtro Pasa-Bajos Gaussiano: H_lp = exp(-D^2 / (2*D0^2))
    h_lp = np.exp(-dist_sq / (2 * (d0 ** 2)))
    # Filtro Pasa-Altos Gaussiano: H_hp = 1 - H_lp
    h_hp = 1.0 - h_lp
 
    #transformacion al dominio frecuencial (fft 2D)
    f = np.fft.fftshift(np.fft.fft2(imagen_gris.astype(np.float32)))

    # Filtrado en frecuencia (Multiplicación punto a punto)
    f_lp = f * h_lp
    f_hp = f * h_hp

    # Reconstrucción mediante IFFT
    img_lp = np.real(np.fft.ifft2(np.fft.ifftshift(f_lp)))
    img_lp = np.clip(img_lp, 0, 255).astype(np.uint8)

    img_hp = np.real(np.fft.ifft2(np.fft.ifftshift(f_hp)))
    img_hp = cv2.normalize(img_hp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return img_lp, img_hp, h_lp, h_hp #devuelve la imagen suavizada, la imagen de bordes o detalles y ambas mascaras frecuenciales


##################################################################################################
#FUNCION PRINCIPAL DE EJECUCION DEL MODULO 1
#Esta funcion contiene todas las funciones presentadas anteriormente y es la que llama el main
#Recibe toda la informacion que se le otorgo el main por lo que no necesita que se le reingrese el video y su path o el frame elegido

def ejecutar_modulo_1(config):
    print("   EJECUTANDO MÓDULO 1: FUNDAMENTOS Y MEJORAMIENTO ")

    # Extrae parámetros del diccionario global del main
    video_path = config['video_path']
    roi = config['roi']
    frame_idx = config.get('frame_idx', 0)

    # Carga el frame recortado con la ROI previamente definida en main.py
    frame_rgb, roi_rgb = cargar_frame_y_roi(video_path, num_frame=frame_idx, roi_coords=roi)
    roi_gris = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2GRAY)


    # 1.1 Carga, Muestreo y Espacios de color
    print("\n 1.1.3 Análisis de la resolucion espacial segun el submuestreo")
    
    #Ejecutamos la funcion de submuestreo sobre la ROI actual
    h_orig, w_orig = roi_rgb.shape[:2]
    (roi_50, roi_25, roi_05), (vis_50, vis_25, vis_05) = analizar_resolucion_espacial(roi_rgb)

    #FIGURA: Resolución Espacial (1 Fila x 4 Columnas)
    fig_res, ax_res = plt.subplots(1, 4, figsize=(16, 4.5))
    fig_res.canvas.manager.set_window_title("1.1 Resolución Espacial: Submuestreo al 50%, 25% y 5%")

    #Renderizado de la imagen original y las versiones submuestradas
    ax_res[0].imshow(roi_rgb)
    ax_res[0].set_title(f"Original (100%)\n{w_orig}x{h_orig} px")

    # Usamos interpolation='nearest' para forzar que Matplotlib muestre los píxeles reales sin suavizar
    #Si no usaramos esta interpolacion, matplotlib aplica un suavizado antialiasing automatico
    ax_res[1].imshow(vis_50, interpolation='nearest')
    ax_res[1].set_title(f"Submuestreo 50%\n{roi_50.shape[1]}x{roi_50.shape[0]} px")

    ax_res[2].imshow(vis_25, interpolation='nearest')
    ax_res[2].set_title(f"Submuestreo 25%\n{roi_25.shape[1]}x{roi_25.shape[0]} px")

    ax_res[3].imshow(vis_05, interpolation='nearest')
    ax_res[3].set_title(f"Submuestreo 5%\n{roi_05.shape[1]}x{roi_05.shape[0]} px (Pixelado)")

    #Grafico
    for a in ax_res:
        a.axis('off')

    plt.tight_layout()
    plt.show()
    
    print("\n 1.1.4 Análisis de cuantizacion")   
    #Vamos a llamar a la funcion de reduccion de bits para aplicarla sobre la ROI y construir una grilla comparativa  
    
    #Cuantización en escala de grises
    gris_4bit = reducir_cuantizacion(roi_gris, 4)
    gris_2bit = reducir_cuantizacion(roi_gris, 2)

    # Cuantización cromática (RGB)
    rgb_4bit = reducir_cuantizacion(roi_rgb, 4)
    rgb_2bit = reducir_cuantizacion(roi_rgb, 2)

    #FIGURA: Cuantización Grises vs Color
    fig_cuant, ax_c = plt.subplots(2, 3, figsize=(14, 8))
    fig_cuant.canvas.manager.set_window_title("1.1 Cuantización: Degradación de Grises y Cromática")

    #Fila 1: Escala de Grises
    #Presentamos en la primera fila la imagen original en grises y sus versiones de 4 y 2 bits
    #Muestra como disminuir el rango de intensidades elimina los degradados suaves de iluminacion y los convierte en bandas discretas
    ax_c[0, 0].imshow(roi_gris, cmap='gray')
    ax_c[0, 0].set_title("Gris Original (8 bits - 256 niveles)")

    ax_c[0, 1].imshow(gris_4bit, cmap='gray')
    ax_c[0, 1].set_title("Gris 4 bits (16 niveles)")

    ax_c[0, 2].imshow(gris_2bit, cmap='gray')
    ax_c[0, 2].set_title("Gris 2 bits (4 niveles)")

    # Fila 2: Espacio de Color RGB
    ax_c[1, 0].imshow(roi_rgb)
    ax_c[1, 0].set_title("RGB Original (24 bits total - 8 bpc)")

    ax_c[1, 1].imshow(rgb_4bit)
    ax_c[1, 1].set_title("RGB 4 bits por canal (12 bits total)")

    ax_c[1, 2].imshow(rgb_2bit)
    ax_c[1, 2].set_title("RGB 2 bits por canal (6 bits total)")

    #Grafico
    for a in ax_c.ravel():
        a.axis('off')

    plt.tight_layout()
    plt.show()  
    
    
    print("\n 1.1.5 Análisis Individual de Canales en Escala de Grises")
    #Queremos analizar que canal independiente de RGB presenta mayor relacion contraste-ruido para aislar la pupila antes de pasar a la binarizacion
    #Llamamos a la funcion de descomposicion cromatica para aplicarla en la ROI
    (r, g, b), (h, s, v), roi_hsv = analizar_espacios_color(roi_rgb)

    #FIGURA: Canales RGB e HSV en escala de grises
    fig_color, ax_col = plt.subplots(2, 4, figsize=(16, 7))
    fig_color.canvas.manager.set_window_title("1.1 Espacios de Color: Inspección Individual de Canales")

    #Fila 1: Muestra la ROI original en color verdadero en [0,0]
    #En las siguientes posiciones despliega cada matriz, forzando el mapa de color monocromatico
    #Al mapear cada canal como intensidad de gris se puede observar como cada canal satura o define mejor distintos aspectos de la imagen
    ax_col[0, 0].imshow(roi_rgb)
    ax_col[0, 0].set_title("ROI Original (RGB)")

    ax_col[0, 1].imshow(r, cmap='gray')
    ax_col[0, 1].set_title("Canal R (Rojo - Gris)")

    ax_col[0, 2].imshow(g, cmap='gray')
    ax_col[0, 2].set_title("Canal G (Verde - Gris)")

    ax_col[0, 3].imshow(b, cmap='gray')
    ax_col[0, 3].set_title("Canal B (Azul - Gris)")

    # Fila 2: Imagen HSV + Componentes H, S, V
    ax_col[1, 0].imshow(roi_hsv)
    ax_col[1, 0].set_title("Imagen en Espacio HSV")

    ax_col[1, 1].imshow(h, cmap='gray')
    ax_col[1, 1].set_title("Componente H (Matiz / Hue)")

    ax_col[1, 2].imshow(s, cmap='gray')
    ax_col[1, 2].set_title("Componente S (Saturación)")

    ax_col[1, 3].imshow(v, cmap='gray')
    ax_col[1, 3].set_title("Componente V (Valor / Brillo)")

    #Grafico
    for a in ax_col.ravel():
        a.axis('off')

    plt.tight_layout()
    plt.show()


    # 1.2 CALIDAD, TRANSFORMACIONES DE INTENSIDAD Y FILTRADO ESPACIAL
    print(" 1.2 TRANSFORMACIONES DE INTENSIDAD Y FILTRADO")
    
    # 1.2.1 HISTOGRAMA Y BRILLO / CONTRASTE
    #Queremos aplicar la transformacion lineal de intensidad sobre la ROI en escala de grises y presentar la comparacion visual 
    #directa y el comportamiento de los histogramas
    
    print("\n Analizando Histograma, Brillo (+50) y Contraste (x1.5)")
    img_mod = modificar_brillo_contraste(roi_gris, alpha=1.5, beta=50) #invoca la funcion definida y le determina un valor de alfa y beta que pueden ser cambiados

    #Organizacion de la figura
    fig121, ax = plt.subplots(2, 2, figsize=(13, 7))
    fig121.canvas.manager.set_window_title("1.2.1 Brillo (+50) y Contraste (x1.5)")

    ax[0, 0].imshow(roi_gris, cmap='gray', vmin=0, vmax=255) #Fijamos el rango para que matplotlib no auto escale el contraste visual del grafico
    ax[0, 0].set_title("Original (Gris)")
    ax[0, 0].axis('off')

    ax[0, 1].imshow(img_mod, cmap='gray', vmin=0, vmax=255)
    ax[0, 1].set_title("Brillo (+50) & Contraste (x1.5)")
    ax[0, 1].axis('off')

    #Calculo de escala para el histograma
    counts_orig, _ = np.histogram(roi_gris.ravel(), bins=256, range=[0, 256])
    max_y = counts_orig.max() * 1.5  # Escala basada en el pico original, para que no aplaste e invisibilice el resto de la forma del histograma

    #Grafico
    ax[1, 0].hist(roi_gris.ravel(), bins=256, range=[0, 256], color='black', alpha=0.7)
    ax[1, 0].set_title("Histograma Original")
    ax[1, 0].grid(True, linestyle='--', alpha=0.5)

    ax[1, 1].hist(img_mod.ravel(), bins=256, range=[0, 256], color='darkblue', alpha=0.7)
    ax[1, 1].set_title("Histograma Modificado")
    ax[1, 1].set_ylim(0, max_y)  # <-- ESTO EVITA QUE LA BARRA DE 255 APLASTE EL RESTO
    ax[1, 1].grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


    # 1.2.2 Y 1.2.3 ESCALADO POR RANGO Y ECUALIZACIONES (GLOBAL VS CLAHE)
    print("\n Evaluando Escalado [f1, f2], Ecualización Global y CLAHE")
    
    f1, f2 = 15, 110  # Rango de intensidades centrado en pupila e iris
    img_escalada = escalado_rango_interes(roi_gris, f1=f1, f2=f2)
    img_eq_global, img_clahe = aplicar_ecualizaciones(roi_gris)

    #Configuracion del grafico
    fig122, ax_eq = plt.subplots(2, 4, figsize=(16, 7))
    fig122.canvas.manager.set_window_title("1.2.2 y 1.2.3 Escalado por Rango vs Ecualización Global vs CLAHE")

    # Fila 1: Imágenes
    ax_eq[0, 0].imshow(roi_gris, cmap='gray')
    ax_eq[0, 0].set_title("Original")
    ax_eq[0, 1].imshow(img_escalada, cmap='gray')
    ax_eq[0, 1].set_title(f"Escalado [{f1}, {f2}]")
    ax_eq[0, 2].imshow(img_eq_global, cmap='gray')
    ax_eq[0, 2].set_title("Ecualización Global")
    ax_eq[0, 3].imshow(img_clahe, cmap='gray')
    ax_eq[0, 3].set_title("CLAHE (Local)")

    for a in ax_eq[0, :]:
        a.axis('off')

    # Fila 2: Histogramas
    ax_eq[1, 0].hist(roi_gris.ravel(), bins=256, range=[0, 256], color='black', alpha=0.7)
    ax_eq[1, 0].set_title("Hist. Original") #Muestra la distribucion acumuluda tipica

    ax_eq[1, 1].hist(img_escalada.ravel(), bins=256, range=[0, 256], color='darkgreen', alpha=0.7)
    ax_eq[1, 1].set_title(f"Hist. Escalado [{f1}, {f2}]") #Muestra la expansion del rango dinamico entre 15 y 110
    ax_eq[1, 1].set_ylim(0, 10000) #Recortamos visualmente el pico de saturacion de 255
    
    ax_eq[1, 2].hist(img_eq_global.ravel(), bins=256, range=[0, 256], color='darkred', alpha=0.7)
    ax_eq[1, 2].set_title("Hist. Ecualizado Global") #Muestra la aplanacion de la densidad acumulada intentando distribuir los pixeles uniformemente entre 0 y 255

    ax_eq[1, 3].hist(img_clahe.ravel(), bins=256, range=[0, 256], color='purple', alpha=0.7)
    ax_eq[1, 3].set_title("Hist. CLAHE") #Histograma suave y continuo que mejora el contraste

    #Grafico
    for a in ax_eq[1, :]:
        a.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

    
    # 1.2.4 SIMULACIÓN DE RUIDOS (GAUSSIANO, SAL Y PIMIENTA, SPECKLE)
    print("\n Generando patrones de ruido en la ROI")
    
    #Se llama a la funcion que genera ruido sintetico sobre la imagen y presenta las comparaciones entre imagenes
    r_gauss, r_sp, r_speckle = agregar_ruidos(roi_gris)

    fig_ruido, ax_r = plt.subplots(1, 4, figsize=(16, 4.5))
    fig_ruido.canvas.manager.set_window_title("1.2.4 Modelos de Ruido")

    ax_r[0].imshow(roi_gris, cmap='gray'); ax_r[0].set_title("Sin Ruido (Original)")
    ax_r[1].imshow(r_gauss, cmap='gray'); ax_r[1].set_title("Ruido Gaussiano")
    ax_r[2].imshow(r_sp, cmap='gray'); ax_r[2].set_title("Ruido Sal y Pimienta")
    ax_r[3].imshow(r_speckle, cmap='gray'); ax_r[3].set_title("Ruido Speckle")

    for a in ax_r:
        a.axis('off')

    plt.tight_layout()
    plt.show()

   
    # 1.2.4 COMPARATIVA DE FILTRADO ESPACIAL (MEDIA, GAUSSIANO, MEDIANA)
    print("\n Evaluando la respuesta de los Filtros Espaciales")
    #Llamamos a la funcion que aplica los distintos filtros sobre las diferentes imagenes generadas con ruido para poder observar que filtro funciona mejor para cada tipo de ruido
    
    ruidos = [("Gaussiano", r_gauss), ("Sal y Pimienta", r_sp), ("Speckle", r_speckle)]
    fig_filt, ax_f = plt.subplots(3, 4, figsize=(16, 10))
    fig_filt.canvas.manager.set_window_title("1.2.4 Matriz de Filtrado Espacial")

    #Bucle de filtrado por fila (iteracion sobre los distintos tipos de ruido)
    for i, (nombre_ruido, img_r) in enumerate(ruidos):
        f_med, f_gau, f_median = aplicar_filtros_espaciales(img_r)

        # Columna 0: Imagen Ruidosa
        ax_f[i, 0].imshow(img_r, cmap='gray')
        ax_f[i, 0].set_ylabel(nombre_ruido, fontsize=11, fontweight='bold')
        if i == 0: ax_f[i, 0].set_title("Imagen Ruidosa")

        # Columna 1: Filtro Media
        ax_f[i, 1].imshow(f_med, cmap='gray')
        if i == 0: ax_f[i, 1].set_title("Filtro Media (5x5)")

        # Columna 2: Filtro Gaussiano
        ax_f[i, 2].imshow(f_gau, cmap='gray')
        if i == 0: ax_f[i, 2].set_title("Filtro Gaussiano (5x5)")

        # Columna 3: Filtro Mediana
        ax_f[i, 3].imshow(f_median, cmap='gray')
        if i == 0: ax_f[i, 3].set_title("Filtro Mediana (5x5)")

    for a in ax_f.ravel():
        a.set_xticks([])
        a.set_yticks([])

    plt.tight_layout()
    plt.show()


    # 1.3 DETECCIÓN DE BORDES Y DOMINIO FRECUENCIAL
   
    # 1.3.1 DETECCIÓN DE BORDES (ORIGINAL VS FILTRADA)
    print("\n Evaluando Operadores de Bordes (Original vs Filtro Mediana)")
    #Aplicamos distintos operadores de deteccion de bordes para analizar el impacto que tiene el suavizado previo
    #con un filtro de mediana antes de aplicar los algoritmos de extraccion de bordes
    
    # Aplicamos filtro Mediana previsor para reducir ruido manteniendo bordes
    img_filtrada_med = cv2.medianBlur(roi_gris, 5)

    #Deteccion de bordes en paralelo
    #Se hace sobre la imagen sin procesar (sufijo _o) y sobre la imagen suavizada (_f)
    rob_o, prw_o, sob_o, lap_o, can_o = detectar_bordes(roi_gris) 
    rob_f, prw_f, sob_f, lap_f, can_f = detectar_bordes(img_filtrada_med)

    fig_bordes, ax_b = plt.subplots(2, 5, figsize=(18, 7))
    fig_bordes.canvas.manager.set_window_title("1.3.1 Comparativa de Operadores de Bordes")
    
    #Bucle de renderizado
    #Va a iterar sobre las 5 columnas alineando cada operador con su respectivo resultado en ambas filas
    #Usa titulos multilinea (\n) para identificar el operador y la condicion de la imagen
    nombres_ops = ["Robert", "Prewitt", "Sobel", "Laplaciano", "Canny"]
    bordes_orig = [rob_o, prw_o, sob_o, lap_o, can_o]
    bordes_filt = [rob_f, prw_f, sob_f, lap_f, can_f]

    for j in range(5):
        # Fila 1: Imagen Original
        ax_b[0, j].imshow(bordes_orig[j], cmap='gray')
        ax_b[0, j].set_title(f"{nombres_ops[j]}\n(Original)")
        ax_b[0, j].axis('off')

        # Fila 2: Imagen Filtrada
        ax_b[1, j].imshow(bordes_filt[j], cmap='gray')
        ax_b[1, j].set_title(f"{nombres_ops[j]}\n(Tras Filtro Mediana)")
        ax_b[1, j].axis('off')

    plt.tight_layout()
    plt.show()

    
    # 1.3.2 TRANSFORMADA DE FOURIER 2D Y RECONSTRUCCIÓN CRUZADA
    print("\n Calculando TFD-2D y Reconstrucciones (Magnitud vs Fase)")

    #Imagen B sintética (Círculo representando una pupila) para prueba cruzada
    h_r, w_r = roi_gris.shape
    img_sintetica = np.zeros((h_r, w_r), dtype=np.uint8)
    cv2.circle(img_sintetica, (w_r // 2, h_r // 2), min(h_r, w_r) // 4, 255, -1)

    #Invocamos la funcion de descomposicion espectral pasandole la imagen real y el disco sintetico
    mag_a, phase_a, rec_mag, rec_phase, rec_cross = calcular_fourier_y_reconstruccion(roi_gris, img_sintetica)

    fig_fourier, ax_f2 = plt.subplots(2, 3, figsize=(15, 8))
    fig_fourier.canvas.manager.set_window_title("1.3.2 TFD-2D: Magnitud, Fase y Reconstrucciones")

    #Renderizado
    #Fila 1: Espectros
    ax_f2[0, 0].imshow(roi_gris, cmap='gray'); ax_f2[0, 0].set_title("ROI Original (Imagen A)")
    ax_f2[0, 1].imshow(np.log(1 + mag_a), cmap='gray'); ax_f2[0, 1].set_title("Espectro Magnitud (Log)")
    ax_f2[0, 2].imshow(phase_a, cmap='gray'); ax_f2[0, 2].set_title("Espectro Fase")

    #Fila 2: Reconstrucciones
    ax_f2[1, 0].imshow(rec_mag, cmap='gray'); ax_f2[1, 0].set_title("Reconstrucción Solo Magnitud")
    ax_f2[1, 1].imshow(rec_phase, cmap='gray'); ax_f2[1, 1].set_title("Reconstrucción Solo Fase")
    ax_f2[1, 2].imshow(rec_cross, cmap='gray'); ax_f2[1, 2].set_title("Cruzada: Mag(A) + Fase(Pupila B)")

    for a in ax_f2.ravel(): a.axis('off')

    plt.tight_layout()
    plt.show()

    
    # 1.3.3 DENSIDAD ESPECTRAL DE POTENCIA (PSD)
    print("\n Estimando Densidad Espectral de Potencia (PSD)")
    
    #Calculo de la densidad espectral de potencia
    #calcular_psd evalua la firma espectral sobre la ROI en escala de grises original y sobre la version procesada con el filtro de mediana
    psd_orig = calcular_psd(roi_gris)
    psd_filt = calcular_psd(img_filtrada_med)

    fig_psd, ax_psd = plt.subplots(1, 2, figsize=(13, 5))
    fig_psd.canvas.manager.set_window_title("1.3.3 Densidad Espectral de Potencia (PSD)")

    #Renderizado y mapeo termico
    ax_psd[0].imshow(psd_orig, cmap='inferno') #inferno emplea una escala pseudocolor uniforme. Esto facilita la interpretacion visual de gradientes espectrales
    ax_psd[0].set_title("PSD Original (Alta energía en Bajas/Altas Frecuencias)")
    ax_psd[0].axis('off')

    ax_psd[1].imshow(psd_filt, cmap='inferno')
    ax_psd[1].set_title("PSD Tras Filtro Mediana (Atenuación de Altas Frecuencias)")
    ax_psd[1].axis('off')

    plt.tight_layout()
    plt.show()

   
    # 1.3.4 FILTRADO EN EL DOMINIO FRECUENCIAL VS ESPACIAL
    #Demuestra de manera practica el teorema de convolucion al comparar la respuesta del filtrado gaussiano en el dominio de la frecuencia frente a la convolucion directa en el dominio del espacio
    print("\n Aplicando Filtros Pasa-Bajos y Pasa-Altos Frecuenciales")

    #Procesamiento frecuencial y filtrado espacial equivalente
    #Se fija la frecuencia de corte y se devuelven las imagenes reconstruidas usando IFFT y sus mascaras Gaussianas
    img_lp, img_hp, h_lp, h_hp = aplicar_filtros_frecuenciales(roi_gris, d0=30)
    #Equivalente Espacial para comparación (Gaussiano 5x5)
    img_gauss_espacial = cv2.GaussianBlur(roi_gris, (5, 5), 1.0)

    fig_freq, ax_fq = plt.subplots(2, 3, figsize=(15, 8))
    fig_freq.canvas.manager.set_window_title("1.3.4 Filtrado Frecuencial vs Espacial")

    #Renderizado
    #Máscaras
    ax_fq[0, 0].imshow(h_lp, cmap='gray'); ax_fq[0, 0].set_title("Máscara Pasa-Bajos Gaussiana")
    ax_fq[0, 1].imshow(h_hp, cmap='gray'); ax_fq[0, 1].set_title("Máscara Pasa-Altos Gaussiana")
    ax_fq[0, 2].imshow(roi_gris, cmap='gray'); ax_fq[0, 2].set_title("ROI Original")
    #Reconstrucciones
    ax_fq[1, 0].imshow(img_lp, cmap='gray'); ax_fq[1, 0].set_title("Reconstrucción Pasa-Bajos (Suavizado)")
    ax_fq[1, 1].imshow(img_hp, cmap='gray'); ax_fq[1, 1].set_title("Reconstrucción Pasa-Altos (Bordes/Detalles)")
    ax_fq[1, 2].imshow(img_gauss_espacial, cmap='gray'); ax_fq[1, 2].set_title("Filtro Gaussiano Espacial (Equivalente)")

    for a in ax_fq.ravel(): a.axis('off')

    plt.tight_layout()
    plt.show()
    #PREGUNTAS DEL FINAL DEL MODULO
    print("\n[PREGUNTA 1]")
    print("¿Qué filtro preserva de manera idónea los bordes de la pupila eliminando")
    print("los destellos/reflejos del flash (glints)?\n")
    print("FILTRO DE MEDIANA (cv2.medianBlur)")
    print("    • Tipo: Filtro no lineal estadístico.")
    print("    • Mecanismo: Ordena los píxeles del vecindario y selecciona la mediana como valor central.")
    print("    • Justificación: A diferencia de la Media o Gaussiano, no promedia valores.")
    print("      Los destellos de luz o ruido impulsivo (valores extremos de intensidad) quedan")
    print("      completamente descartados al ordenar los píxeles. Elimina el reflejo AL 100% SIN DESENFOCAR")
    print("      ni desplazar la frontera del limbo pupilar.")



    print("\n[PREGUNTA 2]")
    print("¿Cómo cuantificar la calidad de una imagen al compararla con la original?")
    print("Calcular PSNR/SSIM.\n")
    print(" Se emplean métricas de referencia completa (Full-Reference):")
    print("    1. PSNR (Peak Signal-to-Noise Ratio): Mide la relación entre la máxima potencia posible de")
    print("       la señal y el error cuadrático medio (MSE) introducido (expresado en dB). Valora la")
    print("       fidelidad del píxel (valores > 30 dB indican alta calidad).")
    print("    2. SSIM (Structural Similarity Index Measure): Evalúa la degradación visual percibida")
    print("       analizando Luminancia, Contraste y Estructura. Varía entre -1 y 1 (1 = idénticas).")
     
    if 'roi_gris' in locals() or 'roi_gris' in globals():
        img_orig = roi_gris
    elif 'roi_rgb' in locals() or 'roi_rgb' in globals():
        img_orig = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2GRAY)
    else:
    # Genera un patrón sintético de pupila con un destello (glint)
        img_orig = np.full((256, 256), 120, dtype=np.uint8)
    cv2.circle(img_orig, (128, 128), 50, 15, -1)   # Pupila oscura
    cv2.circle(img_orig, (140, 110), 6, 255, -1)   # Reflejo del flash

    # Imagen procesada mediante filtro de mediana (para eliminar el destello)
    img_proc = cv2.medianBlur(img_orig, 5)


    #CÁLCULO MANUAL (PASO A PASO)

    mse = np.mean((img_orig.astype(np.float64) - img_proc.astype(np.float64)) ** 2)
    max_pixel = 255.0

    if mse == 0:
        psnr_manual = float('inf')
    else:
        psnr_manual = 10.0 * np.log10((max_pixel ** 2) / mse)


    #CÁLCULO MEDIANTE LIBRERÍAS ESPECIALIZADAS
    psnr_cv = cv2.PSNR(img_orig, img_proc)
    psnr_lib = psnr_sk(img_orig, img_proc, data_range=255)
    ssim_lib = ssim_sk(img_orig, img_proc, data_range=255)


    #IMPRESIÓN DIRECTA EN LA TERMINAL

    print("\n" + "=" * 65, flush=True)
    print("        EVALUACIÓN CUANTITATIVA DE CALIDAD (PSNR / SSIM)", flush=True)
    print("=" * 65, flush=True)
    print(f" • Error Cuadrático Medio (MSE) : {mse:.4f}", flush=True)
    print(f" • PSNR (Fórmula Manual)        : {psnr_manual:.2f} dB", flush=True)
    print(f" • PSNR (cv2.PSNR)              : {psnr_cv:.2f} dB", flush=True)
    print(f" • PSNR (scikit-image)          : {psnr_lib:.2f} dB", flush=True)
    print(f" • SSIM (Índice Estructural)    : {ssim_lib:.4f} / 1.0000", flush=True)
    print("=" * 65 + "\n", flush=True)

    sys.stdout.flush()


    print("\n[ÉXITO] Módulo 1 completado.")