import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def cargar_frame_y_roi(video_path, num_frame=0, roi_coords=None):
    """Carga un frame específico del video y recorta la ROI elegida en main.py."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, num_frame)
    ret, frame_bgr = cap.read()
    cap.release()
    
    if not ret or frame_bgr is None:
        raise ValueError("No se pudo leer el frame del video.")
        
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    
    if roi_coords is not None:
        ymin, ymax, xmin, xmax = roi_coords
        roi_rgb = frame_rgb[ymin:ymax, xmin:xmax]
    else:
        roi_rgb = frame_rgb
        
    return frame_rgb, roi_rgb

def analizar_resolucion_espacial(roi_rgb):
    """
    Realiza el submuestreo de la ROI al 50%, 25% y 5% de la resolución original.
    Devuelve las imágenes en su dimensión reducida y también re-escaladas 
    al tamaño original para apreciar el efecto de pixelado (aliasing espacial).
    """
    h_orig, w_orig = roi_rgb.shape[:2]

    # 1. Submuestreo (Reducción espacial)
    roi_50 = cv2.resize(roi_rgb, (0, 0), fx=0.50, fy=0.50, interpolation=cv2.INTER_AREA)
    roi_25 = cv2.resize(roi_rgb, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    roi_05 = cv2.resize(roi_rgb, (0, 0), fx=0.05, fy=0.05, interpolation=cv2.INTER_AREA)

    # 2. Re-escalado al tamaño original para comparación visual directa
    vis_50 = cv2.resize(roi_50, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
    vis_25 = cv2.resize(roi_25, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
    vis_05 = cv2.resize(roi_05, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)

    return (roi_50, roi_25, roi_05), (vis_50, vis_25, vis_05)

def reducir_cuantizacion(imagen, bits):
    """
    Reduce la profundidad de bits de 8 bits a 'bits' bits.
    Aplica para imágenes en escala de grises (2D) y en color RGB (3D).
    """
    niveles = 2 ** bits
    paso = 256 // niveles
    return ((imagen // paso) * paso).astype(np.uint8)


def descomponer_canales_grises(roi_rgb):
    """
    Separa los canales de color de los espacios RGB y HSV para analizarlos
    de forma individual como imágenes monocromáticas en escala de grises.
    """
    # Espacio RGB
    r, g, b = cv2.split(roi_rgb)
    
    # Espacio HSV
    roi_hsv = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(roi_hsv)
    
    return (r, g, b), (h, s, v)



# =============================================================================
# FUNCIÓN PRINCIPAL DEL MÓDULO (Llamada desde main.py)
# =============================================================================

def ejecutar_modulo_1(config):
    """
    Ejecuta el Módulo 1 usando la configuración de main.py.
    No vuelve a pedir seleccionar el video ni la ROI.
    Muestra las figuras de a una secuencialmente.
    """
    print("\n==================================================")
    print("   EJECUTANDO MÓDULO 1: FUNDAMENTOS Y MEJORAMIENTO ")
    print("==================================================")

    # Extrae parámetros del diccionario global
    video_path = config['video_path']
    roi = config['roi']
    frame_idx = config.get('frame_idx', 0)

    # Carga el frame recortado con la ROI previamente definida en main.py
    frame_rgb, roi_rgb = cargar_frame_y_roi(video_path, num_frame=frame_idx, roi_coords=roi)
    roi_gris = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2GRAY)


   # -------------------------------------------------------------------------
    # 1.1 Carga, Muestreo y Espacios de color
    # -------------------------------------------------------------------------
    print("\n 1.1.3 Análisis de la resolucion espacial segun el submuestreo")
    h_orig, w_orig = roi_rgb.shape[:2]
    (roi_50, roi_25, roi_05), (vis_50, vis_25, vis_05) = analizar_resolucion_espacial(roi_rgb)

    # FIGURA: Resolución Espacial
    fig_res, ax_res = plt.subplots(2, 4, figsize=(16, 7))
    fig_res.canvas.manager.set_window_title("1.1 Resolución Espacial: Submuestreo al 50%, 25% y 5%")

    # Fila 1: Resoluciones en escala real
    ax_res[0, 0].imshow(roi_rgb)
    ax_res[0, 0].set_title(f"Original (100%)\n{w_orig}x{h_orig} px")

    ax_res[0, 1].imshow(roi_50)
    ax_res[0, 1].set_title(f"Submuestreo 50%\n{roi_50.shape[1]}x{roi_50.shape[0]} px")

    ax_res[0, 2].imshow(roi_25)
    ax_res[0, 2].set_title(f"Submuestreo 25%\n{roi_25.shape[1]}x{roi_25.shape[0]} px")

    ax_res[0, 3].imshow(roi_05)
    ax_res[0, 3].set_title(f"Submuestreo 5%\n{roi_05.shape[1]}x{roi_05.shape[0]} px")

    # Fila 2: Re-escalados al tamaño original para visualizar pérdida de definición
    ax_res[1, 0].imshow(roi_rgb)
    ax_res[1, 0].set_title("Original (Referencia)")

    ax_res[1, 1].imshow(vis_50)
    ax_res[1, 1].set_title("50% (Pérdida sutil de detalles)")

    ax_res[1, 2].imshow(vis_25)
    ax_res[1, 2].set_title("25% (Bordes escalonados / Pixelado)")

    ax_res[1, 3].imshow(vis_05)
    ax_res[1, 3].set_title("5% (Pérdida total de morfología)")

    for a in ax_res.ravel():
        a.axis('off')

    plt.tight_layout()
    plt.show()  
    print("\n 1.1.4 Análisis de cuantizacion")     
        
    print("\n 1.1.5 Análisis Individual de Canales en Escala de Grises")
    
    (r, g, b), (h, s, v) = descomponer_canales_grises(roi_rgb)

    # FIGURA 1: Canales RGB y HSV en Escala de Grises
    fig1, ax1 = plt.subplots(2, 4, figsize=(16, 8))
    fig1.canvas.manager.set_window_title("1.1 Análisis de Canales Individuales en Escala de Grises")

    # Fila 1: Imagen Original + Canales RGB monocromáticos
    ax1[0, 0].imshow(roi_rgb)
    ax1[0, 0].set_title("ROI RGB Original")
    
    ax1[0, 1].imshow(r, cmap='gray')
    ax1[0, 1].set_title("Canal R (Rojo)")
    
    ax1[0, 2].imshow(g, cmap='gray')
    ax1[0, 2].set_title("Canal G (Verde)")
    
    ax1[0, 3].imshow(b, cmap='gray')
    ax1[0, 3].set_title("Canal B (Azul)")

    # Fila 2: Gris Conversión Estándar + Componentes HSV monocromáticos
    ax1[1, 0].imshow(roi_gris, cmap='gray')
    ax1[1, 0].set_title("Gris Estándar (Luminancia)")
    
    ax1[1, 1].imshow(h, cmap='gray')
    ax1[1, 1].set_title("Componente H (Matiz)")
    
    ax1[1, 2].imshow(s, cmap='gray')
    ax1[1, 2].set_title("Componente S (Saturación)")
    
    ax1[1, 3].imshow(v, cmap='gray')
    ax1[1, 3].set_title("Componente V (Valor/Brillo)")

    for a in ax1.ravel(): 
        a.axis('off')
        
    plt.tight_layout()
    plt.show()  # Pausa la ejecución hasta cerrar la figura

    # FIGURA 1.B (OPCIONAL): Histogramas de Intensidad para comparar Contraste
    fig_hist, ax_h = plt.subplots(1, 3, figsize=(15, 4))
    fig_hist.canvas.manager.set_window_title("1.1.B Histogramas de Intensidad (Separación Pupila/Fondo)")

    ax_h[0].hist(roi_gris.ravel(), bins=256, range=[0, 256], color='black')
    ax_h[0].set_title("Histograma Gris Estándar")
    ax_h[0].set_ylabel("Frecuencia (píxeles)")

    ax_h[1].hist(r.ravel(), bins=256, range=[0, 256], color='red', alpha=0.7)
    ax_h[1].set_title("Histograma Canal R")

    ax_h[2].hist(v.ravel(), bins=256, range=[0, 256], color='purple', alpha=0.7)
    ax_h[2].set_title("Histograma Componente V (HSV)")

    for a in ax_h:
        a.set_xlabel("Nivel de Gris (0-255)")
        a.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

    # -------------------------------------------------------------------------
    # 1.2 CALIDAD, TRANSFORMACIONES Y FILTRADO ESPACIAL
    # -------------------------------------------------------------------------
    img_bc = cv2.convertScaleAbs(roi_gris, alpha=1.5, beta=50)

    f1, f2 = 30, 120
    img_f = roi_gris.astype(np.float32)
    img_scale = np.clip((img_f - f1) / (f2 - f1 + 1e-5) * 255.0, 0, 255).astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq_clahe = clahe.apply(roi_gris)

    r_sp = (roi_gris.astype(np.float32) / 255.0).copy()
    prob = 0.04
    r_sp[np.random.rand(*r_sp.shape) < (prob / 2)] = 1.0
    r_sp[np.random.rand(*r_sp.shape) < (prob / 2)] = 0.0
    r_sp = (r_sp * 255.0).astype(np.uint8)

    f_media = cv2.blur(r_sp, (5, 5))
    f_mediana = cv2.medianBlur(r_sp, 5)

    # FIGURA 2
    fig2, ax2 = plt.subplots(2, 3, figsize=(12, 7))
    fig2.canvas.manager.set_window_title("1.2 Transformaciones e Intensidad (Cierra esta ventana para ver la siguiente)")
    ax2[0, 0].imshow(roi_gris, cmap='gray'); ax2[0, 0].set_title("Original (Gris)")
    ax2[0, 1].imshow(img_bc, cmap='gray'); ax2[0, 1].set_title("Brillo (+50) & Contraste (x1.5)")
    ax2[0, 2].imshow(eq_clahe, cmap='gray'); ax2[0, 2].set_title("Ecualización CLAHE")
    ax2[1, 0].imshow(r_sp, cmap='gray'); ax2[1, 0].set_title("Ruido Sal y Pimienta")
    ax2[1, 1].imshow(f_media, cmap='gray'); ax2[1, 1].set_title("Filtro Media (5x5)")
    ax2[1, 2].imshow(f_mediana, cmap='gray'); ax2[1, 2].set_title("Filtro Mediana (5x5)")
    for a in ax2.ravel(): a.axis('off')
    plt.tight_layout()
    plt.show()  # <-- Pausa el programa hasta que cierres la ventana

    # -------------------------------------------------------------------------
    # 1.3 DETECCIÓN DE BORDES Y DOMINIO FRECUENCIAL
    # -------------------------------------------------------------------------
    b_canny_orig = cv2.Canny(roi_gris, 40, 120)
    b_canny_filt = cv2.Canny(f_mediana, 40, 120)

    f_transform = np.fft.fft2(roi_gris.astype(np.float32))
    f_shift = np.fft.fftshift(f_transform)
    mag_log = np.log(1.0 + np.abs(f_shift))
    fase = np.angle(f_shift)

    rec_mag = np.real(np.fft.ifft2(np.fft.ifftshift(np.abs(f_shift) * np.exp(1j * 0))))
    rec_fase = np.real(np.fft.ifft2(np.fft.ifftshift(1.0 * np.exp(1j * fase))))
    rec_mag = cv2.normalize(rec_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    rec_fase = cv2.normalize(rec_fase, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    psnr_sp = psnr(roi_gris, r_sp, data_range=255)
    ssim_sp = ssim(roi_gris, r_sp, data_range=255)
    psnr_med = psnr(roi_gris, f_mediana, data_range=255)
    ssim_med = ssim(roi_gris, f_mediana, data_range=255)

    print("\n--- 1.3 Cuantificación de Calidad (PSNR / SSIM) ---")
    print(f"Imagen Ruidosa (Sal y Pimienta): PSNR = {psnr_sp:.2f} dB | SSIM = {ssim_sp:.4f}")
    print(f"Imagen Restaurada (Filtro Mediana): PSNR = {psnr_med:.2f} dB | SSIM = {ssim_med:.4f}")

    # FIGURA 3
    fig3, ax3 = plt.subplots(2, 3, figsize=(12, 7))
    fig3.canvas.manager.set_window_title("1.3 Bordes y Fourier 2D")
    ax3[0, 0].imshow(b_canny_orig, cmap='gray'); ax3[0, 0].set_title("Canny (Original)")
    ax3[0, 1].imshow(b_canny_filt, cmap='gray'); ax3[0, 1].set_title("Canny (Tras Filtro Mediana)")
    ax3[0, 2].imshow(mag_log, cmap='gray'); ax3[0, 2].set_title("TFD-2D (Espectro Magnitud)")
    ax3[1, 0].imshow(rec_mag, cmap='gray'); ax3[1, 0].set_title("Reconstrucción Solo Magnitud")
    ax3[1, 1].imshow(rec_fase, cmap='gray'); ax3[1, 1].set_title("Reconstrucción Solo Fase")
    ax3[1, 2].imshow(fase, cmap='gray'); ax3[1, 2].set_title("TFD-2D (Espectro Fase)")
    for a in ax3.ravel(): a.axis('off')
    plt.tight_layout()
    plt.show()  # <-- Regresa al menú principal al cerrar esta última figura

    print("\n[ÉXITO] Módulo 1 completado.")