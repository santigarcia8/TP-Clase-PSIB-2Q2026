#Modulo 4: Compresion de la informacion de la imagen
import cv2
import sys
import heapq
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from skimage.metrics import peak_signal_noise_ratio as psnr_sk
from skimage.metrics import structural_similarity as ssim_sk

#Definicion de funciones auxiliares

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


def segmentar_roi_pupila(roi_gris):
    #Toma la region de interes del ojo en escalas de grises y genera una mascara binaria donde la pupila queda resaltada en blanco (255)
    #y el resto de la imagen en negro (0)
    
    roi_suave = cv2.medianBlur(roi_gris, 5) #aplica un filtro estadistico de mediana para eliminar ruido impulsivo
    min_val = np.min(roi_suave) #busca el valor de intensidad minimo de la imagen
    thresh_val = np.percentile(roi_suave, 12)#calcula el percentil 12 de las intensidades en escala de grises. Asumimos que el 12% de los pixeles mas oscuros del recorte corresponden a la pupila
    thresh_val = np.clip(thresh_val, min_val + 10, min_val + 40) #Limita los umbrales obtenidos
    _, roi_binaria = cv2.threshold(roi_suave, thresh_val, 255, cv2.THRESH_BINARY_INV) #aplica la binarizacion inversa
    return roi_binaria #devuelve mascara limpia en blanco y negro


def estimar_geometria_pupila(roi_rgb):
   #Aisla la pupila dentro de la ROI recortada
   #Calcula su area, su diametro promedio (en pixeles) y devuelve la imagen con la elipse de ajuste
    roi_gris = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2GRAY)
    h, w = roi_gris.shape
    centro_roi = (w / 2.0, h / 2.0)
    
    # Binarización enfocada exclusivamente en los tonos más oscuros (Pupila)
    roi_binaria = segmentar_roi_pupila(roi_gris)
    
    # Limpieza morfológica para eliminar reflejos y pestañas
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    apertura = cv2.morphologyEx(roi_binaria, cv2.MORPH_OPEN, kernel)
    limpia = cv2.morphologyEx(apertura, cv2.MORPH_CLOSE, kernel)
    
    contornos, _ = cv2.findContours(limpia, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #Extrae contronos externos
    roi_dibujada = roi_rgb.copy()
    
    if not contornos:
        return 0.0, 0.0, roi_dibujada

    area_max_permitida = (h * w) * 0.30  # La pupila no supera el 30% de la ROI, para que no binarice la cuenca del ojo en vez de la pupila
    
    #Cálculo de distancia y puntuacion
    #Se obtiene el centro (cx,cy) del candidato con cv2.moments
    #Mide que tan lejos esta el candidato respecto al centro de la ROI
    
    candidatos = []
    for c in contornos:
        area = cv2.contourArea(c)
        perimetro = cv2.arcLength(c, True)
        if area > 30 and area < area_max_permitida and perimetro > 0:
            circularidad = 4 * np.pi * (area / (perimetro ** 2))
            
            # Centroide del contorno candidato
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
                
                # Distancia normalizada al centro de la ROI
                dist_centro = np.sqrt((cx - centro_roi[0])**2 + (cy - centro_roi[1])**2)
                dist_norm = dist_centro / np.sqrt(w**2 + h**2)
                
                # Puntuación: favorece alta circularidad y posición cercana al centro
                score = circularidad / (1.0 + 3.0 * dist_norm)
                candidatos.append((c, area, score))
                
    if candidatos:
        #Selecciona el contorno candidato con mejor puntuación combinada
        c_max = max(candidatos, key=lambda x: x[2])[0]
    else:
        #Resguardo en caso de no superar los filtros estrictos
        c_max = max(contornos, key=cv2.contourArea)

    area = cv2.contourArea(c_max)

#Si el contorno tiene 5 o mas puntos, le ajusta una elipse, la dibuja sobre la imagen y calcula el diametro promedio
    if len(c_max) >= 5:
        elipse = cv2.fitEllipse(c_max)
        cv2.ellipse(roi_dibujada, elipse, (0, 255, 0), 2)
        (cx, cy), (eje_menor, eje_mayor), angulo = elipse
        diametro_promedio = (eje_menor + eje_mayor) / 2.0
        return area, diametro_promedio, roi_dibujada

    diametro_equiv = 2.0 * np.sqrt(area / np.pi)
    return area, diametro_equiv, roi_dibujada


class NodoHuffman:
    #Define la estructura basica de datos para construir el Arbol de Huffman utilizado 
    #en el algoritmo de compresion sin perdida
    def __init__(self, simbolo, frecuencia): #define los atributos que tiene cada nodo del arbol
        self.simbolo = simbolo #guarda el valor del pixel. Para los nodos internos creados al fusionar ramas, el valor es none
        self.frecuencia = frecuencia #representa la cantidad de veces que aparece ese pixel
        self.izq = None #Punteros a los hijos izquierdo y derecho del nodo dentro del arbol binario
        self.der = None

    def __lt__(self, otro): #sobreescribe el operador "menor que" para poder comparar nodos entre si
        return self.frecuencia < otro.frecuencia #establece que un nodo es "menor" si su frecuencia de aparicion es menor
#Es importante para el funcionamiento de heapq, que necesita saber como ordenar los nodos automaticamente para extraer siempre
#primero los dos noso con la frecuencia mas baja y combinarlos en la construccion del arbol

def construir_arbol_huffman(simbolos_flat): #toma un arreglo plano de datos y genera la tabla de codigos binarios de longitud variable del algoritmo de Huffman
    #Asigna cadenas de bits mas cortas a los pixeles que se repiten con frecuencia y cadenas mas largas a los pixeles raros
    freq_map = Counter(simbolos_flat) #cuenta cuantas veces aparece cada valor de pixel en la imagen
    if not freq_map:
        return {}
    
    heap = [NodoHuffman(sym, freq) for sym, freq in freq_map.items()] #Convierte cada (simbolo,frecuencia) en un objeto de la clase NodoHuffman
    heapq.heapify(heap) #transforma la lista en una cola de prioridad. El nodo con menor frecuencia queda en la raiz

    if len(heap) == 1: #Si la imagen es completamente uniforme, asigna directamene '0' a ese unico simbolo y finaliza
        node = heapq.heappop(heap)
        return {node.simbolo: '0'}

    while len(heap) > 1: #Contruye el arbol desde las hojas hacia la raiz
        #Extrae los dos nodos con menor frecuencia
        n1 = heapq.heappop(heap)
        n2 = heapq.heappop(heap)
        
        #Crea al nodo padre sin simbolo cuya frecuencia es la suma de las frecuencias de ambos hijos
        nodo_padre = NodoHuffman(None, n1.frecuencia + n2.frecuencia)
        #Asigna n1 y n2
        nodo_padre.izq = n1
        nodo_padre.der = n2
        #Inserta el nodo padre de vuelta en el heap
        heapq.heappush(heap, nodo_padre)
   #se repite el ciclo hasta que quede un solo nodo en el heap el cual representa la raiz del arbol completo
    
    codigos = {}
    def recorrer_arbol(nodo, codigo_actual): 
        #funcion recursiva que navega el arbol desde la raiz hacia las hojas
        if nodo is None:
            return
        if nodo.simbolo is not None:
            codigos[nodo.simbolo] = codigo_actual
            return
        recorrer_arbol(nodo.izq, codigo_actual + '0')
        recorrer_arbol(nodo.der, codigo_actual + '1')

    recorrer_arbol(heap[0], '') #inicia el recorrido recursivo desde el nodo raiz con una cadena vacia ''
    return codigos #devuelve un diccionaria donde las claves son los valores de pixel y los valores son sus codigos de Huffman


def evaluar_huffman(imagen):
    #Evalua el rendimiento teorico y practico de la codifiacion de Huffman sobre una imagen
    #Calcula la entropia de Shannon, la tasa de compresion y la longitud media de palabra
    
    flat = imagen.ravel() #Aplana la matriz bidimensional de la imagen en un vector unidimensional
    orig_bytes = len(flat) #obtiene la cantidad total de pixeles
    
    codigos = construir_arbol_huffman(flat) #obtiene el diccionario de codigos de Huffman
    total_bits = sum(len(codigos[val]) for val in flat) #Suma la longitud en bits del codigo asignado a cada uno de los pixeles de la imagen
    comp_bytes = (total_bits + 7) // 8 #convierte el total de bits comprimidos a bytes redondeando hacia arriba
    
    #Cálculo de la tasa de compresion (cuantas veces se redujo el tamaño de la imagen respecto al original)
    cr = orig_bytes / comp_bytes if comp_bytes > 0 else 1.0
    
    #Cálculo de la entropía de Shannon
    counts = np.bincount(flat) #Genera el histograma contando cuantos pixeles existen de cada intensidad
    probs = counts[counts > 0] / orig_bytes #Calcula la probabilidad de cada intensidad presente en la imagen
    entropia = -np.sum(probs * np.log2(probs)) #Aplica la formula de Shannon en bits/pixel
    
    #Cálculo de longitud media
    longitud_media = total_bits / orig_bytes #Determina el promedio real de bits utilizado por cada pixel comprimido
    
    return {
        'orig_bytes': orig_bytes,
        'comp_bytes': comp_bytes,
        'cr': cr,
        'entropia': entropia,
        'longitud_media': longitud_media
    }


def evaluar_rle(imagen, direccion='horizontal'):
    #Evalua el rendimiento del algoritmo de compresion sin perdida Run-Length Encoding
    #Permite comparar el recorrido horizontal frente al vertical
    #Agrupa secuencias consecutivas de pixeles con el mismo valor y las guarda en forma de pares
    
    if direccion == 'horizontal': #Aplana la matriz recorriendola fila por fila
        flat = imagen.ravel()
    elif direccion == 'vertical': #Transpone la matriz y la plana (equivalente a hacer un barrido vertical)
        flat = imagen.T.ravel()
    else:
        raise ValueError("Dirección debe ser 'horizontal' o 'vertical'")
        
    if len(flat) == 0: #En vez de usar bucles for, usa funciones vectorizadas de numpy para contar las corridas en milisegundos
        return 0, 0, 0, 1.0

    diferencias = np.diff(flat) #resta cada pixel con el siguiente
    cambios = np.where(diferencias != 0)[0] #encuentra los indices exactos donde cambia la intensidad
    
    inicios = np.concatenate(([0], cambios + 1)) #obtiene los indices de inicio de cada bloque continuo
    valores = flat[inicios] #extrae la intensidad de cada secuencia de pixeles
    
    num_runs = len(valores) #cuenta la cantidad de corridas generadas
    orig_bytes = flat.size #tamaño original
    comp_bytes = num_runs * 3  # (1 Byte valor + 2 Bytes contador)
    cr = orig_bytes / comp_bytes if comp_bytes > 0 else 1.0 #calcula la tasa de compresion
    
    return num_runs, orig_bytes, comp_bytes, cr


def evaluar_jpeg(roi_rgb, factores_q=[90, 50, 10, 5]):
    #Simula y evalua el comportamiento de la compresion con perdida JPEG sobre una imagen a color variando el factor de calidad Q
    #Calcula tanto la tasa de compresion como la degradacion de la calidad visual (PSNR y SSIM)
    
    resultados = [] #lista para almacenar los diccionarios con las metricas obtenidas para cada factor Q
    orig_bytes = roi_rgb.size #Obtiene el tamaño total de la imagen original en bytes (Alto x Ancho x 3 canales)
    
    for q in factores_q:
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), q] #configura el nivel de calidad que usara el compresor de OpenCV
        roi_bgr = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2BGR) #convierte la imagen de RGB a BGR
        _, enc_img = cv2.imencode('.jpg', roi_bgr, encode_param) #Aplica la compresion JPEG directamente en momeoria RAM y devuelve un vector de bytes comprimido
        comp_bytes = len(enc_img) #mide la longitud en bytes del archivo JPEG resultante
        
        #Reconstruccion de la imagen
        dec_bgr = cv2.imdecode(enc_img, cv2.IMREAD_COLOR)
        dec_rgb = cv2.cvtColor(dec_bgr, cv2.COLOR_BGR2RGB)
        
        #Calculo de metricas (PSNR, SSIM y CR)
        psnr_val = psnr_sk(roi_rgb, dec_rgb, data_range=255) #calcula la relacion señal a ruido de pico en dB
        ssim_val = ssim_sk(roi_rgb, dec_rgb, channel_axis=2, data_range=255) #evalua el indice de similitud estructural considerando los 3 canales de color
        cr = orig_bytes / comp_bytes if comp_bytes > 0 else 1.0
        
        resultados.append({
            'q': q,
            'img_dec': dec_rgb,
            'comp_bytes': comp_bytes,
            'cr': cr,
            'psnr': psnr_val,
            'ssim': ssim_val
        }) #Guarda en un diccionario el factor Q, la imagen decodificada, los bytes comprimidos y las metricas calculadas
        
    return resultados, orig_bytes #devuelve la coleccion completa junto con el peso original en bytes


def evaluar_impacto_diametro_pupilar(roi_rgb):
    #Cuantifica el impacto directo que causa la compresion con perdida JPEG sobre la medicion clinica del diametro y area de la pupila
    #Compara los resultados geometricos obtenidos sobre la imagen original sin comprimir frente a los resultados obtenidos tras comprimirla y decodificarla a distintos Q
    
    #Ejecuta la funcion de segmentacion y ajuste eliptico sobre la ROI original en color verdadero
    area_orig, diam_orig, vis_orig = estimar_geometria_pupila(roi_rgb)
    
    factores = [50, 10] #Define los dos niveles de calidad JPEG a evaluar
    resultados = []
    
    #Compresion, decodificacion y re-segmentacion
    for q in factores:
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), q]
        roi_bgr = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2BGR)
        _, enc_img = cv2.imencode('.jpg', roi_bgr, encode_param)
        dec_bgr = cv2.imdecode(enc_img, cv2.IMREAD_COLOR)
        dec_rgb = cv2.cvtColor(dec_bgr, cv2.COLOR_BGR2RGB)
        
        area_q, diam_q, vis_q = estimar_geometria_pupila(dec_rgb)
        
        #Cálculo de error porcentual relativo
        err_diam = (abs(diam_q - diam_orig) / diam_orig * 100.0) if diam_orig > 0 else 0.0
        err_area = (abs(area_q - area_orig) / area_orig * 100.0) if area_orig > 0 else 0.0
        
        resultados.append({
            'q': q,
            'area': area_q,
            'diametro': diam_q,
            'err_diam': err_diam,
            'err_area': err_area,
            'vis': vis_q
        }) #Guarda en un diccionario el valor de Q, el area y diametro re-calculados, sus respectivos
        #errores porcentuales y la imagen con la elipse dibujada
        
        
    return (area_orig, diam_orig, vis_orig), resultados



# FUNCIÓN PRINCIPAL DE EJECUCIÓN DEL MÓDULO 4
def ejecutar_modulo_4(config):
    print("\n==================================================")
    print("      EJECUTANDO MÓDULO 4: COMPRESIÓN DE IMAGEN   ")
    print("==================================================")

    video_path = config['video_path']
    roi = config['roi']
    frame_idx = config.get('frame_idx', 0)

    frame_rgb, roi_rgb = cargar_frame_y_roi(video_path, num_frame=frame_idx, roi_coords=roi)
    frame_gris = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    roi_gris = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2GRAY)
    roi_binaria = segmentar_roi_pupila(roi_gris)

#4.1 HUFFMAN
    #Ejecuta la funcion evaluar Huffman sobre tres niveles distintos de entropia para comparar el impacto en la compresion
    res_h_frame = evaluar_huffman(frame_gris)
    res_h_roi_g = evaluar_huffman(roi_gris)
    res_h_roi_b = evaluar_huffman(roi_binaria)

    #Imprime una tabla de texto formateada con alineacion de columnas que muestra los resultados en la terminal
    print("\n 4.1 Evaluación de Codificación Huffman")
    print("-" * 65)
    print(f"{'Imagen / Región':<22} | {'Orig (B)':<9} | {'Comp (B)':<9} | {'CR':<6} | {'H(X)':<6} | {'L_med':<6}")
    print("-" * 65)
    print(f"{'Frame Completo (Gris)':<22} | {res_h_frame['orig_bytes']:<9} | {res_h_frame['comp_bytes']:<9} | {res_h_frame['cr']:<6.2f} | {res_h_frame['entropia']:<6.2f} | {res_h_frame['longitud_media']:<6.2f}")
    print(f"{'ROI Ojo (Gris)':<22} | {res_h_roi_g['orig_bytes']:<9} | {res_h_roi_g['comp_bytes']:<9} | {res_h_roi_g['cr']:<6.2f} | {res_h_roi_g['entropia']:<6.2f} | {res_h_roi_g['longitud_media']:<6.2f}")
    print(f"{'ROI Segmentada (Binaria)':<22} | {res_h_roi_b['orig_bytes']:<9} | {res_h_roi_b['comp_bytes']:<9} | {res_h_roi_b['cr']:<6.2f} | {res_h_roi_b['entropia']:<6.2f} | {res_h_roi_b['longitud_media']:<6.2f}")
    print("-" * 65)

    #Graficos y representacion visual
    fig_huff, ax_h = plt.subplots(1, 3, figsize=(14, 4.5))
    fig_huff.canvas.manager.set_window_title("4.1 Codificación Huffman: Entropía y Compresión")

    etiquetas = ['Frame Gris', 'ROI Gris', 'ROI Binaria']
    valores_cr = [res_h_frame['cr'], res_h_roi_g['cr'], res_h_roi_b['cr']]
    valores_ent = [res_h_frame['entropia'], res_h_roi_g['entropia'], res_h_roi_b['entropia']]

    #Subgrafico 1: Tasa de compresion
    ax_h[0].bar(etiquetas, valores_cr, color=['#2b5c8f', '#3690c0', '#67a9cf'])
    ax_h[0].set_title("Tasa de Compresión (CR)")
    ax_h[0].set_ylabel("CR (Veces comprimido)")

    #Subgrafico 2: Entropia de Shannon
    ax_h[1].bar(etiquetas, valores_ent, color=['#d73027', '#fc8d59', '#fee090'])
    ax_h[1].set_title("Entropía de Shannon H(X)")
    ax_h[1].set_ylabel("Bits / Píxel")

    #Subgrafico 3: Visualizacion de la ROI segmentada
    ax_h[2].imshow(roi_binaria, cmap='gray')
    ax_h[2].set_title("ROI Segmentada Binaria")
    ax_h[2].axis('off')

    plt.tight_layout()
    plt.show()

#4.2 Run-Length Encoding (RLE)
    #Llama a la funcion de evaluar RLE comparando el rendimiento del barrido por filas frente al barrido por columnas
    #Muestra una tabla comparativa en la terminal y genera un grafico en matplotlib
    
    runs_rh_h, orig_rh, comp_rh_h, cr_rh_h = evaluar_rle(roi_gris, 'horizontal')
    runs_rh_v, _,       comp_rh_v, cr_rh_v = evaluar_rle(roi_gris, 'vertical')

    runs_rb_h, orig_rb, comp_rb_h, cr_rb_h = evaluar_rle(roi_binaria, 'horizontal')
    runs_rb_v, _,       comp_rb_v, cr_rb_v = evaluar_rle(roi_binaria, 'vertical')

    #Formateo de la tabla en la terminal
    print("\n 4.2 Evaluación de Run-Length Encoding (RLE)")
    print("-" * 75)
    print(f"{'Región / Recorrido':<28} | {'Runs':<7} | {'Orig (B)':<9} | {'Comp (B)':<9} | {'CR':<6}")
    print("-" * 75)
    print(f"{'ROI Gris - Horizontal':<28} | {runs_rh_h:<7} | {orig_rh:<9} | {comp_rh_h:<9} | {cr_rh_h:<6.2f}")
    print(f"{'ROI Gris - Vertical':<28} | {runs_rh_v:<7} | {orig_rh:<9} | {comp_rh_v:<9} | {cr_rh_v:<6.2f}")
    print(f"{'ROI Segmentada - Horizontal':<28} | {runs_rb_h:<7} | {orig_rb:<9} | {comp_rb_h:<9} | {cr_rb_h:<6.2f}")
    print(f"{'ROI Segmentada - Vertical':<28} | {runs_rb_v:<7} | {orig_rb:<9} | {comp_rb_v:<9} | {cr_rb_v:<6.2f}")
    print("-" * 75)

    fig_rle, ax_r = plt.subplots(1, 2, figsize=(12, 5))
    fig_rle.canvas.manager.set_window_title("4.2 RLE: Recorrido Horizontal vs Vertical")

    x = np.arange(2)
    ancho_barra = 0.35

    cr_horiz = [cr_rh_h, cr_rb_h]
    cr_vert = [cr_rh_v, cr_rb_v]

    #Subgrafico 1: Grafico de barras agrupadas (grafico comparativo)
    ax_r[0].bar(x - ancho_barra/2, cr_horiz, ancho_barra, label='Horizontal (Filas)', color='navy')
    ax_r[0].bar(x + ancho_barra/2, cr_vert, ancho_barra, label='Vertical (Columnas)', color='teal')
    ax_r[0].set_xticks(x)
    ax_r[0].set_xticklabels(['ROI Gris', 'ROI Segmentada'])
    ax_r[0].set_title("Comparativa de CR en RLE")
    ax_r[0].set_ylabel("Tasa de Compresión (CR)")
    ax_r[0].legend()
    ax_r[0].grid(True, linestyle='--', alpha=0.5)

    #Subgrafico 2: Mascara y valores obtenidos
    ax_r[1].imshow(roi_binaria, cmap='gray')
    ax_r[1].set_title(f"ROI Segmentada Binaria\nCR Horiz: {cr_rb_h:.1f}x | Vert: {cr_rb_v:.1f}x")
    ax_r[1].axis('off')

    plt.tight_layout()
    plt.show()

#4.3 JPEG
    #Hace la evaluacion y visualizacion de la compresion JPEG con perdida, comparando la 
    #imagen original frente a las versiones comprimidas a diferentes factores de calidad (Q)
    
    #Genera una figura en Matplotlib compuesta por 5 subgraficos dispuestos horizontalmente para observar la degradacion visual,
    #la reduccion en espacio de almacenamiento y la perdida de fidelidad metrica (PSNR y SSIM)
        
    print("\n 4.3 Evaluación de Compresión JPEG (Q = 90, 50, 10, 5)")
    
    #Define la lista con los factores de calidad a evaluar y llama a la funcion de compresion
    #Devuelve la lista con las imagenes decodificadas y sus metricas para cada Q, junto con el peso original en bytes
    factores_q = [90, 50, 10, 5]
    res_jpeg, orig_bytes_jpeg = evaluar_jpeg(roi_rgb, factores_q)

    #Configuracion de la figura
    fig_jpeg, ax_j = plt.subplots(1, 5, figsize=(18, 4.5))
    fig_jpeg.canvas.manager.set_window_title("4.3 Compresión JPEG: Degradación Visual por Factor Q")

    
    ax_j[0].imshow(roi_rgb)
    ax_j[0].set_title(f"Original (RGB)\n{orig_bytes_jpeg / 1024:.1f} KB")
    ax_j[0].axis('off')

    #Bucle de renderizado para cada factor Q
    for idx, res in enumerate(res_jpeg):
        q = res['q']
        #Extraccion de metricas
        comp_kb = res['comp_bytes'] / 1024.0 
        cr = res['cr']
        psnr_val = res['psnr']
        ssim_val = res['ssim']

        ax_j[idx + 1].imshow(res['img_dec'])
        ax_j[idx + 1].set_title(
            f"JPEG Q={q}\n{comp_kb:.1f} KB | CR: {cr:.1f}x\nPSNR: {psnr_val:.1f} dB | SSIM: {ssim_val:.3f}"
        )
        ax_j[idx + 1].axis('off')

    plt.tight_layout()
    plt.show()

#4.4 IMPACTO EN DIÁMETRO PUPILAR
    #Hace la visualizacion grafica del impacto biomedico de la compresion JPEG sobre la estimacion de los parametros geometricos de la pupila
    #Genera una figura con 3 subgraficos para comparar visualmente la imagen original sin comprimir frente a las imagenes comprimidas, destacando la elipse ajustada y los errores porcentuales
    
    #Llama a la funcion de evaluacion
    #Extrae las metricas y la imagen anotada de referencia de la ROI sin comprimir, y la lista con las imagenes decodificadas y los calculos de error para cada Q
    (area_orig, diam_orig, vis_orig), res_impacto = evaluar_impacto_diametro_pupilar(roi_rgb)

    fig_imp, ax_i = plt.subplots(1, 3, figsize=(15, 5))
    fig_imp.canvas.manager.set_window_title("4.4 Impacto de JPEG en la Estimación Biomédica (Pupila)")

    ax_i[0].imshow(vis_orig)
    ax_i[0].set_title(f"Original\nDiámetro: {diam_orig:.2f} px | Área: {area_orig:.1f} px²")
    ax_i[0].axis('off')

    #Bucle de renderizado de los resultados comprimidos
    for idx, r in enumerate(res_impacto):
        q = r['q'] #obtiene el factor de calidad actual
        ax_i[idx + 1].imshow(r['vis']) #dibuja la imagen reconstruida con la elipse verde re-estimada en el subgrafico correspondiente
        ax_i[idx + 1].set_title( #Configura el titulo multilinea
            f"JPEG Q={q}\nDiámetro: {r['diametro']:.2f} px (Error: {r['err_diam']:.2f}%)\nÁrea: {r['area']:.1f} px² (Error: {r['err_area']:.2f}%)"
        )
        ax_i[idx + 1].axis('off')

    plt.tight_layout()
    plt.show()

    # TABLA CONSOLIDADA FINAL
    print("\n" + "=" * 75)
    print("TABLA COMPARATIVA CONSOLIDADA DE TODOS LOS MÉTODOS DE COMPRESIÓN")
    print("=" * 75)
    print(f" {'Método / Configuración':<30} | {'CR':<8} | {'PSNR (dB)':<12} | {'SSIM':<8}")
    print("-" * 75)
    print(f" {'Huffman (ROI Gris)':<30} | {res_h_roi_g['cr']:<8.2f} | {'Infinito':<12} | {'1.0000':<8}")
    print(f" {'Huffman (ROI Binaria)':<30} | {res_h_roi_b['cr']:<8.2f} | {'Infinito':<12} | {'1.0000':<8}")
    print(f" {'RLE Horiz. (ROI Gris)':<30} | {cr_rh_h:<8.2f} | {'Infinito':<12} | {'1.0000':<8}")
    print(f" {'RLE Vert. (ROI Gris)':<30} | {cr_rh_v:<8.2f} | {'Infinito':<12} | {'1.0000':<8}")
    print(f" {'RLE Horiz. (ROI Binaria)':<30} | {cr_rb_h:<8.2f} | {'Infinito':<12} | {'1.0000':<8}")
    print(f" {'RLE Vert. (ROI Binaria)':<30} | {cr_rb_v:<8.2f} | {'Infinito':<12} | {'1.0000':<8}")
    for j in res_jpeg:
        print(f" {'JPEG (Q = ' + str(j['q']) + ')':<30} | {j['cr']:<8.2f} | {j['psnr']:<12.2f} | {j['ssim']:<8.4f}")
    print("=" * 75)

    # DISCUSIÓN TEÓRICA
    print("\n" + "=" * 75)
    print("DISCUSIÓN Y ANÁLISIS DE RESULTADOS")
    print("=" * 75)
    
    print("\n1. EMERGENCIA DE ARTEFACTOS DE BLOQUE (BLOCKING ARTIFACTS):")
    print("   • Los artefactos de bloque derivados de la cuantización DCT (retícula de 8x8 px)")
    print("     empiezan a percibirse visualmente a partir de Q = 10 y resultan dominantes en Q = 5.")

    print("\n2. BORDES NÍTIDOS VS. REGIONES UNIFORMES:")
    print("   • Regiones Uniformes (Piel / Esclerótica / Iris): El efecto de bloque es MÁS NOTORIO")
    print("     a la vista humana debido a la falta de textura, generando discontinuidades en grilla.")
    print("   • Bordes Nítidos (Límite Pupila-Iris): Sufren el fenómeno de RINGING (Oscilaciones de Gibbs).")

    print("\n3. IMPACTO EN EL DIÁMETRO PUPILAR ESTIMADO (MÓDULO 3):")
    print(f"   • Original : Diámetro = {diam_orig:.2f} px | Área = {area_orig:.1f} px²")
    for r in res_impacto:
        print(f"   • JPEG Q={r['q']:<2}: Diámetro = {r['diametro']:.2f} px (Error: {r['err_diam']:.2f}%) | Área = {r['area']:.1f} px² (Error: {r['err_area']:.2f}%)")
    
    print("\n   [CONCLUSIÓN BIOMÉDICA]:")
    print("   • Compresión Q=50: Mantiene el error bajo (<1%). Es óptima para almacenamiento clínico.")
    print("   • Compresión Q=10: Deforma el contorno de la pupila e introduce errores de medición.")
    print("=" * 75 + "\n")

    sys.stdout.flush()
    print("[ÉXITO] Módulo 4 completado.")