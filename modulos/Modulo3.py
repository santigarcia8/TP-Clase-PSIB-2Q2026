#Modulo 3: Extraccion de caracteristicas, analisis de textura y dinamica temporal
#Modulo 3.1: Descriptores morfometricos y de textura en el frame segmentado
#Modulo 3.2: Procesameinto en video y curva temporal fisiologica
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew
from skimage.feature import graycomatrix, graycoprops

NIVELES_GRIS = 64 #cantidad de niveles a los que cuantizamos la ROI (la guia pide 32 o 64)


# =============================================================================
# FUNCIONES AUXILIARES: LECTURA DEL FRAME Y CUANTIZACION DE LA ROI
# =============================================================================

def cargar_frame_y_roi(video_path, num_frame=0, roi_coords=None):
    cap = cv2.VideoCapture(video_path) #abre el video desde la ruta elegida en el main
    cap.set(cv2.CAP_PROP_POS_FRAMES, num_frame) #se para en el frame elegido
    ok, frame_bgr = cap.read()
    cap.release()
    if not ok or frame_bgr is None:
        raise ValueError(f"No se pudo leer el frame {num_frame} del video.")

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB) #OpenCV lee en BGR, lo pasamos a RGB

    if roi_coords is not None:
        ymin, ymax, xmin, xmax = roi_coords
        roi_rgb = frame_rgb[ymin:ymax, xmin:xmax]
    else:
        roi_rgb = frame_rgb
    return frame_rgb, roi_rgb


def cuantizar(imagen, niveles=NIVELES_GRIS):
    #Pasa una imagen de 8 bits (256 niveles, 0-255) a 'niveles' niveles de gris (0 a niveles-1)
    return (imagen.astype(np.uint16) * niveles // 256).astype(np.uint8)


# =============================================================================
# PARTE 3.1 - PASO PREVIO: SEGMENTACION DE LA PUPILA
# =============================================================================
#La consigna pide trabajar "a partir de la ROI segmentada". Como el Modulo 2 grafica cada metodo y no devuelve
#la mascara, armamos aca una version sin graficos que tambien vamos a reutilizar en la parte 3.2 (todos los frames).
#Idea: la pupila es la region MAS OSCURA, REDONDA, con MUCHO CONTRASTE contra el iris y con un BORDE NITIDO.
#Las pestañas y las sombras de la piel tambien son oscuras, pero son finas, alargadas o tocan el borde de la ROI.

def puntuar_mancha(contorno, gris, gradiente):
    #Le da un puntaje a una mancha oscura segun cuanto se parece a una pupila. Devuelve None si se descarta
    h, w = gris.shape
    area = cv2.contourArea(contorno)
    if area < 0.0005 * h * w or area > 0.25 * h * w: #muy chica (ruido) o muy grande (sombra, ceja)
        return None

    envolvente = cv2.convexHull(contorno)
    area_env = cv2.contourArea(envolvente)
    circularidad = 4 * np.pi * area_env / cv2.arcLength(envolvente, True) ** 2  #1 = circulo
    solidez = area / area_env                              #que tan "maciza" es (las pestañas tienen huecos)
    (x, y), r = cv2.minEnclosingCircle(envolvente)
    llenado = area_env / (np.pi * r ** 2)                  #que tanto llena su circulo envolvente (lo alargado llena poco)

    adentro = np.zeros((h, w), np.uint8)
    cv2.drawContours(adentro, [envolvente], -1, 255, thickness=cv2.FILLED)
    anillo = np.zeros((h, w), np.uint8)
    cv2.circle(anillo, (int(x), int(y)), int(1.8 * r), 255, -1)
    anillo[adentro > 0] = 0
    if not anillo.any():
        return None
    contraste = max(gris[anillo > 0].mean() - gris[adentro > 0].mean(), 0) #brillo alrededor - brillo adentro

    borde = cv2.findContours(adentro, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0][0][:, 0, :]
    nitidez = gradiente[borde[:, 1], borde[:, 0]].mean() #gradiente medio sobre el borde: maximo en el borde real de la pupila

    bx, by, bw, bh = cv2.boundingRect(envolvente)
    toca_borde = bx <= 1 or by <= 1 or bx + bw >= w - 1 or by + bh >= h - 1 #sombras y pestañas suelen tocar el borde

    puntaje = circularidad ** 2 * llenado ** 2 * solidez * contraste * nitidez
    if toca_borde:
        puntaje *= 0.3
    return {'puntaje': puntaje, 'envolvente': envolvente, 'centro': (float(x), float(y)), 'area': area_env}


def elegir_pupila(candidatas, tolerancia=0.6):
    #Elige la mancha de mayor puntaje. Pero los distintos umbrales generan manchas "anidadas" (una adentro de otra)
    #con puntajes parecidos, y las mas grandes se meten en el anillo oscuro del iris. Entonces, entre las anidadas
    #con puntaje parecido al mejor (>= 60%), nos quedamos con la MAS CHICA, que es la que sigue el borde negro real.
    #Esto ademas evita que en el video la medicion salte de una mancha a otra entre frames
    if not candidatas:
        return None
    mejor = max(candidatas, key=lambda c: c['puntaje'])
    anidadas = [c for c in candidatas
                if c['puntaje'] >= tolerancia * mejor['puntaje']
                and 0.4 * mejor['area'] <= c['area'] <= mejor['area']
                and cv2.pointPolygonTest(mejor['envolvente'], c['centro'], False) >= 0]
    return min(anidadas, key=lambda c: c['area'])


def segmentar_pupila(roi_rgb):
    #Devuelve la mascara de la pupila (0/255) y su contorno. Si no la encuentra devuelve (mascara vacia, None)
    gris = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2GRAY)
    suave = cv2.medianBlur(gris, 5) #la mediana saca ruido impulsivo sin borronear el borde de la pupila
    h, w = gris.shape
    gradiente = np.hypot(cv2.Sobel(suave, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(suave, cv2.CV_32F, 0, 1, ksize=3))

    k = max(3, int(min(h, w) / 80)) #el tamaño del kernel depende del tamaño de la ROI
    k = k if k % 2 == 1 else k + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    def candidatas_de(binaria):
        binaria = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, kernel)   #apertura: borra pestañas finas y puntitos
        binaria = cv2.morphologyEx(binaria, cv2.MORPH_CLOSE, kernel)  #cierre: tapa huecos chicos (reflejos)
        contornos, _ = cv2.findContours(binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        puntuadas = [puntuar_mancha(c, suave, gradiente) for c in contornos]
        return [p for p in puntuadas if p is not None]

    #Paso 1: umbralizacion multinivel. No sabemos de antemano que tan oscura ni que tan grande es la pupila,
    #asi que probamos varios umbrales (el 1%, 2%, ..., 25% de pixeles mas oscuros) y juntamos todas las manchas
    candidatas = []
    for porcentaje in [1, 2, 3, 5, 8, 12, 16, 20, 25]:
        umbral = np.percentile(suave, porcentaje)
        candidatas += candidatas_de((suave <= umbral).astype(np.uint8) * 255)
    elegida = elegir_pupila(candidatas)
    if elegida is None:
        return np.zeros((h, w), np.uint8), None

    #Paso 2: si el iris es oscuro y se ve entero, la mancha elegida puede ser iris + pupila.
    #Aplicamos Otsu SOLO adentro de esa mancha y bajamos un nivel si adentro hay una mancha mas oscura y redonda.
    #No bajamos si las dos clases tienen brillos parecidos (es solo textura) o si lo oscuro es casi toda
    #la mancha (eso pasa cuando lo unico que se separa es un reflejo)
    for _ in range(3):
        region = np.zeros((h, w), np.uint8)
        cv2.drawContours(region, [elegida['envolvente']], -1, 255, thickness=cv2.FILLED)
        adentro = suave[region > 0]
        u2, _ = cv2.threshold(adentro.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        oscuros, claros = adentro[adentro <= u2], adentro[adentro > u2]
        if oscuros.size == 0 or claros.size == 0:
            break
        if claros.mean() - oscuros.mean() < 20 or oscuros.size / adentro.size > 0.85:
            break
        sub = elegir_pupila(candidatas_de(((suave <= u2) & (region > 0)).astype(np.uint8) * 255))
        if sub is None:
            break
        circ_sub = 4 * np.pi * sub['area'] / cv2.arcLength(sub['envolvente'], True) ** 2
        if circ_sub < 0.6 or sub['area'] < 0.5 * oscuros.size:
            break
        elegida = sub

    #Paso 3: la mascara final es la envolvente convexa de la mancha elegida. Como la pupila es convexa,
    #asi se rellenan los "mordiscos" que dejan los reflejos sobre el borde y los reflejos que quedan adentro
    mascara = np.zeros((h, w), np.uint8)
    cv2.drawContours(mascara, [elegida['envolvente']], -1, 255, thickness=cv2.FILLED)
    return mascara, elegida['envolvente']


# =============================================================================
# PARTE 3.1 - CONSIGNA 1: DESCRIPTORES MORFOMETRICOS (FORMA)
# =============================================================================

# ---- Consigna 1.a: Area (A) en pixeles y Perimetro (P) ----------------------
def calcular_area_perimetro(mascara, contorno):
    area = float(np.count_nonzero(mascara)) #area = cantidad de pixeles blancos de la mascara
    perimetro = cv2.arcLength(contorno, True) #largo del contorno cerrado de la pupila
    return area, perimetro


# ---- Consigna 1.b: Coordenadas del Centroide (Xc, Yc) -----------------------
def calcular_centroide(mascara):
    M = cv2.moments(mascara, binaryImage=True)
    xc = M["m10"] / M["m00"] #centroide = momento de orden 1 / momento de orden 0 (que es el area)
    yc = M["m01"] / M["m00"]
    return xc, yc


# ---- Consigna 1.c: Circularidad y Excentricidad -----------------------------
def calcular_circularidad_excentricidad(mascara, contorno):
    #Circularidad = 4*pi*A/P^2: vale 1 para un circulo perfecto y baja cuanto mas deformada esta la pupila.
    #Usamos el area del poligono del contorno (no la cuenta de pixeles) para que area y perimetro se midan
    #sobre la misma curva; si no, por la discretizacion en pixeles puede dar apenas mayor a 1
    perimetro = cv2.arcLength(contorno, True)
    circularidad = 4 * np.pi * cv2.contourArea(contorno) / perimetro ** 2

    #Excentricidad a partir de los momentos centrales de orden 2 (es la matriz de covarianza de la mancha).
    #Sus autovalores son proporcionales al cuadrado de los semiejes de la elipse equivalente
    M = cv2.moments(mascara, binaryImage=True)
    cov = np.array([[M["mu20"], M["mu11"]], [M["mu11"], M["mu02"]]]) / M["m00"]
    l_menor, l_mayor = np.sort(np.linalg.eigvalsh(cov))
    excentricidad = np.sqrt(1 - l_menor / l_mayor) if l_mayor > 0 else 0.0 #0 = circulo, cerca de 1 = elipse muy aplastada
    return circularidad, excentricidad


def descriptores_morfometricos(mascara, contorno):
    #Junta los tres items de la Consigna 1 en un solo diccionario
    area, perimetro = calcular_area_perimetro(mascara, contorno)
    xc, yc = calcular_centroide(mascara)
    circularidad, excentricidad = calcular_circularidad_excentricidad(mascara, contorno)
    return {
        'area_px': area,
        'perimetro_px': perimetro,
        'centroide_x': xc,
        'centroide_y': yc,
        'circularidad': circularidad,
        'excentricidad': excentricidad,
        'diametro_equivalente_px': 2 * np.sqrt(area / np.pi), #diametro de un circulo con la misma area (para la curva D(t) de 3.2)
    }


def graficar_morfometria(roi_rgb, roi_cuant, mascara, contorno, morfo):
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    fig.canvas.manager.set_window_title("Parte 3.1 - Consigna 1: Descriptores morfométricos")

    ax[0].imshow(roi_cuant, cmap='gray', vmin=0, vmax=NIVELES_GRIS - 1)
    ax[0].set_title(f"ROI cuantizada a {NIVELES_GRIS} niveles")

    ax[1].imshow(mascara, cmap='gray')
    ax[1].set_title(f"1.a) Máscara de la pupila\nA = {morfo['area_px']:.0f} px   P = {morfo['perimetro_px']:.1f} px")

    ax[2].imshow(roi_rgb)
    ax[2].plot(contorno[:, 0, 0], contorno[:, 0, 1], '-', color='lime', lw=1.5, label='Contorno')
    ax[2].plot(morfo['centroide_x'], morfo['centroide_y'], '+r', ms=14, mew=2,
               label=f"1.b) Centroide ({morfo['centroide_x']:.1f}, {morfo['centroide_y']:.1f})")
    ax[2].set_title(f"1.c) Circularidad = {morfo['circularidad']:.3f}\nExcentricidad = {morfo['excentricidad']:.3f}")
    ax[2].legend(loc='lower right', fontsize=8)

    for a in ax:
        a.axis('off')
    plt.tight_layout()
    plt.show()


# =============================================================================
# PARTE 3.1 - CONSIGNA 2: TEXTURA DE PRIMER ORDEN (ESTADISTICA GLOBAL)
# =============================================================================
#La consigna pide calcular Media, Desvio Estandar, Entropia y Asimetria (Skewness) sobre la ROI de la
#pupila/iris en los tres canales de color (R, G, B) por separado, usando la ROI cuantizada (64 niveles).
#Calculamos los estadisticos en dos regiones: la PUPILA (mascara de la segmentacion) y el IRIS (anillo alrededor).

# ---- Consigna 2.a: Region del iris -------------------------------------------
def estimar_radio_iris(roi_gris, morfo, factor_max=6.0):
    #Busca el borde iris-esclerotica con un perfil radial: promediamos el brillo sobre circulos de radio creciente
    #alrededor del centroide de la pupila y buscamos donde mas sube (el iris es mas oscuro que la esclerotica).
    #Usamos solo los sectores izquierdo y derecho (±45° de la horizontal) porque arriba y abajo lo tapan los parpados.
    #Empezamos a buscar en 1.6 veces el radio de la pupila: el iris siempre es al menos ~1.5 veces mas grande que la
    #pupila (pupila de 2 a 8 mm, iris de ~12 mm) y asi no confundimos el borde borroso de la pupila con el del iris
    h, w = roi_gris.shape
    xc, yc = morfo['centroide_x'], morfo['centroide_y']
    r_pupila = morfo['diametro_equivalente_px'] / 2
    r_max = min(factor_max * r_pupila, max(xc, w - 1 - xc)) #no buscar mas lejos que el borde de la ROI
    radios = np.arange(int(1.6 * r_pupila), int(r_max))
    if radios.size < 5:
        return None

    angulos = np.concatenate([np.linspace(-np.pi / 4, np.pi / 4, 30),
                              np.linspace(3 * np.pi / 4, 5 * np.pi / 4, 30)])
    suave = cv2.GaussianBlur(roi_gris, (5, 5), 0).astype(float)
    perfil = []
    for r in radios:
        xs = np.clip(np.round(xc + r * np.cos(angulos)).astype(int), 0, w - 1)
        ys = np.clip(np.round(yc + r * np.sin(angulos)).astype(int), 0, h - 1)
        perfil.append(suave[ys, xs].mean()) #brillo medio sobre el circulo de radio r
    derivada = np.gradient(np.array(perfil))
    i = int(np.argmax(derivada))
    if derivada[i] < 2: #no hay un salto claro de brillo: no encontramos el borde del iris
        return None
    return float(radios[i])


def mascara_del_iris(roi_gris, mascara_pupila, morfo, factor_iris=2.5):
    #El iris lo aproximamos como un anillo: desde 1.15 veces el radio de la pupila (para no tomar el borde de la pupila)
    #hasta un poco antes del borde del iris (85% del radio, el borde esta algo borroso). Si no se encuentra el borde
    #del iris, usamos 'factor_iris' veces el radio de la pupila
    h, w = roi_gris.shape
    r_pupila = morfo['diametro_equivalente_px'] / 2
    r_iris = estimar_radio_iris(roi_gris, morfo)
    r_externo = 0.85 * r_iris if r_iris is not None else factor_iris * r_pupila
    centro = (int(round(morfo['centroide_x'])), int(round(morfo['centroide_y'])))

    anillo = np.zeros((h, w), np.uint8)
    cv2.circle(anillo, centro, int(r_externo), 255, -1)
    cv2.circle(anillo, centro, int(1.15 * r_pupila), 0, -1)
    anillo[mascara_pupila > 0] = 0

    #Arriba y abajo el anillo puede tocar los parpados, y a los costados la esclerotica. Ambos son bastante mas
    #claros que el iris, asi que sacamos los pixeles "atipicamente claros" del anillo (criterio robusto con la
    #mediana y el MAD, que no se dejan influir por esos mismos pixeles)
    valores = roi_gris[anillo > 0].astype(float)
    if valores.size > 0:
        mediana = np.median(valores)
        mad = 1.4826 * np.median(np.abs(valores - mediana))
        anillo[(roi_gris > mediana + 3 * mad) & (anillo > 0)] = 0

    #Respaldo: si por algun motivo el anillo quedo vacio o casi vacio, usamos el anillo fijo de 'factor_iris'
    if np.count_nonzero(anillo) < 50:
        anillo = np.zeros((h, w), np.uint8)
        cv2.circle(anillo, centro, int(factor_iris * r_pupila), 255, -1)
        cv2.circle(anillo, centro, int(1.15 * r_pupila), 0, -1)
        anillo[mascara_pupila > 0] = 0
        r_iris = None
    return anillo, r_iris


def quitar_reflejos(mascara, roi_gris, umbral=230):
    #Saca de la region los reflejos del flash/ventana (pixeles casi blancos): no son textura del tejido
    #y arruinarian la media, el desvio y la asimetria
    limpia = mascara.copy()
    limpia[roi_gris > umbral] = 0
    return limpia


# ---- Consigna 2.b: Media, Desvio, Entropia y Skewness por canal --------------
def textura_primer_orden(canal_cuantizado, mascara, niveles=NIVELES_GRIS):
    #Estadistica global de los pixeles del canal que caen dentro de la mascara
    valores = canal_cuantizado[mascara > 0].astype(np.float64)
    if valores.size == 0:
        return {'media': np.nan, 'desvio': np.nan, 'entropia': np.nan, 'skewness': np.nan}

    media = valores.mean()
    desvio = valores.std()

    #Entropia de Shannon: H = -sum(p * log2(p)), con p = histograma normalizado (probabilidad de cada nivel)
    hist = np.bincount(valores.astype(int), minlength=niveles) / valores.size
    p = hist[hist > 0]
    entropia = -np.sum(p * np.log2(p)) #en bits (maximo posible = log2(64) = 6 bits)

    #Asimetria: >0 cola hacia valores claros, <0 cola hacia valores oscuros, 0 = simetrico
    asimetria = float(skew(valores)) if desvio > 0 else 0.0

    return {'media': media, 'desvio': desvio, 'entropia': entropia, 'skewness': asimetria}


def textura_por_canal_y_region(roi_rgb, regiones):
    #Cuantiza cada canal (R, G, B) a 64 niveles y calcula los 4 estadisticos en cada region
    canales = {'R': cuantizar(roi_rgb[:, :, 0]),
               'G': cuantizar(roi_rgb[:, :, 1]),
               'B': cuantizar(roi_rgb[:, :, 2])}
    resultados = {}  #resultados[region][canal] = {media, desvio, entropia, skewness}
    for nombre_region, mascara in regiones.items():
        resultados[nombre_region] = {c: textura_primer_orden(img, mascara) for c, img in canales.items()}
    return resultados, canales


def graficar_textura_primer_orden(roi_rgb, regiones, resultados):
    estadisticos = [('media', 'Media'), ('desvio', 'Desvío estándar'), ('entropia', 'Entropía [bits]'), ('skewness', 'Skewness')]
    colores = {'R': 'tab:red', 'G': 'tab:green', 'B': 'tab:blue'}
    nombres_region = list(regiones.keys())

    fig, ax = plt.subplots(1, 5, figsize=(20, 4.5))
    fig.canvas.manager.set_window_title("Parte 3.1 - Consigna 2: Textura de primer orden")

    #Imagen con las dos regiones pintadas encima para que se vea donde se calculo cada cosa
    superpuesta = roi_rgb.copy().astype(float)
    superpuesta[regiones['pupila'] > 0] = 0.5 * superpuesta[regiones['pupila'] > 0] + [127, 0, 0]
    superpuesta[regiones['iris'] > 0] = 0.6 * superpuesta[regiones['iris'] > 0] + [0, 100, 0]
    ax[0].imshow(superpuesta.astype(np.uint8))
    ax[0].set_title("2.a) Regiones: pupila (rojo) / iris (verde)")
    ax[0].axis('off')

    x = np.arange(len(nombres_region))
    for i, (clave, titulo) in enumerate(estadisticos):
        for j, canal in enumerate(['R', 'G', 'B']):
            valores = [resultados[reg][canal][clave] for reg in nombres_region]
            ax[i + 1].bar(x + (j - 1) * 0.25, valores, width=0.25, color=colores[canal], label=canal)
        ax[i + 1].set_xticks(x)
        ax[i + 1].set_xticklabels([n.capitalize() for n in nombres_region])
        ax[i + 1].set_title(f"2.b) {titulo}")
        ax[i + 1].grid(axis='y', alpha=0.3)
    ax[1].legend()
    plt.tight_layout()
    plt.show()


# =============================================================================
# PARTE 3.1 - CONSIGNA 3: TEXTURA DE SEGUNDO ORDEN (GLCM)
# =============================================================================

# ---- Consigna 3.a: Construccion de la GLCM dentro de una region --------------
def calcular_glcm(canal_cuantizado, mascara, niveles=NIVELES_GRIS, distancia=1):
    #graycomatrix trabaja sobre la imagen rectangular completa, pero nosotros queremos solo los pares de
    #pixeles que caen ADENTRO de la region (pupila o iris). Sumamos 1 a todos los niveles (quedan de 1 a 64),
    #ponemos en 0 todo lo que esta afuera de la mascara, calculamos la GLCM con 65 niveles y despues borramos
    #la fila y la columna 0. Asi se descartan todos los pares en los que algun pixel cae afuera de la region
    ys, xs = np.nonzero(mascara)
    if ys.size == 0:
        return None
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1 #recortamos al rectangulo de la region
    recorte = canal_cuantizado[y0:y1, x0:x1].astype(np.uint8) + 1
    recorte[mascara[y0:y1, x0:x1] == 0] = 0

    #4 direcciones (0°, 45°, 90°, 135°) para que el resultado no dependa de la orientacion de la textura.
    #symmetric=True cuenta el par (i,j) y tambien el (j,i)
    angulos = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    glcm = graycomatrix(recorte, distances=[distancia], angles=angulos, levels=niveles + 1, symmetric=True, normed=False)
    glcm = glcm[1:, 1:, :, :].astype(np.float64) #sacamos el nivel 0 (= afuera de la region)
    return glcm


# ---- Consigna 3.b: Atributos de la GLCM ------------------------------------------
def atributos_glcm(glcm):
    #graycoprops normaliza la GLCM (la pasa a probabilidades) y calcula cada atributo por direccion.
    #Promediamos las 4 direcciones
    if glcm is None:
        return {'contraste': np.nan, 'homogeneidad': np.nan, 'energia': np.nan, 'correlacion': np.nan}
    return {
        'contraste': graycoprops(glcm, 'contrast').mean(),       #sum p(i,j)*(i-j)^2: variacion local de intensidad (alto = mas textura)
        'homogeneidad': graycoprops(glcm, 'homogeneity').mean(), #sum p(i,j)/(1+(i-j)^2): que tan concentrada esta en la diagonal (alto = mas liso)
        'energia': graycoprops(glcm, 'energy').mean(),           #raiz(sum p(i,j)^2): uniformidad (alto = pocos patrones que se repiten = poca textura)
        'correlacion': graycoprops(glcm, 'correlation').mean(),  #dependencia lineal entre un pixel y su vecino
    }


def glcm_por_canal_y_region(canales, regiones):
    #Calcula la GLCM y sus atributos para cada canal (R, G, B) en cada region (pupila, iris)
    resultados = {}  #resultados[region][canal] = {'atributos': {...}, 'glcm': matriz}
    for nombre_region, mascara in regiones.items():
        resultados[nombre_region] = {}
        for canal, imagen in canales.items():
            glcm = calcular_glcm(imagen, mascara)
            resultados[nombre_region][canal] = {'atributos': atributos_glcm(glcm), 'glcm': glcm}
    return resultados


# ---- Consigna 3.c: Que canal aporta mayor riqueza textural del iris ----------
def canal_mas_rico_en_textura(resultados_glcm, region='iris'):
    #Riqueza textural = mucho contraste y baja homogeneidad y energia (muchos patrones distintos, poco uniforme).
    #Para cada atributo vemos que canal "gana" y despues contamos votos
    canales = list(resultados_glcm[region].keys())
    attr = {c: resultados_glcm[region][c]['atributos'] for c in canales}
    ganadores = {
        'contraste (mayor)': max(canales, key=lambda c: attr[c]['contraste']),
        'homogeneidad (menor)': min(canales, key=lambda c: attr[c]['homogeneidad']),
        'energia (menor)': min(canales, key=lambda c: attr[c]['energia']),
    }
    votos = {c: list(ganadores.values()).count(c) for c in canales}
    return max(votos, key=votos.get), ganadores


def graficar_glcm(resultados_glcm, region='iris'):
    canales = ['R', 'G', 'B']
    colores = {'R': 'tab:red', 'G': 'tab:green', 'B': 'tab:blue'}
    fig, ax = plt.subplots(2, 4, figsize=(18, 8.5))
    fig.canvas.manager.set_window_title("Parte 3.1 - Consigna 3: Textura de segundo orden (GLCM)")

    #Fila de arriba: la GLCM del iris de cada canal (promedio de las 4 direcciones, en escala log para que se vea)
    for j, canal in enumerate(canales):
        glcm = resultados_glcm[region][canal]['glcm']
        if glcm is None: #la region quedo vacia: no hay GLCM para mostrar
            ax[0, j].text(0.5, 0.5, "Sin datos", ha='center', va='center')
            ax[0, j].axis('off')
            continue
        matriz = glcm.mean(axis=(2, 3))
        ocupado = np.nonzero(matriz.sum(axis=0))[0]
        lim = min(NIVELES_GRIS, ocupado.max() + 5) if ocupado.size else NIVELES_GRIS #mostramos solo los niveles que aparecen
        ax[0, j].imshow(np.log1p(matriz[:lim, :lim]), cmap='magma', origin='upper')
        ax[0, j].set_title(f"3.a) GLCM {region} - canal {canal} (log)")
        ax[0, j].set_xlabel("nivel j (vecino)")
        ax[0, j].set_ylabel("nivel i (pixel)")

    #Comparacion de los 4 atributos entre pupila e iris para cada canal (texto)
    ax[0, 3].axis('off')
    texto = "Atributos promedio (4 direcciones, d=1)\n\n"
    for reg in resultados_glcm:
        texto += f"{reg.upper()}\n"
        for canal in canales:
            a = resultados_glcm[reg][canal]['atributos']
            texto += f"  {canal}: C={a['contraste']:.2f}  H={a['homogeneidad']:.2f}  E={a['energia']:.3f}  Corr={a['correlacion']:.2f}\n"
        texto += "\n"
    ax[0, 3].text(0, 0.95, texto, va='top', family='monospace', fontsize=9)

    #Fila de abajo: cada atributo en el iris, por canal
    for i, (clave, titulo) in enumerate([('contraste', 'Contraste'), ('homogeneidad', 'Homogeneidad'),
                                         ('energia', 'Energía'), ('correlacion', 'Correlación')]):
        valores = [resultados_glcm[region][c]['atributos'][clave] for c in canales]
        ax[1, i].bar(canales, valores, color=[colores[c] for c in canales])
        ax[1, i].set_title(f"3.b) {titulo} - {region}")
        ax[1, i].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.show()


# =============================================================================
# FUNCION PRINCIPAL DEL MODULO 3
# =============================================================================

def ejecutar_modulo_3(config):
    print("\n==================================================")
    print("  EJECUTANDO MÓDULO 3: CARACTERÍSTICAS Y CURVA TEMPORAL")
    print("==================================================")

    #Extrae los parametros del diccionario global del main
    video_path = config['video_path']
    roi = config['roi']
    frame_idx = config.get('frame_idx', 0)

    _, roi_rgb = cargar_frame_y_roi(video_path, frame_idx, roi)
    roi_gris = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2GRAY)
    roi_cuant = cuantizar(roi_gris) #ROI cuantizada a 64 niveles (se usa tambien en las consignas 2 y 3 de textura)

    # ---------------- PARTE 3.1 ----------------
    print("\n PARTE 3.1: Descriptores morfométricos y de textura en el frame segmentado")

    #Paso previo: segmentacion
    mascara, contorno = segmentar_pupila(roi_rgb)
    if contorno is None:
        print("[ERROR] No se encontró la pupila en este frame. Probá con otro frame o con otra ROI (opción 5 del menú).")
        return

    #Consigna 1: Descriptores morfometricos
    morfo = descriptores_morfometricos(mascara, contorno)
    print("\n Consigna 1: Descriptores morfométricos (forma)")
    print("-" * 50)
    print(f"1.a) Área (A):                 {morfo['area_px']:.0f} px")
    print(f"     Perímetro (P):            {morfo['perimetro_px']:.1f} px")
    print(f"1.b) Centroide (Xc, Yc):       ({morfo['centroide_x']:.1f}, {morfo['centroide_y']:.1f}) px")
    print(f"1.c) Circularidad (4πA/P²):    {morfo['circularidad']:.3f}")
    print(f"     Excentricidad:            {morfo['excentricidad']:.3f}")
    print(f"     Diámetro equivalente:     {morfo['diametro_equivalente_px']:.1f} px")

    graficar_morfometria(roi_rgb, roi_cuant, mascara, contorno, morfo)

    #Consigna 2: Textura de primer orden
    print("\n Consigna 2: Textura de primer orden (estadística global)")
    mascara_iris, r_iris = mascara_del_iris(roi_gris, mascara, morfo)
    if r_iris is None:
        print("[AVISO] No se encontró el borde del iris; se usa un anillo de 2.5 veces el radio de la pupila.")
    regiones = {'pupila': quitar_reflejos(mascara, roi_gris),
                'iris': quitar_reflejos(mascara_iris, roi_gris)}
    textura1, canales = textura_por_canal_y_region(roi_rgb, regiones)

    print(f"Calculada sobre la ROI cuantizada a {NIVELES_GRIS} niveles, canal por canal")
    print("-" * 66)
    print(f"{'Región':<8} | {'Canal':<5} | {'Media':>7} | {'Desvío':>7} | {'Entropía':>8} | {'Skewness':>8}")
    print("-" * 66)
    for region in regiones:
        for canal in ('R', 'G', 'B'):
            d = textura1[region][canal]
            print(f"{region:<8} | {canal:<5} | {d['media']:>7.2f} | {d['desvio']:>7.2f} | {d['entropia']:>8.3f} | {d['skewness']:>8.3f}")

    graficar_textura_primer_orden(roi_rgb, regiones, textura1)

    #Consigna 3: Textura de segundo orden (GLCM)
    print("\n Consigna 3: Textura de segundo orden (GLCM, distancia 1, promedio de 0°/45°/90°/135°)")
    textura2 = glcm_por_canal_y_region(canales, regiones)

    print("-" * 72)
    print(f"{'Región':<8} | {'Canal':<5} | {'Contraste':>9} | {'Homogen.':>8} | {'Energía':>8} | {'Correlación':>11}")
    print("-" * 72)
    for region in regiones:
        for canal in ('R', 'G', 'B'):
            a = textura2[region][canal]['atributos']
            print(f"{region:<8} | {canal:<5} | {a['contraste']:>9.3f} | {a['homogeneidad']:>8.3f} | {a['energia']:>8.4f} | {a['correlacion']:>11.3f}")

    canal_rico, ganadores = canal_mas_rico_en_textura(textura2, 'iris')
    print("\n3.c) Riqueza textural del estroma iridiano:")
    for criterio, canal in ganadores.items():
        print(f"     - {criterio}: canal {canal}")
    print(f"     => Canal con mayor riqueza textural en el iris: {canal_rico}")
    contrastes = [textura2['iris'][c]['atributos']['contraste'] for c in ('R', 'G', 'B')]
    dif_relativa = 100 * (max(contrastes) - min(contrastes)) / max(contrastes)
    print(f"     (diferencia de contraste entre el mejor y el peor canal: {dif_relativa:.1f} %)")

    print(f"     (diferencia de contraste entre el mejor y el peor canal: {dif_relativa:.1f} %)")

    #Discusion 3.c: que canal aporta mayor riqueza textural del estroma iridiano
    print(f"\n     Discusión: el canal {canal_rico} muestra la mayor riqueza textural del iris, pero la diferencia entre canales")
    print(f"     es de solo {dif_relativa:.1f} %, porque el iris es claro y poco saturado y la compresión del video (H.264)")
    print("     guarda el color a menor resolución que el brillo, así que los tres canales comparten casi la misma textura fina.")

    graficar_glcm(textura2, 'iris')