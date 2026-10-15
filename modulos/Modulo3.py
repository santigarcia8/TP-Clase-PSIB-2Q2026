#Modulo 3: Extraccion de caracteristicas, analisis de textura y dinamica temporal
#Modulo 3.1: Descriptores morfometricos y de textura en el frame segmentado
#Modulo 3.2: Procesamiento en video y curva temporal fisiologica
import os
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew
from scipy.signal import medfilt, savgol_filter
from skimage.feature import graycomatrix, graycoprops

NIVELES_GRIS = 64                 #niveles a los que se cuantiza la ROI (la guia pide 32 o 64)
CARPETA_RESULTADOS = "resultados" #carpeta donde se guardan los CSV del modulo
CANALES = ('R', 'G', 'B')
COLORES = {'R': 'tab:red', 'G': 'tab:green', 'B': 'tab:blue'}


# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================

def cargar_frame_y_roi(video_path, num_frame=0, roi_coords=None):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, num_frame) #se para en el frame elegido
    ok, frame_bgr = cap.read()
    cap.release()
    if not ok or frame_bgr is None:
        raise ValueError(f"No se pudo leer el frame {num_frame} del video.")
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB) #OpenCV lee en BGR, lo pasamos a RGB
    return frame_rgb, recortar(frame_rgb, roi_coords)


def recortar(imagen, roi_coords):
    if roi_coords is None:
        return imagen
    ymin, ymax, xmin, xmax = roi_coords
    return imagen[ymin:ymax, xmin:xmax]


def cuantizar(imagen, niveles=NIVELES_GRIS):
    #Pasa una imagen de 8 bits (0-255) a 'niveles' niveles de gris (0 a niveles-1)
    return (imagen.astype(np.uint16) * niveles // 256).astype(np.uint8)


def circulo(forma, centro, radio):
    #Mascara (0/255) de un circulo lleno
    m = np.zeros(forma, np.uint8)
    cv2.circle(m, (int(centro[0]), int(centro[1])), int(radio), 255, -1)
    return m


def rellenar(forma, contorno):
    #Mascara (0/255) con el interior de un contorno
    m = np.zeros(forma, np.uint8)
    cv2.drawContours(m, [contorno], -1, 255, thickness=cv2.FILLED)
    return m


# =============================================================================
# PARTE 3.1 - PASO PREVIO: SEGMENTACION DE LA PUPILA
# =============================================================================
#La consigna pide trabajar "a partir de la ROI segmentada". Como el Modulo 2 grafica cada metodo y no devuelve
#la mascara, armamos una version sin graficos que tambien se reutiliza en la parte 3.2 (todos los frames).
#Idea: la pupila es la region MAS OSCURA, REDONDA, con MUCHO CONTRASTE contra el iris y con un BORDE NITIDO.
#Las pestañas y las sombras de la piel tambien son oscuras, pero son finas, alargadas o tocan el borde de la ROI.

def puntuar_mancha(contorno, gris, gradiente, previa=None, niveles=None):
    #Puntaje de una mancha oscura segun cuanto se parece a una pupila (None si se descarta)
    h, w = gris.shape
    area = cv2.contourArea(contorno)
    if area < 0.0005 * h * w or area > 0.25 * h * w: #muy chica (ruido) o muy grande (sombra, ceja)
        return None
    env = cv2.convexHull(contorno)
    area_env = cv2.contourArea(env)
    circularidad = 4 * np.pi * area_env / cv2.arcLength(env, True) ** 2 #1 = circulo
    solidez = area / area_env                                         #las pestañas tienen huecos
    (x, y), r = cv2.minEnclosingCircle(env)
    llenado = area_env / (np.pi * r ** 2)                             #lo alargado llena poco su circulo

    adentro = rellenar((h, w), env)
    anillo = circulo((h, w), (x, y), 1.8 * r)
    anillo[adentro > 0] = 0
    if not anillo.any():
        return None
    contraste = max(gris[anillo > 0].mean() - gris[adentro > 0].mean(), 0) #brillo alrededor - brillo adentro
    #Oscuridad: la pupila es de lo mas oscuro de la ROI (1 = tan oscura como el 1% mas oscuro, 0 = como la mediana).
    #Las cejas, el iris marron o las sombras son oscuros pero no tanto, porque mezclan pelos, tejido o piel
    p1, p50 = niveles if niveles is not None else np.percentile(gris, (1, 50))
    oscuridad = np.clip((p50 - np.median(gris[adentro > 0])) / max(p50 - p1, 1), 0, 1)
    borde = cv2.findContours(adentro, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0][0][:, 0, :]
    nitidez = gradiente[borde[:, 1], borde[:, 0]].mean() #gradiente medio sobre el borde

    bx, by, bw, bh = cv2.boundingRect(env)
    toca_borde = bx <= 1 or by <= 1 or bx + bw >= w - 1 or by + bh >= h - 1
    tamaño = area / (area + 0.002 * h * w)                   #las manchitas diminutas (pelos, poros) pesan menos
    #Contexto de ojo: a los costados de la pupila (±30° de la horizontal) primero esta el iris y mas afuera la
    #esclerotica, que es clara. Medimos cuanto sube el brillo a lo largo de esos rayos (hasta 8 radios). Un pelo o
    #una sombra sobre la piel tiene alrededor un brillo parejo, asi que este termino lo distingue de una pupila
    ang = np.r_[np.linspace(-np.pi / 6, np.pi / 6, 7), np.linspace(5 * np.pi / 6, 7 * np.pi / 6, 7)]
    rad = np.arange(1.3 * r, 8 * r, max(1.0, r / 4))
    xs = np.clip(x + np.outer(np.cos(ang), rad), 0, w - 1).astype(int)
    ys = np.clip(y + np.outer(np.sin(ang), rad), 0, h - 1).astype(int)
    perfil = gris[ys, xs].astype(float).mean(axis=0)
    esclera = max(perfil[np.argmin(perfil):].max() - perfil.min(), 0) if perfil.size else 0
    centrado = 1 / (1 + 2 * np.hypot(x - w / 2, y - h / 2) / np.hypot(w, h)) #la ROI se dibuja alrededor del ojo
    #Iris concentrico: alrededor de la pupila esta el iris, y su borde con la esclerotica (blanca) aparece A LA MISMA
    #DISTANCIA hacia la izquierda y hacia la derecha (entre 2 y 6 radios). Alrededor de una letra, un pelo o una sombra
    #los bordes que aparecen estan a distancias desparejas o son debiles
    radios = np.arange(2 * r, 6 * r, max(1.0, r / 4))
    bordes = []
    for centro_ang in (0, np.pi):
        ang = centro_ang + np.linspace(-np.pi / 6, np.pi / 6, 9)
        xs = np.clip(x + np.outer(np.cos(ang), radios), 0, w - 1).astype(int)
        ys = np.clip(y + np.outer(np.sin(ang), radios), 0, h - 1).astype(int)
        derivada = np.diff(gris[ys, xs].astype(float).mean(axis=0)) if radios.size > 2 else np.zeros(1)
        bordes.append((derivada.max(), radios[np.argmax(derivada)]))
    (fuerza_d, radio_d), (fuerza_i, radio_i) = bordes
    iris = min(fuerza_d, fuerza_i) if abs(radio_d - radio_i) < 0.3 * max(radio_d, radio_i) else 0.0
    puntaje = (circularidad ** 2 * llenado ** 2 * solidez * contraste * nitidez * tamaño * centrado
               * (1 + esclera / 25) * oscuridad ** 3 * (1 + max(iris, 0) / 4) * (0.3 if toca_borde else 1))
    if previa is not None: #en video: la pupila casi no se mueve ni cambia de tamaño de un frame al siguiente
        px, py, pr = previa
        puntaje *= 1 / (1 + (np.hypot(x - px, y - py) / (2 * pr)) ** 2) * np.exp(-abs(np.log(np.sqrt(area_env / np.pi) / pr)))
    return {'puntaje': puntaje, 'envolvente': env, 'centro': (float(x), float(y)), 'area': area_env}


def elegir_pupila(candidatas, tolerancia=0.6):
    #Elige la mancha de mayor puntaje. Los distintos umbrales generan manchas "anidadas" con puntajes parecidos y las
    #mas grandes se meten en el anillo oscuro del iris: entre las anidadas con puntaje >= 60% del mejor, nos quedamos
    #con la MAS CHICA, que sigue el borde negro real (y evita que la medicion salte entre frames)
    if not candidatas:
        return None
    mejor = max(candidatas, key=lambda c: c['puntaje'])
    anidadas = [c for c in candidatas if c['puntaje'] >= tolerancia * mejor['puntaje']
                and 0.4 * mejor['area'] <= c['area'] <= mejor['area']
                and cv2.pointPolygonTest(mejor['envolvente'], c['centro'], False) >= 0]
    return min(anidadas, key=lambda c: c['area'])


def segmentar_pupila(roi_rgb, previa=None):
    #Devuelve la mascara de la pupila (0/255) y su contorno, o (mascara vacia, None) si no la encuentra.
    #'previa' = (x, y, radio) de la pupila en los frames anteriores (solo en video), para no saltar a otra mancha
    gris = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2GRAY)
    suave = cv2.medianBlur(gris, 5) #la mediana saca ruido impulsivo sin borronear el borde
    h, w = gris.shape
    gradiente = np.hypot(cv2.Sobel(suave, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(suave, cv2.CV_32F, 0, 1, ksize=3))
    k = max(3, int(min(h, w) / 80)) #kernel proporcional al tamaño de la ROI (impar)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k + (k % 2 == 0),) * 2)
    niveles = np.percentile(suave, (1, 50)) #se calculan una vez para todas las manchas

    def candidatas_de(binaria):
        binaria = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, kernel)  #apertura: borra pestañas finas y puntitos
        binaria = cv2.morphologyEx(binaria, cv2.MORPH_CLOSE, kernel) #cierre: tapa huecos chicos (reflejos)
        puntuadas = [puntuar_mancha(c, suave, gradiente, previa, niveles)
                     for c in cv2.findContours(binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]]
        return [p for p in puntuadas if p is not None]

    #Paso 1: umbralizacion multinivel (1%, 2%, ..., 25% de pixeles mas oscuros) y eleccion de la mejor mancha
    candidatas = []
    for porcentaje in [1, 2, 3, 5, 8, 12, 16, 20, 25]:
        candidatas += candidatas_de((suave <= np.percentile(suave, porcentaje)).astype(np.uint8) * 255)
    elegida = elegir_pupila(candidatas)
    if elegida is None:
        return np.zeros((h, w), np.uint8), None

    #Paso 2: si el iris es oscuro, la mancha elegida puede ser iris + pupila. Aplicamos Otsu SOLO adentro de ella
    #y bajamos un nivel si hay una mancha mas oscura y redonda. No bajamos si las clases tienen brillos parecidos
    #(es solo textura) o si lo oscuro es casi toda la mancha (lo unico que se separa es un reflejo)
    for _ in range(3):
        region = rellenar((h, w), elegida['envolvente'])
        adentro = suave[region > 0]
        u2, _ = cv2.threshold(adentro.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        oscuros, claros = adentro[adentro <= u2], adentro[adentro > u2]
        if oscuros.size == 0 or claros.size == 0 or claros.mean() - oscuros.mean() < 20 or oscuros.size / adentro.size > 0.85:
            break
        sub = elegir_pupila(candidatas_de(((suave <= u2) & (region > 0)).astype(np.uint8) * 255))
        if sub is None or 4 * np.pi * sub['area'] / cv2.arcLength(sub['envolvente'], True) ** 2 < 0.6 or sub['area'] < 0.5 * oscuros.size:
            break
        elegida = sub

    #Paso 3: mascara = envolvente convexa (la pupila es convexa: se rellenan los "mordiscos" de los reflejos)
    contorno = completar_con_circulo(elegida, gradiente, (h, w))
    return rellenar((h, w), contorno), contorno


def completar_con_circulo(elegida, gradiente, forma):
    #Si un reflejo "vela" parte de la pupila, la mancha oscura queda recortada de ese lado. Como la pupila es redonda,
    #ajustamos un circulo (minimos cuadrados) a los puntos del borde con gradiente fuerte (el borde real pupila-iris)
    #y nos quedamos con la union de la mancha y ese circulo. Si el circulo no es razonable, queda la mancha.
    env = elegida['envolvente']
    borde = cv2.findContours(rellenar(forma, env), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0][0][:, 0, :].astype(float)
    g = gradiente[borde[:, 1].astype(int), borde[:, 0].astype(int)]
    fuertes = borde[g >= np.percentile(g, 40)]
    if len(fuertes) < 10:
        return env
    #circulo por minimos cuadrados: x^2 + y^2 = 2a x + 2b y + c
    A = np.c_[2 * fuertes, np.ones(len(fuertes))]
    (a, b, c), *_ = np.linalg.lstsq(A, (fuertes ** 2).sum(axis=1), rcond=None)
    radio = np.sqrt(c + a ** 2 + b ** 2)
    r_mancha = np.sqrt(elegida['area'] / np.pi)
    if not (0.9 * r_mancha <= radio <= 1.4 * r_mancha) or np.hypot(a - elegida['centro'][0], b - elegida['centro'][1]) > 0.5 * r_mancha:
        return env
    union = rellenar(forma, env)
    union[circulo(forma, (round(a), round(b)), radio) > 0] = 255
    return cv2.convexHull(max(cv2.findContours(union, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0], key=cv2.contourArea))


def referencia_pupila(video_path, roi, frame_centro, n=11, ventana_s=1.0):
    #Consenso entre frames: segmenta n frames cercanos (±1 s) y se queda con la ubicacion y el tamaño en los que
    #coinciden la mayoria. La pupila aparece siempre en el mismo lugar; los errores (cejas, letras, sombras) van
    #cambiando de un frame a otro. Devuelve (x, y, radio) o None si no hay acuerdo suficiente.
    cap = cv2.VideoCapture(video_path)
    fps, total = cap.get(cv2.CAP_PROP_FPS) or 30.0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    detecciones = []
    for fr in np.linspace(max(0, frame_centro - ventana_s * fps), min(total - 1, frame_centro + ventana_s * fps), n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fr))
        ok, frame_bgr = cap.read()
        if not ok:
            continue
        mascara, contorno = segmentar_pupila(recortar(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), roi))
        if contorno is not None:
            m = descriptores_morfometricos(mascara, contorno)
            detecciones.append((m['centroide_x'], m['centroide_y'], m['diametro_equivalente_px'] / 2))
    cap.release()
    if not detecciones:
        return None
    det = np.array(detecciones)
    #cada deteccion "vota" por las que estan cerca (menos de 1.5 radios) y tienen un tamaño parecido
    apoyos = [(np.hypot(det[:, 0] - x, det[:, 1] - y) < 1.5 * r) & (np.abs(np.log(det[:, 2] / r)) < 0.4) for x, y, r in det]
    mejor = max(apoyos, key=np.sum)
    return tuple(np.median(det[mejor], axis=0)) if mejor.sum() >= max(3, n // 3) else None


def avisar_si_roi_grande(config):
    #Con el frame entero (o una ROI muy grande) aparecen objetos parecidos a una pupila (letras, pelo, sombras) y la
    #deteccion puede fallar, sobre todo con ojos oscuros. Se avisa para que se dibuje la ROI alrededor del ojo
    cap = cv2.VideoCapture(config['video_path'])
    area_frame = cap.get(cv2.CAP_PROP_FRAME_WIDTH) * cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cap.release()
    roi = config['roi']
    area_roi = area_frame if roi is None else (roi[1] - roi[0]) * (roi[3] - roi[2])
    if area_roi > 0.4 * area_frame:
        print("[AVISO] La ROI ocupa casi todo el frame: la pupila se puede confundir con otros objetos oscuros")
        print("        (letras, pelo, sombras). Para mejores resultados, dibujá la ROI alrededor del ojo (opción 5 del menú).")


def preparar_frame(config):
    #Segmenta el frame de trabajo una sola vez y lo reutiliza en las 3 consignas de la 3.1
    #(se vuelve a calcular si en el main se cambia el frame o la ROI)
    clave = (config['video_path'], config.get('frame_idx', 0), tuple(config['roi']) if config['roi'] else None)
    if config.get('frame_m3', {}).get('clave') != clave:
        avisar_si_roi_grande(config)
        _, roi_rgb = cargar_frame_y_roi(config['video_path'], config.get('frame_idx', 0), config['roi'])
        referencia = referencia_pupila(config['video_path'], config['roi'], config.get('frame_idx', 0))
        mascara, contorno = segmentar_pupila(roi_rgb, referencia)
        config['frame_m3'] = {'clave': clave, 'roi_rgb': roi_rgb, 'roi_gris': cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2GRAY),
                              'mascara': mascara, 'contorno': contorno}
    f = config['frame_m3']
    if f['contorno'] is None:
        print("[ERROR] No se encontró la pupila en este frame. Probá con otro frame o con otra ROI (opción 5 del menú).")
        return None
    return f


# =============================================================================
# PARTE 3.1 - CONSIGNA 1: DESCRIPTORES MORFOMETRICOS (FORMA)
# =============================================================================

def descriptores_morfometricos(mascara, contorno):
    M = cv2.moments(mascara, binaryImage=True)
    #1.a) Area = cantidad de pixeles de la mascara | Perimetro = largo del contorno cerrado
    area = float(np.count_nonzero(mascara))
    perimetro = cv2.arcLength(contorno, True)
    #1.b) Centroide = momentos de orden 1 / momento de orden 0 (area)
    xc, yc = M["m10"] / M["m00"], M["m01"] / M["m00"]
    #1.c) Circularidad = 4*pi*A/P^2 (1 = circulo). Usamos el area del poligono del contorno para que area y perimetro
    #     se midan sobre la misma curva (con la cuenta de pixeles puede dar apenas > 1 por la discretizacion)
    circularidad = 4 * np.pi * cv2.contourArea(contorno) / perimetro ** 2
    #     Excentricidad: autovalores de la matriz de covarianza (momentos centrales de orden 2), proporcionales al
    #     cuadrado de los semiejes de la elipse equivalente. 0 = circulo, cerca de 1 = elipse muy aplastada
    cov = np.array([[M["mu20"], M["mu11"]], [M["mu11"], M["mu02"]]]) / M["m00"]
    l_menor, l_mayor = np.sort(np.linalg.eigvalsh(cov))
    excentricidad = np.sqrt(1 - l_menor / l_mayor) if l_mayor > 0 else 0.0
    return {'area_px': area, 'perimetro_px': perimetro, 'centroide_x': xc, 'centroide_y': yc,
            'circularidad': circularidad, 'excentricidad': excentricidad,
            'diametro_equivalente_px': 2 * np.sqrt(area / np.pi)} #diametro de un circulo de igual area (para D(t))


def consigna_1_morfometria(config):
    f = preparar_frame(config)
    if f is None:
        return
    m = descriptores_morfometricos(f['mascara'], f['contorno'])
    print("\n PARTE 3.1 - Consigna 1: Descriptores morfométricos (forma)")
    print(f"1.a) Área (A) = {m['area_px']:.0f} px | Perímetro (P) = {m['perimetro_px']:.1f} px")
    print(f"1.b) Centroide (Xc, Yc) = ({m['centroide_x']:.1f}, {m['centroide_y']:.1f}) px")
    print(f"1.c) Circularidad (4πA/P²) = {m['circularidad']:.3f} | Excentricidad = {m['excentricidad']:.3f}")
    print(f"     Diámetro equivalente = {m['diametro_equivalente_px']:.1f} px")

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    fig.canvas.manager.set_window_title("Parte 3.1 - Consigna 1: Descriptores morfométricos")
    ax[0].imshow(cuantizar(f['roi_gris']), cmap='gray', vmin=0, vmax=NIVELES_GRIS - 1)
    ax[0].set_title(f"ROI cuantizada a {NIVELES_GRIS} niveles")
    ax[1].imshow(f['mascara'], cmap='gray')
    ax[1].set_title(f"1.a) Máscara de la pupila\nA = {m['area_px']:.0f} px   P = {m['perimetro_px']:.1f} px")
    ax[2].imshow(f['roi_rgb'])
    ax[2].plot(f['contorno'][:, 0, 0], f['contorno'][:, 0, 1], '-', color='lime', lw=1.5, label='Contorno')
    ax[2].plot(m['centroide_x'], m['centroide_y'], '+r', ms=14, mew=2,
               label=f"1.b) Centroide ({m['centroide_x']:.1f}, {m['centroide_y']:.1f})")
    ax[2].set_title(f"1.c) Circularidad = {m['circularidad']:.3f}\nExcentricidad = {m['excentricidad']:.3f}")
    ax[2].legend(loc='lower right', fontsize=8)
    for a in ax:
        a.axis('off')
    plt.tight_layout()
    plt.show()


# =============================================================================
# PARTE 3.1 - CONSIGNA 2: TEXTURA DE PRIMER ORDEN (ESTADISTICA GLOBAL)
# =============================================================================
#Media, Desvio, Entropia y Skewness sobre la ROI cuantizada (64 niveles), en R, G y B por separado,
#en dos regiones: la PUPILA (mascara de la segmentacion) y el IRIS (anillo alrededor de la pupila).

def mascara_del_iris(roi_gris, mascara_pupila, morfo, factor_iris=2.5):
    #2.a) El iris se aproxima como un anillo desde 1.15 veces el radio de la pupila hasta el 85% del radio del iris.
    #El borde iris-esclerotica se busca con un perfil radial: brillo medio sobre circulos de radio creciente
    #(solo sectores izquierdo y derecho, ±45°, porque arriba y abajo tapan los parpados) y nos quedamos con el
    #radio donde mas sube el brillo. Se busca desde 1.6 veces el radio de la pupila (el iris siempre es al menos
    #~1.5 veces mas grande) para no confundirlo con el borde borroso de la pupila.
    h, w = roi_gris.shape
    xc, yc, rp = morfo['centroide_x'], morfo['centroide_y'], morfo['diametro_equivalente_px'] / 2
    radios = np.arange(int(1.6 * rp), int(min(6 * rp, max(xc, w - 1 - xc))))
    r_iris = None
    if radios.size >= 5:
        ang = np.r_[np.linspace(-np.pi / 4, np.pi / 4, 30), np.linspace(3 * np.pi / 4, 5 * np.pi / 4, 30)]
        suave = cv2.GaussianBlur(roi_gris, (5, 5), 0).astype(float)
        perfil = [suave[np.clip(np.round(yc + r * np.sin(ang)).astype(int), 0, h - 1),
                        np.clip(np.round(xc + r * np.cos(ang)).astype(int), 0, w - 1)].mean() for r in radios]
        derivada = np.gradient(np.array(perfil))
        if derivada.max() >= 2: #hay un salto claro de brillo
            r_iris = float(radios[np.argmax(derivada)])

    centro = (round(xc), round(yc))
    def anillo_hasta(r_ext):
        a = circulo((h, w), centro, r_ext)
        a[circulo((h, w), centro, 1.15 * rp) > 0] = 0
        a[mascara_pupila > 0] = 0
        return a

    anillo = anillo_hasta(0.85 * r_iris if r_iris else factor_iris * rp)
    #Sacamos del anillo los pixeles atipicamente claros (parpados, esclerotica) con un criterio robusto (mediana + MAD)
    valores = roi_gris[anillo > 0].astype(float)
    if valores.size:
        mediana = np.median(valores)
        anillo[(roi_gris > mediana + 3 * 1.4826 * np.median(np.abs(valores - mediana))) & (anillo > 0)] = 0
    if np.count_nonzero(anillo) < 50: #respaldo: si quedo casi vacio, anillo fijo de factor_iris veces el radio
        anillo, r_iris = anillo_hasta(factor_iris * rp), None
    return anillo, r_iris


def regiones_y_canales(f):
    #Regiones pupila e iris (sin reflejos: pixeles casi blancos que no son tejido) y canales R, G, B cuantizados
    morfo = descriptores_morfometricos(f['mascara'], f['contorno'])
    iris, r_iris = mascara_del_iris(f['roi_gris'], f['mascara'], morfo)
    regiones = {}
    for nombre, m in (('pupila', f['mascara']), ('iris', iris)):
        regiones[nombre] = m.copy()
        regiones[nombre][f['roi_gris'] > 230] = 0
    canales = {c: cuantizar(f['roi_rgb'][:, :, i]) for i, c in enumerate(CANALES)}
    return regiones, canales, r_iris


def textura_primer_orden(canal, mascara, niveles=NIVELES_GRIS):
    #2.b) Estadistica global de los pixeles del canal dentro de la mascara
    v = canal[mascara > 0].astype(np.float64)
    if v.size == 0:
        return dict.fromkeys(('media', 'desvio', 'entropia', 'skewness'), np.nan)
    p = np.bincount(v.astype(int), minlength=niveles) / v.size #histograma normalizado = probabilidad de cada nivel
    p = p[p > 0]
    return {'media': v.mean(), 'desvio': v.std(),
            'entropia': -np.sum(p * np.log2(p)),                   #Shannon, en bits (maximo log2(64) = 6)
            'skewness': float(skew(v)) if v.std() > 0 else 0.0}   #>0 cola hacia claros, <0 hacia oscuros


def consigna_2_primer_orden(config):
    f = preparar_frame(config)
    if f is None:
        return
    regiones, canales, r_iris = regiones_y_canales(f)
    res = {reg: {c: textura_primer_orden(canales[c], m) for c in CANALES} for reg, m in regiones.items()}

    print("\n PARTE 3.1 - Consigna 2: Textura de primer orden (ROI cuantizada a 64 niveles)")
    if r_iris is None:
        print("[AVISO] No se encontró el borde del iris; se usa un anillo de 2.5 veces el radio de la pupila.")
    print(f"{'Región':<7}|{'Canal':^7}|{'Media':>8} |{'Desvío':>8} |{'Entropía':>9} |{'Skewness':>9}")
    for reg in res:
        for c in CANALES:
            d = res[reg][c]
            print(f"{reg:<7}|{c:^7}|{d['media']:>8.2f} |{d['desvio']:>8.2f} |{d['entropia']:>9.3f} |{d['skewness']:>9.3f}")

    fig, ax = plt.subplots(1, 5, figsize=(20, 4.5))
    fig.canvas.manager.set_window_title("Parte 3.1 - Consigna 2: Textura de primer orden")
    vista = f['roi_rgb'].astype(float)
    vista[regiones['pupila'] > 0] = 0.5 * vista[regiones['pupila'] > 0] + [127, 0, 0]
    vista[regiones['iris'] > 0] = 0.6 * vista[regiones['iris'] > 0] + [0, 100, 0]
    ax[0].imshow(vista.astype(np.uint8)); ax[0].axis('off')
    ax[0].set_title("2.a) Regiones: pupila (rojo) / iris (verde)")
    for i, (clave, titulo) in enumerate([('media', 'Media'), ('desvio', 'Desvío estándar'),
                                         ('entropia', 'Entropía [bits]'), ('skewness', 'Skewness')]):
        for j, c in enumerate(CANALES):
            ax[i + 1].bar(np.arange(2) + (j - 1) * 0.25, [res[r][c][clave] for r in res], 0.25, color=COLORES[c], label=c)
        ax[i + 1].set_xticks(range(2)); ax[i + 1].set_xticklabels(['Pupila', 'Iris'])
        ax[i + 1].set_title(f"2.b) {titulo}"); ax[i + 1].grid(axis='y', alpha=0.3)
    ax[1].legend()
    plt.tight_layout()
    plt.show()


# =============================================================================
# PARTE 3.1 - CONSIGNA 3: TEXTURA DE SEGUNDO ORDEN (GLCM)
# =============================================================================
#La GLCM cuenta cuantas veces un pixel de nivel i tiene un vecino de nivel j (distancia 1, en 0°/45°/90°/135°).
#A diferencia de la textura de primer orden (solo el histograma), tiene en cuenta la DISPOSICION ESPACIAL.

def glcm_region(canal, mascara, niveles=NIVELES_GRIS):
    #3.a) GLCM solo con los pares de pixeles que caen ADENTRO de la region: sumamos 1 a todos los niveles,
    #ponemos en 0 lo de afuera, calculamos con 65 niveles y borramos la fila y columna 0 (= afuera de la region)
    ys, xs = np.nonzero(mascara)
    if ys.size == 0:
        return None, dict.fromkeys(('contraste', 'homogeneidad', 'energia', 'correlacion'), np.nan)
    sl = (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1)) #recorte al rectangulo de la region
    recorte = canal[sl].astype(np.uint8) + 1
    recorte[mascara[sl] == 0] = 0
    glcm = graycomatrix(recorte, [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=niveles + 1,
                        symmetric=True)[1:, 1:].astype(np.float64)
    #3.b) Atributos (graycoprops normaliza la GLCM; promediamos las 4 direcciones):
    return glcm, {
        'contraste': graycoprops(glcm, 'contrast').mean(),       #sum p(i,j)(i-j)^2: variacion local (alto = mas textura)
        'homogeneidad': graycoprops(glcm, 'homogeneity').mean(), #sum p(i,j)/(1+(i-j)^2): concentracion en la diagonal
        'energia': graycoprops(glcm, 'energy').mean(),           #raiz(sum p^2): uniformidad (alto = poca textura)
        'correlacion': graycoprops(glcm, 'correlation').mean(),  #dependencia lineal entre pixel y vecino
    }


def consigna_3_glcm(config):
    f = preparar_frame(config)
    if f is None:
        return
    regiones, canales, _ = regiones_y_canales(f)
    res = {reg: {c: glcm_region(canales[c], m) for c in CANALES} for reg, m in regiones.items()}

    print("\n PARTE 3.1 - Consigna 3: Textura de segundo orden (GLCM, d=1, promedio de 0°/45°/90°/135°)")
    print(f"{'Región':<7}|{'Canal':^7}|{'Contraste':>10} |{'Homogen.':>9} |{'Energía':>8} |{'Correlación':>12}")
    for reg in res:
        for c in CANALES:
            a = res[reg][c][1]
            print(f"{reg:<7}|{c:^7}|{a['contraste']:>10.3f} |{a['homogeneidad']:>9.3f} |{a['energia']:>8.4f} |{a['correlacion']:>12.3f}")

    #3.c) Riqueza textural del iris: gana el canal con mayor contraste y menor homogeneidad y energia (por votos)
    attr = {c: res['iris'][c][1] for c in CANALES}
    ganadores = {'contraste (mayor)': max(CANALES, key=lambda c: attr[c]['contraste']),
                 'homogeneidad (menor)': min(CANALES, key=lambda c: attr[c]['homogeneidad']),
                 'energia (menor)': min(CANALES, key=lambda c: attr[c]['energia'])}
    canal_rico = max(CANALES, key=list(ganadores.values()).count)
    contrastes = [attr[c]['contraste'] for c in CANALES]
    dif_relativa = 100 * (max(contrastes) - min(contrastes)) / max(contrastes)
    print("\n3.c) Riqueza textural del estroma iridiano: " + " | ".join(f"{k}: {v}" for k, v in ganadores.items()))
    print(f"     => Canal con mayor riqueza textural: {canal_rico} (diferencia de contraste entre canales: {dif_relativa:.1f} %)")
    if dif_relativa < 10:
        print(f"     Discusión: la diferencia entre canales es chica ({dif_relativa:.1f} %): en un iris claro y poco saturado R, G y B varían")
        print("     casi juntos, y la compresión del video (H.264) guarda el color a menor resolución que el brillo, así que los")
        print("     tres canales comparten casi la misma textura fina. No se puede afirmar que un canal sea claramente mejor.")
    else:
        print(f"     Discusión: el canal {canal_rico} aporta claramente más textura ({dif_relativa:.1f} % más contraste). En iris pigmentados")
        print("     (marrones) la melanina del estroma absorbe el azul y el verde, por eso esos canales quedan oscuros y con poco")
        print("     contraste, mientras que en el rojo se ven mejor las fibras y criptas del estroma.")

    fig, ax = plt.subplots(2, 4, figsize=(18, 8.5))
    fig.canvas.manager.set_window_title("Parte 3.1 - Consigna 3: Textura de segundo orden (GLCM)")
    for j, c in enumerate(CANALES):
        glcm = res['iris'][c][0]
        ax[0, j].set_title(f"3.a) GLCM iris - canal {c} (log)")
        if glcm is None:
            ax[0, j].text(0.5, 0.5, "Sin datos", ha='center'); ax[0, j].axis('off')
            continue
        matriz = glcm.mean(axis=(2, 3))
        lim = min(NIVELES_GRIS, np.nonzero(matriz.sum(axis=0))[0].max() + 5) #solo los niveles que aparecen
        ax[0, j].imshow(np.log1p(matriz[:lim, :lim]), cmap='magma')
        ax[0, j].set_xlabel("nivel j (vecino)"); ax[0, j].set_ylabel("nivel i (pixel)")
    ax[0, 3].axis('off')
    ax[0, 3].text(0, 0.95, "\n".join(f"{reg.upper()} {c}: C={res[reg][c][1]['contraste']:.2f} H={res[reg][c][1]['homogeneidad']:.2f} "
                                     f"E={res[reg][c][1]['energia']:.3f} Corr={res[reg][c][1]['correlacion']:.2f}"
                                     for reg in res for c in CANALES), va='top', family='monospace', fontsize=8)
    for i, prop in enumerate(['contraste', 'homogeneidad', 'energia', 'correlacion']):
        ax[1, i].bar(CANALES, [attr[c][prop] for c in CANALES], color=[COLORES[c] for c in CANALES])
        ax[1, i].set_title(f"3.b) {prop.capitalize()} - iris"); ax[1, i].grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()


# =============================================================================
# PARTE 3.2 - CONSIGNA 1: SEGMENTACION AUTOMATICA FRAME A FRAME Y VALIDACION
# =============================================================================

def medir_video_frame_a_frame(video_path, roi):
    #1.a) Recorre el video, segmenta la pupila en cada frame (misma funcion que la 3.1) y guarda area, diametro,
    #circularidad y centroide. Si en un frame no se encuentra la pupila se guarda NaN
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0 #si el video no trae los fps, asumimos 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    claves = ('area_px', 'diametro_equivalente_px', 'circularidad', 'centroide_x', 'centroide_y')
    inicial = referencia_pupila(video_path, roi, int(fps)) #consenso en los primeros 2 s para arrancar el seguimiento
    filas, recientes, forma = [], [], None
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        forma = recortar(frame_bgr, roi).shape[:2]
        #referencia = mediana de las ultimas 15 detecciones (la mediana no se deja arrastrar por un frame erroneo)
        previa = tuple(np.median(recientes[-15:], axis=0)) if len(recientes) >= 5 else inicial
        mascara, contorno = segmentar_pupila(recortar(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), roi), previa)
        morfo = descriptores_morfometricos(mascara, contorno) if contorno is not None else {}
        filas.append([morfo.get(k, np.nan) for k in claves])
        if morfo:
            recientes.append((morfo['centroide_x'], morfo['centroide_y'], morfo['diametro_equivalente_px'] / 2))
        if len(filas) % 30 == 0:
            print(f"\r     Procesando frame {len(filas)}/{total}...", end="")
    cap.release()
    print(f"\r     Procesados {len(filas)} frames.                 ")
    datos = np.array(filas, dtype=float)
    return {'fps': fps, 'frame': np.arange(len(filas)), 'tiempo_s': np.arange(len(filas)) / fps,
            'area_px': datos[:, 0], 'diametro_px': datos[:, 1], 'circularidad': datos[:, 2],
            'centroide_x': datos[:, 3], 'centroide_y': datos[:, 4], 'encontrada': ~np.isnan(datos[:, 0]), 'forma_roi': forma}


def obtener_medicion(config):
    #Mide el video una sola vez y lo reutiliza en las 4 consignas de la 3.2 (se vuelve a medir si cambia la ROI)
    clave = (config['video_path'], tuple(config['roi']) if config['roi'] else None)
    if config.get('medicion_m3', {}).get('clave') != clave:
        avisar_si_roi_grande(config)
        print("     Midiendo la pupila en todos los frames del video...")
        config['medicion_m3'] = medir_video_frame_a_frame(config['video_path'], config['roi'])
        config['medicion_m3']['clave'] = clave
    return config['medicion_m3']


def guardar_curva_csv(medicion, nombre_video, d_limpia=None, a_limpia=None):
    #Salida estandarizada: curva frame a frame (cruda y, si ya se calculo, sin artefactos) en un CSV
    os.makedirs(CARPETA_RESULTADOS, exist_ok=True)
    ruta = os.path.join(CARPETA_RESULTADOS, f"{nombre_video}_curva_frame_a_frame.csv")
    ok = frames_confiables(medicion)
    with open(ruta, 'w', newline='', encoding='utf-8') as f:
        e = csv.writer(f)
        e.writerow(['frame', 'tiempo_s', 'area_px', 'diametro_px', 'circularidad', 'frame_confiable']
                   + (['area_sin_artefactos_px', 'diametro_sin_artefactos_px'] if d_limpia is not None else []))
        for k in range(len(medicion['frame'])):
            e.writerow([k, f"{medicion['tiempo_s'][k]:.4f}", f"{medicion['area_px'][k]:.1f}",
                        f"{medicion['diametro_px'][k]:.3f}", f"{medicion['circularidad'][k]:.4f}", int(ok[k])]
                       + ([f"{a_limpia[k]:.1f}", f"{d_limpia[k]:.3f}"] if d_limpia is not None else []))
    return ruta


def frames_confiables(medicion, circ_min=0.8, salto_max=0.2, devolver_motivos=False):
    #Marca que frames tienen una segmentacion confiable. Los que no lo son son ARTEFACTOS y se reemplazan por
    #interpolacion (consignas 3 y 4). Cada criterio corresponde a una fuente de artefacto:
    d = np.nan_to_num(medicion['diametro_px'])
    mediana = np.nanmedian(medicion['diametro_px'])
    circ = np.nan_to_num(medicion['circularidad'])
    #mediana de los vecinos (~0.25 s a cada lado, sin el propio frame), calculada solo con frames que pasan los otros
    #criterios: si no, una zona de detecciones erroneas "arrastra" la mediana y deja pasar frames malos
    previos = medicion['encontrada'] & (circ >= circ_min) & (d >= 0.5 * mediana) & (d <= 2 * mediana)
    mediana_movil = d.copy()
    for i in range(len(d)):
        for mitad in (int(medicion['fps'] / 4), int(medicion['fps'])): #si no hay vecinos validos cerca (ej: despues
            vecinos = [d[j] for j in range(max(0, i - mitad), min(len(d), i + mitad + 1)) if j != i and previos[j]]
            if len(vecinos) >= 3:                                       #de un hueco largo), se buscan hasta 1 s
                mediana_movil[i] = np.median(vecinos)
                break
    h, w = medicion['forma_roi']
    cx, cy = np.nan_to_num(medicion['centroide_x']), np.nan_to_num(medicion['centroide_y'])
    #pupila cerca del borde de la ROI: a los costados hace falta lugar para el iris (2 radios); arriba y abajo
    #alcanza con que la pupila entre completa (1 radio), porque ahi igual estan los parpados
    en_borde = (np.minimum(cx, w - 1 - cx) < d) | (np.minimum(cy, h - 1 - cy) < d / 2)
    paso = np.hypot(np.diff(medicion['centroide_x']), np.diff(medicion['centroide_y']))
    paso = np.nan_to_num(paso, nan=np.inf)
    desplazamiento = np.minimum(np.r_[np.inf, paso], np.r_[paso, np.inf]) #distancia al frame anterior Y al siguiente
    motivos = {
        #parpadeo / ojo cerrado: no hay pupila, o el parpado la tapa en parte y deja de ser redonda
        'parpadeo u oclusion': ~medicion['encontrada'] | (circ < circ_min),
        #ojo fuera de cuadro o segmentacion erronea: tamaño muy distinto al habitual del video
        'tamaño anomalo': (d < 0.5 * mediana) | (d > 2 * mediana),
        #movimiento ocular brusco: el centro de la pupila esta a mas de medio radio del frame anterior y del siguiente
        #(un frame borroso en medio de un movimiento rapido, o una deteccion suelta en otro lugar)
        'movimiento ocular': desplazamiento > 0.5 * np.maximum(d / 2, 1),
        #ruido de segmentacion: el diametro salta mas de un 20% respecto de los frames vecinos
        'salto brusco': np.abs(d - mediana_movil) > salto_max * mediana_movil,
        #el ojo se movio hasta el borde de la ROI: parte del iris queda afuera y la segmentacion no es confiable
        'pupila en el borde de la ROI': en_borde,
    }
    ok = ~np.any(list(motivos.values()), axis=0)
    #margen: los frames pegados a un artefacto (~0.1 s antes y despues) suelen estar "medio contaminados"
    #(el parpado empezando a bajar, el ojo terminando de moverse), asi que tambien se descartan
    m = max(1, int(0.1 * medicion['fps']))
    margen = ok & ~(cv2.erode(ok.astype(np.uint8).reshape(1, -1), np.ones((1, 2 * m + 1), np.uint8),
                              borderType=cv2.BORDER_REPLICATE).ravel().astype(bool))
    motivos['margen alrededor de un artefacto'] = margen
    ok &= ~margen
    #tramos de frames "buenos" de menos de medio segundo en medio de una zona mala tambien se descartan
    tramo_corto = np.zeros_like(ok)
    i = 0
    while i < len(ok):
        j = i
        while j < len(ok) and ok[j] == ok[i]:
            j += 1
        tramo_corto[i:j] = ok[i] and j - i < max(1, int(medicion['fps'] / 2))
        i = j
    motivos['tramo corto entre artefactos'] = tramo_corto
    ok &= ~tramo_corto
    return (ok, motivos) if devolver_motivos else ok


def anotar_frame_manual(roi_rgb, titulo):
    #1.b) Se muestra la ROI agrandada (sin el contorno automatico, para no influir) y se hace click en al menos
    #5 puntos del borde de la pupila. Se ajusta una elipse (cv2.fitEllipse) y se calcula area y diametro equivalente.
    #Teclas: ENTER = aceptar | r = borrar puntos | s = saltear frame | ESC = terminar
    h, w = roi_rgb.shape[:2]
    escala = max(1.0, 800.0 / max(h, w))
    base = cv2.resize(cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2BGR), (int(w * escala), int(h * escala)), interpolation=cv2.INTER_CUBIC)
    puntos = []
    cv2.namedWindow(titulo, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(titulo, lambda ev, x, y, *_: puntos.append((x / escala, y / escala)) if ev == cv2.EVENT_LBUTTONDOWN else None)
    while True:
        vista = base.copy()
        for px, py in puntos:
            cv2.circle(vista, (int(px * escala), int(py * escala)), 3, (0, 255, 0), -1)
        if len(puntos) >= 5: #con 5 puntos ya se puede ajustar una elipse: la mostramos
            (cx, cy), (e1, e2), ang = cv2.fitEllipse(np.array(puntos, np.float32))
            cv2.ellipse(vista, ((cx * escala, cy * escala), (e1 * escala, e2 * escala), ang), (0, 255, 255), 1)
        cv2.putText(vista, "Marca el borde de la PUPILA (circulo negro del centro), NO el del iris",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(vista, f"Puntos: {len(puntos)} | ENTER aceptar | r borrar | s saltear | ESC terminar",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow(titulo, vista)
        tecla = cv2.waitKey(20) & 0xFF
        if tecla in (13, 10) and len(puntos) >= 5:
            cv2.destroyWindow(titulo)
            _, (e1, e2), _ = cv2.fitEllipse(np.array(puntos, np.float32))
            area = np.pi * (e1 / 2) * (e2 / 2)
            return {'area_px': area, 'diametro_px': 2 * np.sqrt(area / np.pi)} #mismo criterio que el automatico
        if tecla == ord('r'):
            puntos.clear()
        if tecla in (ord('s'), 27):
            cv2.destroyWindow(titulo)
            return None if tecla == ord('s') else 'terminar'


def anotaciones_manuales(config, medicion, nombre_video):
    #1.b) Reutiliza las anotaciones guardadas o anota a mano n frames repartidos en el video (solo frames confiables)
    ruta = os.path.join(CARPETA_RESULTADOS, f"{nombre_video}_anotaciones_manuales.csv")
    if os.path.exists(ruta) and input("1.b) Hay anotaciones manuales guardadas. ¿Usarlas? (S/n): ").strip().lower() != 'n':
        with open(ruta, newline='', encoding='utf-8') as f:
            return {int(r['frame']): {'area_px': float(r['area_px']), 'diametro_px': float(r['diametro_px'])} for r in csv.DictReader(f)}
    resp = input("1.b) ¿Cuántos frames querés anotar a mano? [ENTER = 10]: ").strip()
    n = int(resp) if resp.isdigit() and int(resp) >= 2 else 10
    validos = np.flatnonzero(frames_confiables(medicion))
    elegidos = validos[np.linspace(0, len(validos) - 1, n).astype(int)]
    print("     Hacé click en al menos 5 puntos del borde de la pupila y ENTER (r borrar, s saltear, ESC terminar).")
    anot = {}
    for k, fr in enumerate(elegidos):
        #se muestra la zona alrededor de la pupila (4 radios) agrandada, para que el circulo negro se vea grande
        #(el area y el diametro no dependen de donde este el recorte)
        roi_rgb = cargar_frame_y_roi(config['video_path'], int(fr), config['roi'])[1]
        x, y, r = medicion['centroide_x'][fr], medicion['centroide_y'][fr], medicion['diametro_px'][fr] / 2
        lado = int(max(4 * r, 40))
        recorte = roi_rgb[max(int(y) - lado, 0):int(y) + lado, max(int(x) - lado, 0):int(x) + lado]
        r = anotar_frame_manual(recorte, f"Anotacion manual {k + 1}/{len(elegidos)} - frame {fr}")
        if r == 'terminar':
            break
        if r is not None:
            anot[int(fr)] = r
    os.makedirs(CARPETA_RESULTADOS, exist_ok=True)
    with open(ruta, 'w', newline='', encoding='utf-8') as f:
        e = csv.writer(f)
        e.writerow(['frame', 'area_px', 'diametro_px'])
        e.writerows([fr, f"{a['area_px']:.1f}", f"{a['diametro_px']:.3f}"] for fr, a in sorted(anot.items()))
    print(f"     Anotaciones guardadas en: {ruta}")
    return anot


def consigna_4_validacion(config):
    print("\n PARTE 3.2 - Consigna 1: Segmentación automática frame a frame y validación")
    medicion = obtener_medicion(config)
    nombre_video = os.path.splitext(os.path.basename(config['video_path']))[0]
    print(f"1.a) Pupila encontrada en {medicion['encontrada'].sum()} de {len(medicion['frame'])} frames | "
          f"Diámetro: media {np.nanmean(medicion['diametro_px']):.1f} px, mín {np.nanmin(medicion['diametro_px']):.1f}, "
          f"máx {np.nanmax(medicion['diametro_px']):.1f}")
    print(f"     Medición guardada en: {guardar_curva_csv(medicion, nombre_video)}")

    anot = anotaciones_manuales(config, medicion, nombre_video)
    if len(anot) < 2:
        print("[AVISO] Hacen falta al menos 2 frames anotados para calcular MAE/RMSE.")
        return
    #1.c) MAE = promedio de |automatico - manual| ; RMSE = raiz del promedio de (automatico - manual)^2
    #     (penaliza mas los errores grandes) ; sesgo = promedio de (automatico - manual)
    frames = np.array(sorted(anot))
    res = {}
    for clave in ('diametro_px', 'area_px'):
        manual = np.array([anot[f][clave] for f in frames])
        auto = medicion[clave][frames]
        res[clave] = (manual, auto, np.mean(np.abs(auto - manual)), np.sqrt(np.mean((auto - manual) ** 2)))
    d_man, d_aut = res['diametro_px'][:2]
    print(f"1.c) Validación en {len(frames)} frames anotados:")
    print(f"     Diámetro: MAE = {res['diametro_px'][2]:.2f} px | RMSE = {res['diametro_px'][3]:.2f} px | "
          f"error relativo = {100 * np.mean(np.abs(d_aut - d_man) / d_man):.1f} % | sesgo = {np.mean(d_aut - d_man):+.2f} px")
    print(f"     Área:     MAE = {res['area_px'][2]:.0f} px² | RMSE = {res['area_px'][3]:.0f} px²")

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    fig.canvas.manager.set_window_title("Parte 3.2 - Consigna 1: Validación automático vs. manual")
    for a, (clave, unidad) in zip(ax, [('diametro_px', 'Diámetro [px]'), ('area_px', 'Área [px²]')]):
        manual, auto, mae, rmse = res[clave]
        lim = [min(manual.min(), auto.min()) * 0.9, max(manual.max(), auto.max()) * 1.1]
        a.plot(lim, lim, 'k--', lw=1, label='Automático = manual')
        a.plot(manual, auto, 'o', ms=7, label='Frames anotados')
        for fr, x, y in zip(frames, manual, auto):
            a.annotate(str(fr), (x, y), textcoords='offset points', xytext=(4, 4), fontsize=7)
        a.set_xlabel(f"{unidad} manual"); a.set_ylabel(f"{unidad} automático")
        a.set_title(f"1.c) {unidad.split()[0]}: MAE = {mae:.2f} | RMSE = {rmse:.2f}")
        a.legend(); a.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# =============================================================================
# PARTE 3.2 - CONSIGNA 2: CURVA TEMPORAL 1D DEL DIAMETRO D(t) Y DEL AREA A(t)
# =============================================================================

def ajustar_escala(ax, valores):
    #Escala del eje Y ajustada a los frames confiables (si no, los artefactos extremos aplastan la curva)
    lo, hi = np.nanmin(valores), np.nanmax(valores)
    ax.set_ylim(lo - 0.3 * (hi - lo), hi + 0.3 * (hi - lo))


def consigna_5_curva(config):
    print("\n PARTE 3.2 - Consigna 2: Curva temporal 1D D(t) y A(t)")
    medicion = obtener_medicion(config)
    t, ok = medicion['tiempo_s'], frames_confiables(medicion)
    print(f"     Duración: {t[-1]:.1f} s ({len(ok)} frames a {medicion['fps']:.1f} fps) | "
          f"frames confiables: {ok.sum()} ({100 * ok.mean():.1f} %)")
    fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    fig.canvas.manager.set_window_title("Parte 3.2 - Consigna 2: Curva temporal")
    d_limpia, a_limpia, _ = limpiar_curva(medicion)
    for a, (clave, etiqueta, limpia) in zip(ax, [('diametro_px', 'Diámetro D(t) [px]', d_limpia), ('area_px', 'Área A(t) [px²]', a_limpia)]):
        a.plot(t[ok], medicion[clave][ok], '.', ms=3, color='0.55', label='Frames confiables (medición cruda)')
        a.plot(t, limpia, '-', lw=2, color='tab:blue', label='Curva sin artefactos (interpolada y filtrada, ver consigna 4)')
        a.plot(t[~ok], medicion[clave][~ok], 'x', ms=5, color='tab:red', label='Frames no confiables (artefactos)')
        ajustar_escala(a, medicion[clave][ok])
        a.set_ylabel(etiqueta); a.grid(alpha=0.3)
    ax[0].set_title(f"Curva temporal de la pupila - {os.path.basename(config['video_path'])}")
    ax[0].legend(fontsize=8)
    ax[1].set_xlabel("Tiempo [s]")
    plt.tight_layout()
    plt.show()


# =============================================================================
# PARTE 3.2 - CONSIGNA 3: PARAMETROS FISIOLOGICOS
# =============================================================================

def limpiar_curva(medicion):
    #Metodo de correccion de artefactos (ver consigna 4): los frames no confiables se reemplazan por interpolacion
    #lineal. Despues se aplica un filtro de mediana (~0.3 s), que elimina los saltos de uno o dos frames de la
    #segmentacion sin redondear los cambios rapidos reales, y un filtro Savitzky-Golay (~0.3 s) que suaviza el resto
    ok = frames_confiables(medicion)
    t = medicion['tiempo_s']
    ventana = int(0.3 * medicion['fps']) // 2 * 2 + 1
    return [savgol_filter(medfilt(np.interp(t, t[ok], medicion[c][ok]), ventana), ventana, 2)
            for c in ('diametro_px', 'area_px')] + [ok]


def detectar_estimulo(video_path, ok, fps):
    #Si se prende/apaga una luz, el estimulo es el mayor salto de brillo del video.
    #Si no hay un salto claro (ej: el estimulo es abrir el ojo), se toma el primer frame confiable
    cap = cv2.VideoCapture(video_path)
    brillo = []
    while True:
        leido, frame = cap.read()
        if not leido:
            break
        brillo.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean())
    cap.release()
    #Cambio de brillo en ventanas de ~0.25 s (la camara ajusta la exposicion, asi que el cambio puede no ser instantaneo)
    brillo = medfilt(np.array(brillo), 3)
    k = max(1, int(0.25 * fps))
    cambio = brillo[k:] - brillo[:-k]
    #solo se buscan cambios mientras se ve el ojo (entre el primer y el ultimo frame confiable): al principio o al
    #final del video la camara suele moverse y el brillo cambia sin que haya un estimulo
    validos = np.flatnonzero(ok)
    cambio[:max(validos[0] - k, 0)] = 0
    cambio[max(validos[-1] - k, 0):] = 0
    i = int(np.argmax(np.abs(cambio)))
    if abs(cambio[i]) > 15: #cambio claro (en niveles de gris): el estimulo es el frame donde el brillo cambia mas rapido
        j = i + int(np.argmax(np.abs(np.diff(brillo[i:i + k + 1]))))
        return (j + 1) / fps, "cambio de brillo (luz " + ("prendida)" if cambio[i] > 0 else "apagada)")
    if np.argmax(ok) > 0:
        return np.argmax(ok) / fps, "apertura del ojo (primer frame confiable)"
    return 0.0, "NO se detectó estímulo: ni cambio de luz ni apertura del ojo. Si lo conocés, ingresalo a mano"


def parametros_fisiologicos(t, d, a, t_est):
    #3.a) Basal: promedio del segundo previo al estimulo (si el ojo estaba cerrado, los primeros 0.2 s despues)
    basal = (t >= t_est - 1) & (t < t_est)
    if basal.sum() < 3:
        basal = (t >= t_est) & (t < t_est + 0.2)
    d_basal, a_basal = d[basal].mean(), a[basal].mean()
    #3.b) Amplitud: diferencia entre el basal y el extremo de la respuesta en los 5 s posteriores
    #     (si la pupila se achica es una contraccion -registro A-, si se agranda una redilatacion -registro B-)
    post = np.flatnonzero((t >= t_est) & (t <= t_est + 5))
    i_min, i_max = post[np.argmin(d[post])], post[np.argmax(d[post])]
    contraccion = (d_basal - d[i_min]) >= (d[i_max] - d_basal)
    i_ext = i_min if contraccion else i_max
    amplitud = abs(d[i_ext] - d_basal)
    #3.c) Latencia: desde el estimulo hasta que el diametro se aparta del basal un 5% de la amplitud
    i_inicio = post[np.argmax(np.abs(d[post] - d_basal) >= 0.05 * amplitud)]
    #3.d) Velocidades (derivada del diametro): pico de contraccion antes del minimo y de redilatacion despues
    v = np.gradient(d, t)
    return {'tipo': 'A (contraccion)' if contraccion else 'B (redilatacion)',
            't_estimulo_s': t_est, 'latencia_s': t[i_inicio] - t_est,
            'diametro_basal_px': d_basal, 'diametro_extremo_px': d[i_ext], 't_extremo_s': t[i_ext],
            'amplitud_diametro_px': amplitud, 'amplitud_area_px': abs(a[i_ext] - a_basal),
            'vel_contraccion_px_s': -v[post[0]:i_min + 1].min() if contraccion else np.nan,
            'vel_redilatacion_px_s': v[i_min:].max() if contraccion else v[post[0]:i_max + 1].max()}


def guardar_parametros(nombre_video, p):
    #3.e) Los parametros de cada registro se guardan en un CSV comun (se reemplaza la fila si el video ya estaba)
    #     para calcular la variabilidad entre registros (trials)
    os.makedirs(CARPETA_RESULTADOS, exist_ok=True)
    ruta = os.path.join(CARPETA_RESULTADOS, "parametros_registros.csv")
    filas = []
    if os.path.exists(ruta):
        with open(ruta, newline='', encoding='utf-8') as f:
            filas = [r for r in csv.DictReader(f) if r['video'] != nombre_video]
    filas.append({'video': nombre_video, **{k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in p.items()}})
    with open(ruta, 'w', newline='', encoding='utf-8') as f:
        e = csv.DictWriter(f, fieldnames=list(filas[-1].keys()))
        e.writeheader()
        e.writerows(filas)
    return ruta, filas


def consigna_6_parametros(config):
    print("\n PARTE 3.2 - Consigna 3: Parámetros fisiológicos")
    medicion = obtener_medicion(config)
    nombre_video = os.path.splitext(os.path.basename(config['video_path']))[0]
    t = medicion['tiempo_s']
    d, a, ok = limpiar_curva(medicion)
    t_est, criterio = detectar_estimulo(config['video_path'], ok, medicion['fps'])
    resp = input(f"     Estímulo detectado en t = {t_est:.2f} s ({criterio}). ENTER para aceptar o ingresá otro tiempo [s]: ").strip()
    if resp:
        t_est = float(resp.replace(',', '.'))

    p = parametros_fisiologicos(t, d, a, t_est)
    print(f"     Tipo de respuesta: {p['tipo']}")
    print(f"3.a) Diámetro basal = {p['diametro_basal_px']:.1f} px")
    print(f"3.b) Amplitud = {p['amplitud_diametro_px']:.1f} px (diámetro) | {p['amplitud_area_px']:.0f} px² (área)")
    print(f"3.c) Tiempo de latencia = {1000 * p['latencia_s']:.0f} ms")
    sin_medir = 1 - ok[(t >= t_est) & (t <= t_est + 0.5)].mean() #fraccion de artefactos justo despues del estimulo
    if sin_medir > 0.5:
        print(f"     [AVISO] La latencia no es confiable: el {100 * sin_medir:.0f} % de los frames en los 0.5 s posteriores")
        print("     al estímulo son artefactos (ej: la cámara ajustando la exposición), así que esa parte de la curva es interpolada.")
    print(f"3.d) Velocidad de contracción = {p['vel_contraccion_px_s']:.1f} px/s | redilatación = {p['vel_redilatacion_px_s']:.1f} px/s")
    ruta, filas = guardar_parametros(nombre_video, p)
    print(f"     Curva guardada en: {guardar_curva_csv(medicion, nombre_video, d, a)} | Parámetros en: {ruta}")
    filas = [f for f in filas if f['tipo'] == p['tipo']] #variabilidad solo entre registros del mismo tipo
    if len(filas) >= 2:
        print(f"3.e) Variabilidad entre {len(filas)} registros {p['tipo']} (media ± desvío):")
        for clave, unidad in [('latencia_s', 's'), ('amplitud_diametro_px', 'px')]:
            v = np.array([float(f[clave]) for f in filas])
            print(f"     {clave}: {v.mean():.3f} ± {v.std(ddof=1):.3f} {unidad}")
    else:
        print("3.e) Para la variabilidad entre registros, procesá al menos 2 videos del mismo tipo.")

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.canvas.manager.set_window_title("Parte 3.2 - Consigna 3: Parámetros fisiológicos")
    ax.plot(t[ok], medicion['diametro_px'][ok], '.', ms=3, color='0.6', label='D(t) medido')
    ax.plot(t, d, '-', lw=2, color='tab:blue', label='D(t) interpolado y suavizado')
    ax.axvline(t_est, color='orange', lw=2, label='Estímulo')
    ax.axvline(t_est + p['latencia_s'], color='tab:purple', ls='--', label=f"Inicio de la respuesta (latencia {1000 * p['latencia_s']:.0f} ms)")
    ax.axhline(p['diametro_basal_px'], color='k', ls=':', label='Diámetro basal')
    ax.plot(p['t_extremo_s'], p['diametro_extremo_px'], 'v', ms=10, color='tab:green', label=f"Extremo (amplitud {p['amplitud_diametro_px']:.1f} px)")
    ax.set_xlabel("Tiempo [s]"); ax.set_ylabel("Diámetro [px]"); ax.set_title(f"Registro {p['tipo']}")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# =============================================================================
# PARTE 3.2 - CONSIGNA 4: FUENTES DE ARTEFACTOS Y METODO DE FILTRADO
# =============================================================================
#Metodo implementado:
# 1) DETECCION (frames_confiables): un frame es artefacto si cumple alguno de los criterios: parpadeo u oclusion,
#    tamaño anomalo, movimiento ocular brusco, salto brusco o pupila en el borde de la ROI; ademas se descarta un margen de ~0.1 s alrededor de
#    cada artefacto y los tramos cortos de frames buenos entre artefactos.
# 2) CORRECCION (limpiar_curva): esos frames se reemplazan por interpolacion lineal entre los frames confiables
#    vecinos y la curva se filtra con una mediana (~0.3 s) y un Savitzky-Golay (~0.3 s).

def consigna_7_artefactos(config):
    print("\n PARTE 3.2 - Consigna 4: Fuentes de artefactos y método de filtrado")
    medicion = obtener_medicion(config)
    t = medicion['tiempo_s']
    ok, motivos = frames_confiables(medicion, devolver_motivos=True)
    d_limpia = limpiar_curva(medicion)[0]
    print(f"     Frames descartados: {(~ok).sum()} de {len(ok)} ({100 * (~ok).mean():.1f} %). Por fuente (un frame puede tener varias):")
    for motivo, m in motivos.items():
        print(f"       - {motivo:<34} {m.sum():>4} frames")

    #Agrupamos los frames malos consecutivos en episodios, con el motivo mas frecuente de cada uno
    episodios, i = [], 0
    while i < len(ok):
        j = i
        while j < len(ok) and ok[j] == ok[i]:
            j += 1
        if not ok[i]:
            reales = [m for m in motivos if m not in ('margen alrededor de un artefacto', 'tramo corto entre artefactos')]
            conteo = {m: motivos[m][i:j].sum() for m in reales}
            principal = max(conteo, key=conteo.get) if max(conteo.values()) > 0 else max(motivos, key=lambda m: motivos[m][i:j].sum())
            episodios.append((t[i], t[j - 1], j - i, principal))
        i = j
    print(f"     Episodios de artefacto ({len(episodios)}):")
    for ini, fin, n, motivo in episodios:
        print(f"       {ini:6.2f} s - {fin:6.2f} s  ({n:>3} frames)  {motivo}")
    print("     Método: los frames marcados se descartan y se reemplazan por interpolación lineal entre los")
    print("     frames confiables vecinos; luego la curva se filtra con una mediana y un Savitzky-Golay (~0.3 s cada uno).")

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.canvas.manager.set_window_title("Parte 3.2 - Consigna 4: Artefactos")
    for k, (ini, fin, _, _) in enumerate(episodios):
        ax.axvspan(ini, fin + 1 / medicion['fps'], color='tab:red', alpha=0.15, label='Episodio de artefacto' if k == 0 else None)
    ax.plot(t, medicion['diametro_px'], '.', ms=3, color='0.6', label='D(t) crudo (todos los frames)')
    ax.plot(t, d_limpia, '-', lw=2, color='tab:blue', label='D(t) sin artefactos (interpolado + suavizado)')
    ajustar_escala(ax, medicion['diametro_px'][ok])
    ax.set_xlabel("Tiempo [s]"); ax.set_ylabel("Diámetro [px]")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# =============================================================================
# FUNCION PRINCIPAL DEL MODULO 3
# =============================================================================

def ejecutar_modulo_3(config):
    print("\n==================================================")
    print("  EJECUTANDO MÓDULO 3: CARACTERÍSTICAS Y CURVA TEMPORAL")
    print("==================================================")
    opciones = {
        "1": ("Parte 3.1 - Consigna 1: Descriptores morfométricos", consigna_1_morfometria),
        "2": ("Parte 3.1 - Consigna 2: Textura de primer orden", consigna_2_primer_orden),
        "3": ("Parte 3.1 - Consigna 3: Textura de segundo orden (GLCM)", consigna_3_glcm),
        "4": ("Parte 3.2 - Consigna 1: Segmentación frame a frame y validación (MAE/RMSE)", consigna_4_validacion),
        "5": ("Parte 3.2 - Consigna 2: Curva temporal D(t) y A(t)", consigna_5_curva),
        "6": ("Parte 3.2 - Consigna 3: Parámetros fisiológicos", consigna_6_parametros),
        "7": ("Parte 3.2 - Consigna 4: Fuentes de artefactos y método de filtrado", consigna_7_artefactos),
    }
    while True:
        print("\n--------------------------------------------------")
        print(f" Frame de trabajo (Parte 3.1): {config.get('frame_idx', 0)}")
        for num, (texto, _) in opciones.items():
            print(f"{num}. {texto}")
        print("0. Volver al menú principal")
        opcion = input("Seleccione una opción: ").strip()
        if opcion == "0":
            return
        if opcion in opciones:
            opciones[opcion][1](config)
        else:
            print("Opción inválida.")