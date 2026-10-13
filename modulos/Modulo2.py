#Modulo 2: Segmentacion y morfologia matematica de la pupila
#Modulo 2.1: Algoritmos de segmentacion
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import active_contour
from skimage.filters import gaussian
from skimage.segmentation import watershed
from skimage.filters import sobel

def lectura_frame(video_path, num_frame=0, roi_coords=None):
    cap = cv2.VideoCapture(video_path) #abre el archivo desde el path dado
    cap.set(cv2.CAP_PROP_POS_FRAMES, num_frame) #define el frame seleccionado por el usuario dentro de los del video
    ok, frame_bgr = cap.read()  #devuelve un array donde cada posición tiene tres números: BGR por convención de CV2
    cap.release()
    assert ok, f"No se pudo leer el frame {num_frame}, revisá video_path" #hago un STOP que si da error ok, NO pueda seguir
    
    frame_rgb=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB) #reordena los colores de BGR a RGB para el procesamiento
    
    if roi_coords is not None:
        ymin, ymax, xmin, xmax = roi_coords #como yo le doy roi_coords, separo esa info en las cuatro coordenadas concretas
        roi_rgb = frame_rgb[ymin:ymax, xmin:xmax] #toma los píxeles que hay entre las 4 coordenadas
    else:
        roi_rgb = frame_rgb
        
    return frame_rgb, roi_rgb

def otsu(roi_rgb):
    #Para hacer el cálculo con Otsu, necesito la imagen en escala de grises para poder umbralizar. Por eso gamma todo a esos valores
    grises = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2GRAY) #función que pasa de rgb a grises
    
    umbral, mascara = cv2.threshold(grises, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU) #uso la función que devuelve la máscara de Otsu y el valor del umbral que separa al histograma
    print(f"Umbral de Otsu: {umbral}")

    fig, ax = plt.subplots(1, 4, figsize=(16, 4))
    ax[0].imshow(roi_rgb); ax[0].set_title("ROI") #visualizo la ROI original en color
    ax[1].imshow(grises, cmap="gray"); ax[1].set_title("ROI en escala de grises") #visualizo la ROI original en gris
    ax[2].hist(grises.ravel(), bins=256); ax[2].axvline(umbral, color="r"); ax[2].set_title("Histograma + umbral") #grafio el histograma con el umbral calculado
    ax[3].imshow(mascara, cmap="gray"); ax[3].set_title("Máscara Otsu") #visualizo la máscara final donde se suone que la pupila queda blanca y el iris negro
    for a in (ax[0], ax[1], ax[3]): a.axis("off")
    plt.show()
    
    return mascara, umbral

def kmeans(roi_rgb):
    cv2.setRNGSeed(0)   #pongo la misma semilla siempre en 0 para no tener problema si se runnea más de una vez el código de forma seguida

    #como tengo que separar por croma e intensidad, separo en L (brillo) A/B (croma)
    lab = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2LAB) #función nativa que transforma los números de cada píxel de RGB a LAB

    datos = lab.reshape(-1, 3).astype(np.float32) #reestrcturo. es una matriz de la cantidad de filas de píxeles de la ROI, y tres columnas: L, A y B

    criterio = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1) #defino un criterio de cuándo el algoritmo para. digo que pare luego de 100 iteraciones o cuando los centroides de cada grupo se muevan menos de 0.1 entre iteraaciones
    
    fig, ax = plt.subplots(2, 4, figsize=(16, 7))
    mascaras = {}

    for i, K in enumerate([2, 3, 4, 5]): #le deino un vector con los 4 Ks a probar (grupos en los que se separará)
        _, etiquetas, centros = cv2.kmeans(datos, K, None, criterio, 10, cv2.KMEANS_PP_CENTERS) #función de CV2 que le gamma la matriz LAB, el valor de K a separar, el criterio a usar y le pido que con KMEANS_PP_CENTRES elija bien los centros para que sean aleatorios pero NO queden muy juntos. quiero que estén separados para que no ponga cercanos valores de pupila o riis que definitivamente son distintos
        etiquetas = etiquetas.reshape(roi_rgb.shape[:2]) #define a qué grupo pertenece cada píxel

        #separo por intesidad (L), por lo que tengo que ordenarlos y elegir el mínimo para poder distinguirlos por ese brillo
        k_pupila = np.argmin(centros[:, 0])
        mask_k = (etiquetas == k_pupila).astype(np.uint8) * 255 #arma la máscara para ir multiplicando y ponendo los píxeles en cada grupo
        mascaras[K] = mask_k
        #hago las filas y columnas con los distintos grupos
        ax[0, i].imshow(etiquetas, cmap="viridis"); ax[0, i].set_title(f"Clusters K={K}")
        ax[1, i].imshow(mask_k, cmap="gray");       ax[1, i].set_title(f"Máscara pupila K={K}")
        ax[0, i].axis("off"); ax[1, i].axis("off")

    plt.tight_layout()
    plt.show()
    
    return mascaras

def snakes(roi_rgb):
    lab = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2LAB)
    
    datos = lab.reshape(-1, 3).astype(np.float32)
    criterio = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1)
    
    #quiero definir una badna inicial aproximada a la pupila, por eso elijo el Kmeans de 4 pra tener una región aproximada
    K = 4
    _, etiquetas, centros = cv2.kmeans(datos, K, None, criterio, 10, cv2.KMEANS_PP_CENTERS)
    etiquetas = etiquetas.reshape(roi_rgb.shape[:2])
    k_pupila = np.argmin(centros[:, 0])
    mask_k = (etiquetas == k_pupila).astype(np.uint8) * 255
    #Una vez que hice todo el kmeans de 4, le calculo para usar la snake

    n, lab_cc, stats, cents = cv2.connectedComponentsWithStats(mask_k)
    #lo que hace esta funcion es fijarse "manchas" que tiene la máscara. Yo quiero quedarme solo con la pupila para que el snake vaya por ella, entonces al tener la mascara binaria, tomo el fondo (0) como la mancha 0, y después
    #donde haya otros números, son manchas aisladas. Con esas "zonas"/"manchas", defino su área teniendo en cuenta bordes y los píxeles dentro, y me fijo la más grande que debería ser la pupila. De esta forma, me aseguro de tomar el centro de ella
    #para la snake y no otras "manchas" como podrían ser el flash reflejado u pestañas en el medio, etc.
    h, w = mask_k.shape #defino el tamaño de la matriz para poder ver los centros
    centro_roi = np.array([w / 2, h / 2])

    #ahora hace el puntaje de cada una de las zonas para ver cuál es la mejor (pupila)
    mejor, mejor_score = None, -1
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA] #calcula el área de cada mancha
        dist = np.linalg.norm(cents[i] - centro_roi)
        score = area / (1 + dist) #el puntaje depende de qué tn grande sea la zona y qué tan cerca del centro. se spupone qe la pupila es el centro de la ROI total entonces la distancia sería chica
        if score > mejor_score:
            mejor, mejor_score = i, score

    mask_pupila = (lab_cc == mejor).astype(np.uint8) * 255 #defino entonces la máscara según el mejor puntaje

    gris = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2GRAY) #gamma todo a escala de grises

    #llevo todo a normalización para que la snake vea bien los bordes 
    img = gaussian(gris, sigma=2, preserve_range=True) / 255.0

    #defino la snake
    M = cv2.moments(mask_pupila) #esta función da números de la forma del snake
    cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"] #calculo el promedio de esos números que sería el centro
    radio_pupila = np.sqrt(mask_pupila.astype(bool).sum() / np.pi) #defino el radio
    radio_ini = radio_pupila * 1.5  #lo multiplico por 1.5 para que esté por fuera de la pupila y se vaya achicando en las iteraciones

    t = np.linspace(0, 2 * np.pi, 200) #armo el vector de los ángulos de ese ciruclo de la snake de 200 puntos
    init = np.column_stack([cy + radio_ini * np.sin(t),    
        cx + radio_ini * np.cos(t)])  #uso las coordenadas radiales para definir la snake con los valores de los ángulos 

    #defino los parámetros de elastcidad y beta para que se vaya cerrando la snake
    snake = active_contour(img, init,
        alpha=0.015,    #elasticidad
        beta=10,       #rigidez
        gamma=0.001,    #paso
        w_line=0,
        w_edge=1,       
        max_num_iter=2500)

    # Armo máscara binaria a partir de la snake (para usarla después en Dice/IoU) --> pongo el interior (pupila) blanco y el exterior negro
    pts = np.round(snake[:, ::-1]).astype(np.int32)  
    mask_snake = np.zeros_like(mask_pupila)
    cv2.fillPoly(mask_snake, [pts], 255)

    fig, ax = plt.subplots(1, 3, figsize=(14, 5))
    ax[0].imshow(roi_rgb) #pongo la imagen en color con la snake inicial sobre ella
    ax[0].plot(init[:, 1], init[:, 0], "--r", lw=1.5)
    ax[0].set_title("Contorno inicial (rojo)")
    ax[1].imshow(roi_rgb)
    ax[1].plot(snake[:, 1], snake[:, 0], "-g", lw=2)
    ax[1].set_title("Contorno final (verde)") #pongo la snake final con la imagen 
    ax[2].imshow(mask_snake, cmap="gray"); ax[2].set_title("Máscara de la snake") #lo pongo en byn
    for a in ax: a.axis("off")
    plt.show()
    
    return mask_snake, mask_pupila

def watershed_pupila(roi_rgb, mask_pupila):

    gris = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2GRAY)

    gradiente = sobel(gris.astype(float) / 255.0) #aplico un filtro sobel para calcular gradiente y ver donde están los bordes

    
    radio = np.sqrt((mask_pupila > 0).sum() / np.pi) #se fija el radio de la pupila con la mascara que saque del snake, es la misma
    # Centro de la pupila
    M = cv2.moments(mask_pupila)
    cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]


    marcadores = np.ones(gris.shape, dtype=np.int32) #hago una matriz toda de unos paa despues ir discriminando entre pupila o NO pupila
    cv2.circle(marcadores, (int(cx), int(cy)), int(1.5 * radio), 0, -1) #defino un círculo un poco más grande de la pupila
    cv2.circle(marcadores, (int(cx), int(cy)), int(0.5 * radio), 2, -1) #defino otro círculo un poco más chico que la pupila

    #se supone que en la zona que entran esos dos dos círculos, es seguro pupila así que asigna 2. Entre medio, pone 0 porque es lo que watershed definirá. Fuera del grande es 1 porque no es pupila


    etiquetas_ws = watershed(gradiente, marcadores) #la función va "llenando" las zonas donde tiene que decidir si es pupila o no. recibe el gradiente (indica los picos) y marcadores (donde hay 2, 1 y 0)
    mask_ws = (etiquetas_ws == 2).astype(np.uint8) * 255 #define las etiquetas con la máscara donde hay 2 que es pupila. ya no hay mas 0 porque definió el algoritmo

    #plotting
    fig, ax = plt.subplots(1, 4, figsize=(16, 4))
    ax[0].imshow(gradiente, cmap="gray");  ax[0].set_title("Gradiente (relieve)")
    ax[1].imshow(marcadores, cmap="viridis"); ax[1].set_title("Marcadores")
    ax[2].imshow(mask_ws, cmap="gray");    ax[2].set_title("Máscara Watershed")
    ax[3].imshow(roi_rgb)
    ax[3].contour(mask_ws, colors="lime", linewidths=1)
    ax[3].set_title("Contorno sobre la ROI")
    for a in ax: a.axis("off")
    plt.show()
    
    return mask_ws
    
def edac(mascara):
    #Erosionar
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)) #defino el Kernel de 5x5 
    erosion = cv2.erode(mascara, kernel) #le aplico el kernel a la máscara para erosionarlo
    
    #Dilatación
    dilatacion = cv2.dilate(mascara, kernel) #le aplico el kernel a la máscara 
    
    #apertura
    apertura = cv2.dilate(erosion, kernel) #podría hacerse con función nativa pero ya que armé lo otro, lo uso
    
    #cierre
    cierre = cv2.erode(dilatacion, kernel) #idem anterior
    
    fig, ax = plt.subplots(1, 5, figsize=(18, 4))
    ax[0].imshow(mascara, cmap="gray");    ax[0].set_title("Original")
    ax[1].imshow(erosion, cmap="gray");    ax[1].set_title("Erosión")
    ax[2].imshow(dilatacion, cmap="gray"); ax[2].set_title("Dilatación")
    ax[3].imshow(apertura, cmap="gray");   ax[3].set_title("Apertura")
    ax[4].imshow(cierre, cmap="gray");     ax[4].set_title("Cierre")
    for a in ax:
        a.axis("off")
    plt.show()
    
    return erosion, dilatacion, apertura, cierre

    
    

def ejecutar_modulo_2(config):

    print("   EJECUTANDO MÓDULO 2: SEGMENTACIÓN Y MORFOLOGÍA DE LA PUPILA ")
    print("--------------------------------------------------------")
    print("Parte 2.1")
    # Extrae parámetros del diccionario global del main
    video_path = config['video_path']
    roi = config['roi']
    frame_idx = config.get('frame_idx', 0)
    
    frame_rgb, roi_rgb = lectura_frame(video_path=video_path,num_frame=frame_idx, roi_coords=roi) #lep gamma los datos para que devuelva la ROI y frame

    print("Ejecutando segmentación por Otsu")
    
    mascara_otsu, umbral_otsu = otsu(roi_rgb)

    print("Ejecutando segmentación por K-Means")
    
    mascaras_kmeans = kmeans(roi_rgb=roi_rgb)

    print("Ejecutando segmentación por Contornos activos (snakes)")
    
    mask_snake, mask_pupila = snakes(roi_rgb)
    
    print("Ejecutando segmentación por Watershed")
    
    mask_ws = watershed_pupila(roi_rgb, mask_pupila)

    
    print("----------------------------------------------------")
    print("Parte 2.2")
    j = True
    while j:
        print("A continuación, elija la máscara del método que quiera mejorar")
        print("1. Otsu")
        print("2. K-means")
        print("3. Snakes")
        print("4. Watershed")
        print("5. Finalizar")
        
        try:
            caso = int(input("Seleccione una opción (1-5):"))
            if caso == 1:
                mascara = mascara_otsu
                j = False
            elif caso == 2:
                n = True
                while n:
                    try:
                        subcaso = int(input("Ingrese el valor de 2 a 5 del valor de K: "))
                    except ValueError:
                        subcaso = 0
                    if subcaso in mascaras_kmeans:
                        n = False
                    else:
                        print("Por favor ingrese un número entre 2 y 5")
                mascara = mascaras_kmeans[subcaso]
                j = False
            elif caso == 3:
                mascara = mask_snake
                j = False
            elif caso == 4: 
                mascara = mask_ws
                j = False
            elif caso == 5:
                return
            else:
                print("Por favor, ingrese un número válido")
        except ValueError:
            print("Por favor, ingrese un número válido")
        
    print("Ejecutando la erosión, dilatación, apertura y cierre de la máscara")
    
    mascara_erosionada, mascara_dilatada, mascara_apertura, mascara_cierre = edac(mascara=mascara)
    
    