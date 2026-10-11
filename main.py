import os
import sys
import cv2
import tkinter as tk
from tkinter import filedialog

# Importación de los módulos del proyecto
from modulos import Modulo1
from modulos import Modulo2
from modulos import Modulo3
from modulos import Modulo4

# =============================================================================
# FUNCIONES AUXILIARES Y SELECCIÓN (INTEGRADAS EN MAIN)
# =============================================================================

def seleccionar_video_gui():
    """Abre una ventana exploradora de archivos para seleccionar el video."""
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    
    ruta_video = filedialog.askopenfilename(
        title="Seleccionar Video del Ojo (Reflejo Fotomotor)",
        filetypes=[
            ("Archivos de Video", "*.mp4 *.avi *.mov *.mkv *.wmv"),
            ("Todos los archivos", "*.*")
        ]
    )
    root.destroy()
    return ruta_video


def seleccionar_roi_interactivo(video_path, num_frame=0):
    """
    Abre una ventana interactiva de OpenCV con el primer frame del video.
    Permite arrastrar y soltar con el mouse para encuadrar la ROI del ojo.
    - Presionar ENTER o ESPACIO para confirmar la selección.
    - Presionar 'c' para cancelar.
    Devuelve: tupla (ymin, ymax, xmin, xmax)
    """
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, num_frame)
    ret, frame_bgr = cap.read()
    cap.release()

    if not ret or frame_bgr is None:
        raise ValueError("No se pudo leer el frame del video para seleccionar la ROI.")

    window_name = "Seleccionar ROI del Ojo -> Dibuje recuadro con el mouse y presione ENTER"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # Herramienta nativa de OpenCV para recortar ROI (x, y, ancho, alto)
    bbox = cv2.selectROI(window_name, frame_bgr, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(window_name)

    x, y, w, h = bbox
    
    # Si el usuario cierra sin seleccionar nada, toma todo el frame por defecto
    if w == 0 or h == 0:
        print("No se seleccionó ninguna ROI específica. Se utilizará el frame completo.")
        h_frame, w_frame = frame_bgr.shape[:2]
        return 0, h_frame, 0, w_frame

    ymin, ymax = int(y), int(y + h)
    xmin, xmax = int(x), int(x + w)

    return ymin, ymax, xmin, xmax


def obtener_info_video(video_path):
    """Obtiene datos técnicos del archivo de video."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ancho = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    alto = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    return {
        'fps': fps if fps > 0 else 30.0,
        'total_frames': total_frames,
        'ancho': ancho,
        'alto': alto,
        'duracion_sec': total_frames / (fps if fps > 0 else 30.0)
    }


# =============================================================================
# FLUJO PRINCIPAL Y MENÚ DE NAVEGACIÓN
# =============================================================================

def main():
    print("==================================================")
    print("      PROCESAMIENTO DE IMÁGENES BIOMÉDICAS        ")
    print("         Análisis del Reflejo Fotomotor           ")
    print("==================================================")

    # 1. Selección del archivo de video mediante explorador gráfico
    print("\n[1/3] Seleccione el archivo de video en la ventana emergente...")
    video_path = seleccionar_video_gui()
    
    if not video_path:
        print("No se seleccionó ningún archivo de video. Programa finalizado.")
        return

    print(f"Ruta seleccionada: {video_path}")

    # 2. Lectura de información técnica del video
    info = obtener_info_video(video_path)
    print(f"Propiedades: {info['ancho']}x{info['alto']} px | {info['fps']:.2f} FPS | {info['duracion_sec']:.1f} segundos ({info['total_frames']} frames)")

    # 3. Selección interactiva de la ROI dibujando un recuadro sobre el ojo
    print("\n[2/3] Dibuje el recuadro alrededor del ojo/pupila y presione ENTER...")
    roi = seleccionar_roi_interactivo(video_path, num_frame=0)
    print(f"Coordenadas ROI (ymin, ymax, xmin, xmax): {roi}")

    # 4. Diccionario de Configuración global para compartir con los módulos
    config = {
        'video_path': video_path,
        'roi': roi,             # (ymin, ymax, xmin, xmax)
        'frame_idx': 0,         # Frame de trabajo inicial
        'info_video': info
    }

    # 5. Menú Interactivo de Ejecución
    while True:
        print("\n--------------------------------------------------")
        print("                MENÚ DE MÓDULOS                   ")
        print("--------------------------------------------------")
        print("1. Módulo 1: Fundamentos, Espacios de Color y Filtrado")
        print("2. Módulo 2: Segmentación y Morfología (Pupila)")
        print("3. Módulo 3: Caracterización y Curva Temporal")
        print("4. Módulo 4: Compresión de Imagen")
        print("5. Cambiar ROI o Video de trabajo")
        print("0. Salir")
        
        opcion = input("\nIngrese una opción (0-5): ").strip()

        if opcion == "1":
            Modulo1.ejecutar_modulo_1(config)
        elif opcion == "2":
            print("[INFO] El Módulo 2 se implementará en la siguiente etapa.")
        elif opcion == "3":
            print("[INFO] El Módulo 3 se implementará en la siguiente etapa.")
        elif opcion == "4":
            print("[INFO] El Módulo 4 se implementará en la siguiente etapa.")
        elif opcion == "5":
            print("\nRe-seleccionando ROI...")
            config['roi'] = seleccionar_roi_interactivo(config['video_path'], num_frame=0)
            print(f"Nueva ROI registrada: {config['roi']}")
        elif opcion == "0":
            print("Saliendo del programa...")
            break
        else:
            print("Opción inválida. Ingrese un número entre 0 y 5.")

if __name__ == "__main__":
    main()