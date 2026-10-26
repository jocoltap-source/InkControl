# -*- coding: utf-8 -*-
"""
Control de música por gestos de mano para tatuadores
Modo 1: Clasificación simple por reglas
Modo 2: ML entrenable con landmarks
Version mejorada: normalización, debug y contadores de muestras
Autores: Equipo Proyecto IA
"""
# pylint: disable=E0401

import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import tensorflow as tf
from tensorflow import keras  # pylint: disable=E0611
from tensorflow.keras import layers  # pylint: disable=E0611
import os
import csv
import threading
from PIL import Image, ImageTk
import customtkinter as ctk
import time
from tkinter import filedialog
import string
import pandas as pd

# =======================
# Configuración
# =======================
ACCIONES = ["play_pausa", "volumen_up", "volumen_down", "siguiente"]
LETRAS = [chr(i) for i in range(65, 91)]
TIPOS_CONTROL = ["gestos", "letras"]
MODO = "simple"   # "simple" o "ML"

DATASET_FILE = "gestos_dataset.csv"
MODEL_FILE = "modelo_gestos.h5"
LABELS_MAP_FILE = "labels_map.csv"

DATASET_FILE_LETRAS = "letras_dataset.csv"
MODEL_FILE_LETRAS = "modelo_letras.h5"
LABELS_MAP_FILE_LETRAS = "labels_map_letras.csv"


# =======================
# MediaPipe Hands
# =======================
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

ultimo_gesto = None

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# =======================
# Funciones auxiliares
# =======================
def extraer_landmarks(results):
    """
    Devuelve landmarks normalizados (63 valores) relativos a la muñeca.
    Normalizar resta la posición del landmark 0 (muñeca) para reducir variación.
    """
    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        coords = np.array([[lm.x, lm.y, lm.z] for lm in hand.landmark], dtype=float)
        base = coords[0].copy()            # muñeca como referencia
        coords = coords - base             # normalizar por la muñeca
        return coords.flatten()            # 21 * 3 = 63 valores
    return None

def regla_simple(landmarks):
    """Clasificación simple por reglas (sigue usando coordenadas 'crudas' si se desea)."""
    if landmarks is None:
        return None

    # aquí asumimos la estructura flatten (x,y,z)*21; usamos solo Y
    y = [landmarks[i * 3 + 1] for i in range(21)]
    dedos = {
        "pulgar": y[4] < y[3],
        "indice": y[8] < y[6],
        "medio": y[12] < y[10],
        "anular": y[16] < y[14],
        "meñique": y[20] < y[18],
    }

    if all(dedos.values()):
        return "play_pausa"
    elif dedos["pulgar"] and not dedos["indice"]:
        return "volumen_up"
    elif not dedos["pulgar"] and dedos["meñique"]:
        return "volumen_down"
    elif dedos["indice"] and not dedos["medio"]:
        return "siguiente"
    return None



def ejecutar_accion(accion):
    """Mapea acción a control multimedia (usa pyautogui)."""
    if accion == "play_pausa":
        pyautogui.press("playpause")
        print("Pausa/Reproducir")
    elif accion == "volumen_up":
        pyautogui.press("volumeup")
        print("Subir volumen")
    elif accion == "volumen_down":
        pyautogui.press("volumedown")
        print("Bajar volumen")
    elif accion == "siguiente":
        pyautogui.press("nexttrack")
        print("Siguiente canción")
    else:
        print("Acción no mapeada:", accion)

# =======================
# CSV / Dataset helpers
# =======================
def guardar_gesto_csv(label, landmarks):
    """Guarda vector de landmarks con su etiqueta. Maneja errores de permisos."""
    try:
        # asegurarnos que landmarks es iterable
        with open(DATASET_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([label] + list(landmarks))
    except PermissionError as e:
        print("PermissionError al intentar guardar el dataset:", e)
    except Exception as e:
        print("Error al guardar gesto:", e)

def contar_muestras():
    """Cuenta cuántas muestras hay por etiqueta en el dataset."""
    counts = {}
    if not os.path.exists(DATASET_FILE):
        return counts
    try:
        with open(DATASET_FILE, "r", newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    continue
                label = row[0]
                counts[label] = counts.get(label, 0) + 1
    except Exception as e:
        print("Error leyendo dataset para contar muestras:", e)
    return counts

# =======================
# Modelo / Entrenamiento
# =======================
def crear_modelo(input_dim=63, num_clases=4):
    """ML simple para clasificación de gestos (fully-connected, no conv)"""
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.Dense(num_clases, activation="softmax")
    ])
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model

def entrenar_modelo():
    """Entrena ML y guarda modelo + mapeo. Devuelve (model, label_to_id)."""
    data, labels = [], []

    if not os.path.exists(DATASET_FILE):
        print(f"No se encontró {DATASET_FILE}. Guarda gestos primero.")
        return None, None

    with open(DATASET_FILE, "r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            labels.append(row[0])
            # convertir el resto a float
            try:
                data.append([float(x) for x in row[1:]])
            except ValueError:
                # fila corrupta: saltar
                continue

    if not data:
        print("No hay datos válidos en el dataset.")
        return None, None

    print(f"Leyendo dataset: {DATASET_FILE}")
    print(f"Número de muestras cargadas: {len(data)}")
    print(f"Etiquetas detectadas: {set(labels)}")

    etiquetas_unicas = sorted(list(set(labels)))
    label_to_id = {etiquetas_unicas[i]: i for i in range(len(etiquetas_unicas))}
    y = np.array([label_to_id[l] for l in labels])
    X = np.array(data)

    model = crear_modelo(input_dim=X.shape[1], num_clases=len(etiquetas_unicas))
    model.fit(X, y, epochs=15, batch_size=16, validation_split=0.2)
    # guardar modelo (usar formato Keras nativo es recomendado, pero mantenemos .h5 por compat)
    model.save(MODEL_FILE)

    # Guardamos mapeo índice -> etiqueta
    with open(LABELS_MAP_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        for etiqueta, idx in label_to_id.items():
            writer.writerow([idx, etiqueta])

    print("Modelo entrenado y guardado en", MODEL_FILE)
    print("Mapeo guardado en", LABELS_MAP_FILE)
    return model, label_to_id

def cargar_labels_map():
    """Carga labels_map.csv y devuelve dict idx->label y label->idx"""
    if not os.path.exists(LABELS_MAP_FILE):
        return None, None
    idx_to_label = {}
    label_to_idx = {}
    with open(LABELS_MAP_FILE, "r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            idx = int(row[0])
            label = row[1]
            idx_to_label[idx] = label
            label_to_idx[label] = idx
    return idx_to_label, label_to_idx

def cargar_labels_map_archivo(path):
    """Carga un archivo de labels_map genérico (para letras o gestos)."""
    if not os.path.exists(path):
        return None, None
    idx_to_label = {}
    label_to_idx = {}
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            idx = int(row[0])
            label = row[1]
            idx_to_label[idx] = label
            label_to_idx[label] = idx
    return idx_to_label, label_to_idx


def predecir_ML(model, landmarks, idx_to_label_cache):
    """Predice gesto con ML y devuelve nombre de acción (usa cache de labels)."""
    if landmarks is None or model is None:
        return None
    pred = model.predict(landmarks.reshape(1, -1), verbose=0)
    clase = int(np.argmax(pred))
    label = None
    if idx_to_label_cache:
        label = idx_to_label_cache.get(clase)
    else:
        # fallback: intentar leer archivo
        idx_to_label, _ = cargar_labels_map()
        if idx_to_label:
            label = idx_to_label.get(clase)
    print("Predicción raw clase:", clase, "-> etiqueta:", label)
    return label

def cargar_base_letras(self):
    """Permite seleccionar modelo de letras desde la PC."""
    try:
        ruta_modelo = filedialog.askopenfilename(
            title="Seleccionar modelo de letras (.h5)",
            filetypes=[("Modelos Keras", "*.h5"), ("Todos los archivos", "*.*")]
        )
        if not ruta_modelo:
            self.lbl_estado.configure(text="No se seleccionó archivo", text_color="orange")
            return

        self.model = keras.models.load_model(ruta_modelo)
        self.lbl_estado.configure(text=f"Modelo cargado: {os.path.basename(ruta_modelo)}", text_color="lightgreen")

        # Seleccionar mapa de etiquetas
        ruta_labels = filedialog.askopenfilename(
            title="Seleccionar mapa de etiquetas (.csv)",
            filetypes=[("CSV", "*.csv"), ("Todos los archivos", "*.*")]
        )
        if ruta_labels:
            self.idx_to_label, self.label_to_idx = cargar_labels_map_archivo(ruta_labels)
            self.lbl_estado.configure(text="Modelo y etiquetas cargadas", text_color="lightgreen")
        else:
            self.lbl_estado.configure(text="Modelo cargado sin mapa de etiquetas", text_color="orange")

    except Exception as e:
        print("Error cargando base de letras:", e)
        self.lbl_estado.configure(text="Error al cargar base letras", text_color="red")

def cargar_base_gestos(self):
    """Permite seleccionar modelo de gestos desde la PC."""
    try:
        ruta_modelo = filedialog.askopenfilename(
            title="Seleccionar modelo de gestos (.h5)",
            filetypes=[("Modelos Keras", "*.h5"), ("Todos los archivos", "*.*")]
        )
        if not ruta_modelo:
            self.lbl_estado.configure(text="No se seleccionó archivo", text_color="orange")
            return

        self.model = keras.models.load_model(ruta_modelo)
        self.lbl_estado.configure(text=f"Modelo cargado: {os.path.basename(ruta_modelo)}", text_color="lightgreen")

        ruta_labels = filedialog.askopenfilename(
            title="Seleccionar mapa de etiquetas (.csv)",
            filetypes=[("CSV", "*.csv"), ("Todos los archivos", "*.*")]
        )
        if ruta_labels:
            self.idx_to_label, self.label_to_idx = cargar_labels_map_archivo(ruta_labels)
            self.lbl_estado.configure(text="Modelo y etiquetas cargadas", text_color="lightgreen")
        else:
            self.lbl_estado.configure(text="Modelo cargado sin mapa de etiquetas", text_color="orange")

    except Exception as e:
        print("Error cargando base de gestos:", e)
        self.lbl_estado.configure(text="Error al cargar base gestos", text_color="red")

# =======================
# Interfaz gráfica
# =======================
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class InterfazGestos(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Control de música por gestos")
        self.geometry("1400x900")
        self.running = False
        self.modo = ctk.StringVar(value=MODO)
        self.model = None
        self.cap = None
        self.idx_to_label = None  # cache del mapeo
        self.label_to_idx = None
        self.ultimo_gesto_tiempo = 0    
        self.COOLDOWN = 2.0
        self.tipo_control = ctk.StringVar(value="gestos")

        # Layout
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Panel lateral
        # Panel lateral (más ancho)
        self.panel = ctk.CTkFrame(self, width=350)
        self.panel.pack_propagate(False)
        self.panel.pack(side="left", fill="y", padx=10, pady=10)


        ctk.CTkLabel(self.panel, text="Control Gestual", font=("Arial", 20, "bold")).pack(pady=10)
        ctk.CTkLabel(self.panel, text="Modo de funcionamiento").pack(pady=5)
        self.selector_modo = ctk.CTkOptionMenu(
            self.panel,
            values=["ML", "simple"],
            variable=self.modo,
            command=self.actualizar_modo
        )
        self.selector_modo.pack(pady=5)

        ctk.CTkButton(self.panel, text="Iniciar cámara", command=self.iniciar_camara).pack(pady=10)
        ctk.CTkButton(self.panel, text="Detener cámara", command=self.detener_camara).pack(pady=5)

        self.btn_entrenar = ctk.CTkButton(self.panel, text="Entrenar modelo", command=self.entrenar)
        if self.modo.get() == "ML":
            self.btn_entrenar.pack(pady=5)

        # Botón para guardar gestos (solo visible en modo ML)
        self.btn_guardar_gesto = ctk.CTkButton(self.panel, text="Guardar gesto", command=self.guardar_gesto)
        if self.modo.get() == "ML":
            self.btn_guardar_gesto.pack(pady=5)


        # --- TIPO DE CONTROL ---
        self.label_tipo = ctk.CTkLabel(self.panel, text="Tipo de control")
        self.selector_tipo = ctk.CTkOptionMenu(
            self.panel,
            values=TIPOS_CONTROL,
            variable=self.tipo_control,
            command=self.actualizar_tipo
        )

        # --- ACCIÓN / LETRA ---
        self.label_accion = ctk.CTkLabel(self.panel, text="Acción / Letra asociada")
        self.selector_accion = ctk.CTkOptionMenu(self.panel, values=ACCIONES)

        # Botones específicos para letras (solo aparecen si tipo = letras)
        self.btn_cargar_base_letras = ctk.CTkButton(self.panel, text="Cargar base letras", command=self.cargar_base_letras)
        self.btn_nueva_base_letras = ctk.CTkButton(self.panel, text="Nueva base letras", command=self.nueva_base_letras)

        # Botones específicos para gestos (solo aparecen si tipo = gestos)
        self.btn_cargar_base_gestos = ctk.CTkButton(self.panel, text="Cargar base gestos", command=self.cargar_base_gestos)
        self.btn_nueva_base_gestos = ctk.CTkButton(self.panel, text="Nueva base gestos", command=self.nueva_base_gestos)

        # Empiezan ocultos
        for b in [
            self.btn_cargar_base_letras, self.btn_nueva_base_letras,
            self.btn_cargar_base_gestos, self.btn_nueva_base_gestos
        ]:
            b.pack_forget()

        # estado y contadores
        self.lbl_estado = ctk.CTkLabel(self.panel, text="Esperando acción...", text_color="gray")
        self.lbl_estado.pack(pady=(12, 6))

        # Contadores por etiqueta (solo visibles en modo ML)
        self.label_muestras = ctk.CTkLabel(self.panel, text="Muestras por etiqueta:", anchor="w")
        self.counters_text = ctk.CTkLabel(self.panel, text="", anchor="w", justify="left")

        self.btn_borrar_base = ctk.CTkButton(self.panel, text="Borrar base", command=self.borrar_base)  

        # Empiezan ocultos si el modo inicial es "simple"
        if self.modo.get() == "ML":
            self.label_muestras.pack(pady=(8, 4))
            self.counters_text.pack()

        # Vista cámara
        self.lbl_video = ctk.CTkLabel(self, width=1300, height=850)
        self.lbl_video.pack(side="right", expand=True, fill="both", padx=10, pady=10)


        # Etiqueta de gesto detectado
        self.lbl_accion = ctk.CTkLabel(self, text="", font=("Arial", 18, "bold"))
        self.lbl_accion.pack(side="bottom", pady=(6, 12))

        # actualizar contadores al iniciar UI
        self.actualizar_contadores()

    # ======== Funciones UI ========
    def actualizar_contadores(self):
        """Actualiza los contadores de muestras según el tipo de control."""
        tipo = self.tipo_control.get()

        # Escoge el archivo adecuado
        if tipo == "letras":
            dataset_file = DATASET_FILE_LETRAS
            valores = LETRAS
        else:
            dataset_file = DATASET_FILE
            valores = ACCIONES

        counts = {}
        if os.path.exists(dataset_file):
            try:
                with open(dataset_file, "r", newline="") as f:
                    reader = csv.reader(f)
                    next(reader, None)  # saltar encabezado
                    for row in reader:
                        if not row:
                            continue
                        label = row[0]
                        counts[label] = counts.get(label, 0) + 1
            except Exception as e:
                print("Error leyendo dataset:", e)

        # Construir texto
        if not counts:
            texto = "No hay muestras aún."
        else:
            lines = [f"{v}: {counts.get(v, 0)}" for v in valores]
            texto = "\n".join(lines)

        self.counters_text.configure(text=texto)
    
        # ======== Funciones para letras (bases) ========
    def cargar_base_letras(self):
        """Carga una base existente de letras (modelo y dataset)."""
        try:
            if os.path.exists(MODEL_FILE_LETRAS):
                self.model = keras.models.load_model(MODEL_FILE_LETRAS)
                self.idx_to_label, self.label_to_idx = cargar_labels_map_archivo(LABELS_MAP_FILE_LETRAS)
                self.lbl_estado.configure(text="Base de letras cargada", text_color="lightgreen")
                print("Modelo de letras cargado:", MODEL_FILE_LETRAS)
            else:
                self.lbl_estado.configure(text="No se encontró modelo de letras", text_color="red")
                print("No existe archivo de modelo de letras:", MODEL_FILE_LETRAS)
            self.tipo_control.set("letras")
            self.actualizar_contadores()
        except Exception as e:
            print("Error cargando base de letras:", e)
            self.lbl_estado.configure(text="Error al cargar base letras", text_color="red")

    def cargar_csv_letras(self):
        try:
            ruta = filedialog.askopenfilename(
                title="Seleccionar base de letras",
                filetypes=[("CSV files", "*.csv")]
            )
            if not ruta:
                self.lbl_estado.configure(text="No seleccionaste archivo", text_color="orange")
                return

            df = pd.read_csv(ruta)
            conteos = df["clase"].value_counts()
            total = len(df)
            self.lbl_estado.configure(text=f"{total} muestras cargadas", text_color="green")

            # Actualizar contadores A-Z
            for i in range(26):
                letra = chr(65 + i)
                if letra in conteos:
                    self.contadores_letras[i].configure(text=str(conteos[letra]))
                else:
                    self.contadores_letras[i].configure(text="0")

        except Exception as e:
            self.lbl_estado.configure(text=f"Error cargando base: {str(e)}", text_color="red")


    def nueva_base_letras(self):
        """Crea una nueva base vacía para recolectar gestos de letras."""
        try:
            # Elimina dataset y modelo anteriores si existen
            for f in [DATASET_FILE_LETRAS, MODEL_FILE_LETRAS, LABELS_MAP_FILE_LETRAS]:
                if os.path.exists(f):
                    os.remove(f)
                    print(f"Archivo anterior eliminado: {f}")
            # Crear CSV vacío con encabezado opcional
            with open(DATASET_FILE_LETRAS, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["label"] + [f"p{i}" for i in range(63)])
            self.lbl_estado.configure(text="Nueva base de letras creada", text_color="lightblue")
            print("Nueva base de letras inicializada:", DATASET_FILE_LETRAS)
            self.actualizar_contadores()
        except Exception as e:
            print("Error creando base de letras:", e)
            self.lbl_estado.configure(text="Error al crear base letras", text_color="red")

    def cargar_base_gestos(self):
        """Carga una base existente de gestos (modelo y dataset)."""
        try:
            if os.path.exists(MODEL_FILE):
                self.model = keras.models.load_model(MODEL_FILE)
                self.idx_to_label, self.label_to_idx = cargar_labels_map_archivo(LABELS_MAP_FILE)
                self.lbl_estado.configure(text="Base de gestos cargada", text_color="lightgreen")
                print("Modelo de gestos cargado:", MODEL_FILE)
            else:
                self.lbl_estado.configure(text="No se encontró modelo de gestos", text_color="red")
                print("No existe archivo de modelo de gestos:", MODEL_FILE)
            self.tipo_control.set("gestos")
            self.actualizar_contadores()
        except Exception as e:
            print("Error cargando base de gestos:", e)
            self.lbl_estado.configure(text="Error al cargar base gestos", text_color="red")

    def nueva_base_gestos(self):
        try:
        #Si existe un dataset previo, intentar borrarlo pero sin fallar si está usado
            try:
                if os.path.exists(DATASET_FILE):
                    os.remove(DATASET_FILE)
            except PermissionError:
                self.lbl_estado.configure(
                    text="Cierra el archivo gestos_dataset.csv antes de continuar",
                    text_color="red"
                )
                return

            #Crear el archivo desde cero
            with open(DATASET_FILE, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["label"] + [f"p{i}" for i in range(63)])

            self.lbl_estado.configure(
                text="Nueva base de gestos creada correctamente",
                text_color="lightblue"
            )

            #Actualizar contadores
            self.actualizar_contadores()

        except Exception as e:
            print("Error creando base de gestos:", e)
            self.lbl_estado.configure(
                text="Error al crear base de gestos",
                text_color="red"
            )

    def cargar_csv_gestos(self):
        try:
            ruta = filedialog.askopenfilename(
                title="Seleccionar base de gestos",
                filetypes=[("CSV files", "*.csv")]
            )
            if not ruta:
                self.lbl_estado.configure(text="No seleccionaste archivo", text_color="orange")
                return

            df = pd.read_csv(ruta)
            conteos = df["clase"].value_counts()
            total = len(df)
            self.lbl_estado.configure(text=f"{total} muestras cargadas", text_color="green")

            # Actualizar contadores de acciones
            for i, accion in enumerate(ACCIONES):
                if accion in conteos:
                    self.contadores_gestos[i].configure(text=str(conteos[accion]))
                else:
                    self.contadores_gestos[i].configure(text="0")

        except Exception as e:
            self.lbl_estado.configure(text=f"Error cargando base: {str(e)}", text_color="red")


    def iniciar_camara(self):
        if not self.running:
            self.running = True
            self.cap = cv2.VideoCapture(0)  # pylint: disable=no-member
            # intentar cargar modelo y mapa si existen
            if self.modo.get() == "ML" and os.path.exists(MODEL_FILE):
                try:
                    self.model = keras.models.load_model(MODEL_FILE)
                    # cargar mapeo
                    self.idx_to_label, self.label_to_idx = cargar_labels_map()
                    print("Modelo cargado correctamente:", MODEL_FILE)
                    print("Labels cargadas:", self.idx_to_label)
                    self.lbl_estado.configure(text="Modelo cargado", text_color="lightgreen")
                except Exception as e:
                    print("Error cargando modelo:", e)
                    self.lbl_estado.configure(text="Error cargando modelo", text_color="red")
            else:
                self.lbl_estado.configure(text="Cámara iniciada (modo simple si no hay modelo)", text_color="lightgreen")
            threading.Thread(target=self.mostrar_video, daemon=True).start()

    def detener_camara(self):
        self.running = False
        if self.cap:
            self.cap.release()
            self.lbl_video.configure(image=None)
            self.lbl_accion.configure(text="")
            self.lbl_estado.configure(text="Cámara detenida", text_color="orange")

    def mostrar_video(self):
        while self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # pylint: disable=no-member
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # pylint: disable=no-member
            results = hands.process(frame_rgb)
            landmarks = extraer_landmarks(results)
            accion = None

            # ========= BORDE VERDE / ROJO SEGÚN DETECCIÓN =========
            color = (0,255,0) if results.multi_hand_landmarks else (0,0,0)
            cv2.rectangle(frame, (0,0), (frame.shape[1], frame.shape[0]), color, 3) # pylint: disable=no-member

            # === CLASIFICACIÓN SIMPLE ===
            if self.modo.get() == "simple":
                tipo = self.tipo_control.get()

                if tipo == "gestos":
                    accion = regla_simple(landmarks)
                    if accion:
                        if accion == "play_pausa":  # solo play_pausa tendrá cooldown
                            ahora = time.time()
                            if ahora - self.ultimo_gesto_tiempo >= self.COOLDOWN:
                                ejecutar_accion(accion)
                                self.ultimo_gesto_tiempo = ahora
                                self.lbl_accion.configure(text=f"Acción: {accion}")
                        else:
                            ejecutar_accion(accion)  # los demás sin cooldown
                            self.lbl_accion.configure(text=f"Acción: {accion}")


            # modo ML
            elif self.modo.get() == "ML" and self.model is not None:
                accion = predecir_ML(self.model, landmarks, self.idx_to_label)

                # Si hay predicción
                if accion is not None:

                    # Si está en modo letras (A, B, C...)
                    if self.tipo_control.get() == "letras":
                        self.lbl_accion.configure(text=f"Letra: {accion}")

                        ahora = time.time()
                        if ahora - self.ultimo_gesto_tiempo >= 2.0:  # cooldown de 2.0 s
                            pyautogui.write(accion)
                            print("Letra escrita:", accion)
                            self.ultimo_gesto_tiempo = ahora

                    # Si está en modo gestos (play, pausa, etc.)
                    else:
                        self.lbl_accion.configure(text=f"Gesto: {accion}")

                        ahora = time.time()
                        if ahora - self.ultimo_gesto_tiempo >= self.COOLDOWN:
                            ejecutar_accion(accion)
                            self.ultimo_gesto_tiempo = ahora
                            print("Gesto ejecutado:", accion)
                        else:
                            print("Detección ignorada por cooldown")
                else:
                    self.lbl_accion.configure(text="—")


            # Dibujar landmarks si los hay
            if results.multi_hand_landmarks:
                for hand in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

            # Mostrar frame en la interfaz (convertir a PhotoImage)
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # pylint: disable=no-member
            imgtk = ImageTk.PhotoImage(image=Image.fromarray(img))
            self.lbl_video.imgtk = imgtk
            self.lbl_video.configure(image=imgtk)

        if self.cap:
            self.cap.release()

    def actualizar_tipo(self, *_):
        self.actualizar_contadores()
        tipo = self.tipo_control.get()
        # Ocultar todo primero
        for b in [
            self.btn_cargar_base_letras, self.btn_nueva_base_letras,
            self.btn_cargar_base_gestos, self.btn_nueva_base_gestos
        ]:
            b.pack_forget()

        if tipo == "gestos":
            self.selector_accion.configure(values=ACCIONES)
            self.btn_cargar_base_gestos.pack(pady=5)
            self.btn_nueva_base_gestos.pack(pady=5)
        else:
            self.selector_accion.configure(values=LETRAS)
            self.btn_cargar_base_letras.pack(pady=5)
            self.btn_nueva_base_letras.pack(pady=5)


    # ======== Mostrar / Ocultar controles según modo ========
    def mostrar_controles_ML(self):
        """Muestra los elementos del modo ML."""
        self.btn_entrenar.pack(pady=5)
        self.label_tipo.pack(pady=(8, 4))
        self.selector_tipo.pack(pady=5)
        self.label_accion.pack(pady=(8, 4))
        self.selector_accion.pack(pady=5)
        self.label_muestras.pack(pady=(8, 4))
        self.counters_text.pack()
        self.btn_guardar_gesto.pack(pady=5)
        # Botones de letras (se controlan con tipo de control)
        if self.tipo_control.get() == "letras":
            self.btn_cargar_base_letras.pack(pady=5)
            self.btn_nueva_base_letras.pack(pady=5)

        self.btn_borrar_base.pack(pady=5)

    def ocultar_controles_ML(self):
        """Oculta los elementos específicos del modo ML (solo deja lo básico)."""
        self.btn_entrenar.pack_forget()
        self.label_tipo.pack_forget()
        self.selector_tipo.pack_forget()
        self.label_accion.pack_forget()
        self.selector_accion.pack_forget()
        self.label_muestras.pack_forget()
        self.counters_text.pack_forget()
        self.btn_guardar_gesto.pack_forget()
        self.btn_cargar_base_letras.pack_forget()
        self.btn_nueva_base_letras.pack_forget()
        self.btn_borrar_base.pack_forget()

    def actualizar_modo(self, *args):
        """Cambia visibilidad según modo seleccionado."""
        if self.modo.get() == "ML":
            self.mostrar_controles_ML()
        else:
            self.ocultar_controles_ML()


    def guardar_gesto(self):
        label = self.selector_accion.get()
        tipo = self.tipo_control.get()

        if not label:
            self.lbl_estado.configure(text="Selecciona una acción o letra", text_color="orange")
            return
        if self.modo.get() != "ML":
            self.lbl_estado.configure(text="Cambia a modo 'ML' para guardar", text_color="orange")
            return
        if not self.cap:
            self.lbl_estado.configure(text="Cámara no iniciada", text_color="red")
            return

        ret, frame = self.cap.read()
        if not ret:
            return

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # pylint: disable=no-member
        results = hands.process(frame_rgb)
        landmarks = extraer_landmarks(results)

        if landmarks is None:
            self.lbl_estado.configure(text="No se detectó una mano", text_color="red")
            return

        #convertir a lista de floats antes de guardar
        landmarks = list(map(float, landmarks))

        dataset_file = DATASET_FILE_LETRAS if tipo == "letras" else DATASET_FILE

        try:
            file_exists = os.path.exists(dataset_file)
            with open(dataset_file, "a", newline="") as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(["label"] + [f"p{i}" for i in range(len(landmarks))])
                writer.writerow([label] + landmarks)

            self.lbl_estado.configure(text=f" {tipo.capitalize()} '{label}' guardado", text_color="lightblue")
            print(f"{tipo.capitalize()} '{label}' guardado en {dataset_file}")
            self.actualizar_contadores()

        except Exception as e:
            print("Error guardando gesto:", e)
            self.lbl_estado.configure(text="Error al guardar", text_color="red")

    def entrenar(self):
        if self.modo.get() != "ML":
            self.lbl_estado.configure(text="Selecciona modo 'ML' para entrenar", text_color="orange")
            return
        self.lbl_estado.configure(text="Entrenando modelo...", text_color="yellow")
        threading.Thread(target=self._entrenar_hilo, daemon=True).start()

    def _entrenar_hilo(self):
        model, label_map = entrenar_modelo()
        if model is not None:
            # recargar modelo y map de etiquetas en memoria
            try:
                self.model = keras.models.load_model(MODEL_FILE)
            except Exception as e:
                print("Error recargando modelo tras entrenamiento:", e)
                self.lbl_estado.configure(text="Error cargando modelo tras entrenamiento", text_color="red")
                return
            self.idx_to_label, self.label_to_idx = cargar_labels_map()
            self.lbl_estado.configure(text="Modelo entrenado y cargado", text_color="lightgreen")
            # actualizar contadores por si algo cambió
            self.actualizar_contadores()
        else:
            self.lbl_estado.configure(text="Entrenamiento fallido", text_color="red")

    def borrar_base(self):
        """Elimina base CSV + modelo dependiendo del tipo de control."""
        tipo = self.tipo_control.get()

        if tipo == "gestos":
            archivos = [DATASET_FILE, MODEL_FILE, LABELS_MAP_FILE]
            texto = "gestos"
        else:
            archivos = [DATASET_FILE_LETRAS, MODEL_FILE_LETRAS, LABELS_MAP_FILE_LETRAS]
            texto = "letras"

        error_permiso = False
        for f in archivos:
            if os.path.exists(f):
                try:
                    os.remove(f)
                    print("Archivo eliminado:", f)
                except PermissionError:
                    error_permiso = True

        if error_permiso:
            self.lbl_estado.configure(
                text="Cierra el CSV antes de borrarlo",
                text_color="red"
            )
            return

        # Reset de variables internas
        self.model = None
        self.idx_to_label = None
        self.label_to_idx = None

        self.lbl_estado.configure(
            text=f"Base de {texto} eliminada",
            text_color="lightblue"
        )
        self.actualizar_contadores()


if __name__ == "__main__":
    app = InterfazGestos()
    app.mainloop()
