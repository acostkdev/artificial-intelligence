import numpy as np
import os
import random
import shutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img

CARPETA_RAW = os.path.join(os.path.dirname(__file__), "raw_descargas")
CARPETA_DATASET = os.path.join(os.path.dirname(__file__), "dataset")
TOTAL_POR_CLASE = 5000
SEMILLA = 42
TAMANO = 64

random.seed(SEMILLA)
np.random.seed(SEMILLA)

CLASES = ["aranas", "ballenas", "changos", "pajaros", "ranas"]

def contar_imagenes(carpeta):
    total = 0
    for f in os.listdir(carpeta):
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
            total += 1
    return total

datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    brightness_range=[0.7, 1.3]
)

def generar_aumentadas(origen, destino, cantidad_generar):
    os.makedirs(destino, exist_ok=True)
    imagenes = []
    for f in os.listdir(origen):
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
            imagenes.append(os.path.join(origen, f))
    if not imagenes:
        print(f"  [WARN] No hay imagenes en {origen}")
        return 0
    generadas = 0
    rondas = 0
    max_rondas = 100
    while generadas < cantidad_generar and rondas < max_rondas:
        for ruta in imagenes:
            if generadas >= cantidad_generar:
                break
            try:
                img = load_img(ruta, target_size=(TAMANO, TAMANO))
                x = img_to_array(img)
                x = x.reshape((1,) + x.shape)
                for batch in datagen.flow(x, batch_size=1, save_to_dir=destino,
                                          save_prefix=f"aug_{generadas:06d}", save_format='jpg'):
                    generadas += 1
                    break
            except Exception as e:
                continue
        rondas += 1
    return generadas

print("=" * 60)
print("ARMADO DEL DATASET BALANCEADO (5000 POR CLASE)")
print("=" * 60)

for clase in CLASES:
    print(f"\n--- {clase.upper()} ---")
    carpeta_raw = os.path.join(CARPETA_RAW, clase)
    carpeta_destino = os.path.join(CARPETA_DATASET, clase)
    os.makedirs(carpeta_destino, exist_ok=True)

    disponibles = contar_imagenes(carpeta_raw)
    print(f"  Disponibles en raw: {disponibles}")

    for f in os.listdir(carpeta_destino):
        ruta = os.path.join(carpeta_destino, f)
        if os.path.isfile(ruta):
            os.remove(ruta)

    if disponibles >= TOTAL_POR_CLASE:
        todas = [f for f in os.listdir(carpeta_raw) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]
        seleccion = random.Random(SEMILLA).sample(todas, TOTAL_POR_CLASE)
        for i, archivo in enumerate(seleccion):
            ext = os.path.splitext(archivo)[1].lower()
            if ext == '.jpeg':
                ext = '.jpg'
            shutil.copy2(os.path.join(carpeta_raw, archivo), os.path.join(carpeta_destino, f"real_{i:06d}{ext}"))
        print(f"  Muestreadas {TOTAL_POR_CLASE} imagenes reales")
    else:
        todas = [f for f in os.listdir(carpeta_raw) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]
        for i, archivo in enumerate(todas):
            ext = os.path.splitext(archivo)[1].lower()
            if ext == '.jpeg':
                ext = '.jpg'
            shutil.copy2(os.path.join(carpeta_raw, archivo), os.path.join(carpeta_destino, f"real_{i:06d}{ext}"))
        copiadas = len(todas)
        faltan = TOTAL_POR_CLASE - copiadas
        print(f"  Copiadas {copiadas} reales, faltan {faltan} por aumentacion")
        generadas = generar_aumentadas(carpeta_raw, carpeta_destino, faltan)
        print(f"  Generadas {generadas} imagenes por aumentacion")

    total_final = contar_imagenes(carpeta_destino)
    print(f"  Total final en dataset/{clase}: {total_final}")

print("\n" + "=" * 60)
print("VERIFICACION FINAL")
print("=" * 60)
for clase in CLASES:
    total = contar_imagenes(os.path.join(CARPETA_DATASET, clase))
    print(f"  {clase}: {total} imagenes")
print("=" * 60)
