import numpy as np
import os
import sys
import shutil
import random
import subprocess
import tarfile
import kagglehub

CARPETA_DATASET = os.path.join(os.path.dirname(__file__), "dataset")
CARPETA_RAW = os.path.join(os.path.dirname(__file__), "raw_descargas")
TOTAL_POR_CLASE = 5000
SEMILLA = 42

random.seed(SEMILLA)
np.random.seed(SEMILLA)

os.makedirs(CARPETA_DATASET, exist_ok=True)
os.makedirs(CARPETA_RAW, exist_ok=True)

CLASES = ["aranas", "ballenas", "changos", "pajaros", "ranas"]

def contar_imagenes(carpeta):
    total = 0
    for raiz, _, archivos in os.walk(carpeta):
        for archivo in archivos:
            if archivo.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                total += 1
    return total

def muestrear_y_copiar(origen, destino, cantidad, prefijo="img"):
    os.makedirs(destino, exist_ok=True)
    imagenes = []
    for raiz, _, archivos in os.walk(origen):
        for archivo in archivos:
            if archivo.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                imagenes.append(os.path.join(raiz, archivo))
    if not imagenes:
        return 0
    rng = random.Random(SEMILLA)
    seleccion = rng.sample(imagenes, min(cantidad, len(imagenes)))
    for i, ruta in enumerate(seleccion):
        ext = os.path.splitext(ruta)[1].lower()
        if ext == '.jpeg':
            ext = '.jpg'
        destino_archivo = os.path.join(destino, f"{prefijo}_{i:06d}{ext}")
        shutil.copy2(ruta, destino_archivo)
    return len(seleccion)


print("=" * 60)
print("FASE 1: DESCARGAR DATASETS")
print("=" * 60)

# --- 1. ARANAS ---
print("\n[1/5] ARANAS")
carpeta_aranas_raw = os.path.join(CARPETA_RAW, "aranas")
os.makedirs(carpeta_aranas_raw, exist_ok=True)

ruta_spiders = os.path.expanduser(
    "~/.cache/kagglehub/datasets/gpiosenka/yikes-spiders-15-species/versions/4"
)
if os.path.exists(ruta_spiders):
    copiadas = muestrear_y_copiar(
        os.path.join(ruta_spiders, "train"),
        carpeta_aranas_raw, 99999, "kaggle"
    )
    print(f"  Copiadas {copiadas} de gpiosenka spiders")
else:
    print("  [WARN] gpiosenka spiders no encontrado en cache")

ruta_local_aranas = os.path.join(CARPETA_DATASET, "aranas")
if os.path.exists(ruta_local_aranas):
    for f in os.listdir(ruta_local_aranas):
        ruta_origen = os.path.join(ruta_local_aranas, f)
        if os.path.isfile(ruta_origen):
            shutil.copy2(ruta_origen, os.path.join(carpeta_aranas_raw, f"local_{f}"))

total_aranas = contar_imagenes(carpeta_aranas_raw)
print(f"  Total aranas: {total_aranas}")


# --- 2. BALLENAS ---
print("\n[2/5] BALLENAS")
carpeta_ballenas_raw = os.path.join(CARPETA_RAW, "ballenas")
os.makedirs(carpeta_ballenas_raw, exist_ok=True)

ruta_sea = os.path.expanduser(
    "~/.cache/kagglehub/datasets/vencerlanz09/sea-animals-image-dataste/versions/5"
)
carpeta_whale = None
if os.path.exists(ruta_sea):
    for entry in os.listdir(ruta_sea):
        if "whale" in entry.lower():
            carpeta_whale = os.path.join(ruta_sea, entry)
            break
    if carpeta_whale and os.path.isdir(carpeta_whale):
        copiadas = muestrear_y_copiar(carpeta_whale, carpeta_ballenas_raw, 99999, "sea")
        print(f"  Copiadas {copiadas} de sea-animals")
    else:
        print("  [WARN] No se encontro carpeta Whale")
else:
    print("  [WARN] sea-animals no encontrado en cache")

ruta_local_ballenas = os.path.join(CARPETA_DATASET, "ballenas")
if os.path.exists(ruta_local_ballenas):
    for f in os.listdir(ruta_local_ballenas):
        ruta_origen = os.path.join(ruta_local_ballenas, f)
        if os.path.isfile(ruta_origen):
            shutil.copy2(ruta_origen, os.path.join(carpeta_ballenas_raw, f"local_{f}"))

total_ballenas = contar_imagenes(carpeta_ballenas_raw)
print(f"  Total ballenas: {total_ballenas}")


# --- 3. CHANGOS ---
print("\n[3/5] CHANGOS")
carpeta_changos_raw = os.path.join(CARPETA_RAW, "changos")
os.makedirs(carpeta_changos_raw, exist_ok=True)

ruta_monos_cache = os.path.expanduser(
    "~/.cache/kagglehub/datasets/slothkong/10-monkey-species/versions/2"
)
if not os.path.exists(ruta_monos_cache):
    print("  Descargando 10-monkey-species desde Kaggle...")
    try:
        ruta_monos_cache = kagglehub.dataset_download("slothkong/10-monkey-species")
    except Exception as e:
        print(f"  [ERROR] {e}")

if os.path.exists(ruta_monos_cache):
    copiadas = muestrear_y_copiar(ruta_monos_cache, carpeta_changos_raw, 99999, "mono")
    print(f"  Copiadas {copiadas} de 10-monkey-species")
else:
    print("  [WARN] No hay dataset de monos")

ruta_local_changos = os.path.join(CARPETA_DATASET, "changos")
if os.path.exists(ruta_local_changos):
    for f in os.listdir(ruta_local_changos):
        ruta_origen = os.path.join(ruta_local_changos, f)
        if os.path.isfile(ruta_origen):
            shutil.copy2(ruta_origen, os.path.join(carpeta_changos_raw, f"local_{f}"))

total_changos = contar_imagenes(carpeta_changos_raw)
print(f"  Total changos: {total_changos}")


# --- 4. PAJAROS ---
print("\n[4/5] PAJAROS")
carpeta_pajaros_raw = os.path.join(CARPETA_RAW, "pajaros")
os.makedirs(carpeta_pajaros_raw, exist_ok=True)

ruta_pajaros_cache = os.path.expanduser(
    "~/.cache/kagglehub/datasets/veeralakrishna/200-bird-species-with-11788-images/versions/1"
)
if not os.path.exists(ruta_pajaros_cache):
    print("  Descargando 200-bird-species desde Kaggle...")
    try:
        ruta_pajaros_cache = kagglehub.dataset_download(
            "veeralakrishna/200-bird-species-with-11788-images"
        )
    except Exception as e:
        print(f"  [ERROR] {e}")
else:
    print("  Ya descargado en cache")

if os.path.exists(ruta_pajaros_cache):
    tgz = os.path.join(ruta_pajaros_cache, "CUB_200_2011.tgz")
    extraido = os.path.join(ruta_pajaros_cache, "extraido")
    if os.path.exists(tgz) and not os.path.exists(extraido):
        print("  Extrayendo CUB_200_2011.tgz...")
        os.makedirs(extraido, exist_ok=True)
        with tarfile.open(tgz) as tar:
            tar.extractall(path=extraido)
        print("  Extraccion completa")
    carpeta_aves = os.path.join(extraido, "CUB_200_2011", "images")
    if os.path.exists(carpeta_aves):
        copiadas = muestrear_y_copiar(carpeta_aves, carpeta_pajaros_raw, 99999, "ave")
        print(f"  Copiadas {copiadas} imagenes de aves")
    else:
        print(f"  [WARN] No se encontro carpeta images en {extraido}")
else:
    print("  [WARN] No hay dataset de aves")

total_pajaros = contar_imagenes(carpeta_pajaros_raw)
print(f"  Total pajaros: {total_pajaros}")


# --- 5. RANAS ---
print("\n[5/5] RANAS")
carpeta_ranas_raw = os.path.join(CARPETA_RAW, "ranas")
os.makedirs(carpeta_ranas_raw, exist_ok=True)

ruta_frog_git = os.path.join(CARPETA_RAW, "frog-dataset")
if not os.path.exists(ruta_frog_git) or not os.listdir(ruta_frog_git):
    shutil.rmtree(ruta_frog_git, ignore_errors=True)
    print("  Clonando frog-dataset desde GitHub...")
    try:
        subprocess.run(
            ["git", "clone", "https://github.com/jonshamir/frog-dataset.git", ruta_frog_git],
            check=True, capture_output=True, timeout=300
        )
    except subprocess.TimeoutExpired:
        print("  [ERROR] git clone timed out, reintentando con shallow clone...")
        subprocess.run(
            ["git", "clone", "--depth", "1",
             "https://github.com/jonshamir/frog-dataset.git", ruta_frog_git],
            check=True, capture_output=True, timeout=300
        )
    except Exception as e:
        print(f"  [ERROR] No se pudo clonar: {e}")
else:
    print("  frog-dataset ya existe")

if os.path.exists(ruta_frog_git) and os.listdir(ruta_frog_git):
    copiadas = muestrear_y_copiar(ruta_frog_git, carpeta_ranas_raw, 99999, "rana")
    print(f"  Copiadas {copiadas} imagenes de ranas")
else:
    print("  [WARN] No hay dataset de ranas")

total_ranas = contar_imagenes(carpeta_ranas_raw)
print(f"  Total ranas: {total_ranas}")


# --- REPORTE FINAL ---
print("\n" + "=" * 60)
print("REPORTE FINAL - IMAGENES POR CLASE")
print("=" * 60)

totales = {
    "aranas": contar_imagenes(carpeta_aranas_raw),
    "ballenas": contar_imagenes(carpeta_ballenas_raw),
    "changos": contar_imagenes(carpeta_changos_raw),
    "pajaros": contar_imagenes(carpeta_pajaros_raw),
    "ranas": contar_imagenes(carpeta_ranas_raw),
}

for clase in CLASES:
    reales = totales[clase]
    faltan = max(0, TOTAL_POR_CLASE - reales)
    estado = "COMPLETO" if reales >= TOTAL_POR_CLASE else f"FALTAN {faltan}"
    print(f"  {clase}: {reales}/{TOTAL_POR_CLASE} -> {estado}")

print("\nClases incompletas se rellenaran con aumentacion offline")
print("en la siguiente fase.")
print("=" * 60)
