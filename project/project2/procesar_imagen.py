import numpy as np
from PIL import Image
import rembg

COLOR_FONDO = np.array([255, 255, 255], dtype=np.uint8)

_sesion = None

def _get_sesion():
    global _sesion
    if _sesion is None:
        _sesion = rembg.new_session()
    return _sesion

def remover_fondo(pil_img):
    session = _get_sesion()
    resultado = rembg.remove(pil_img, session=session)
    arr = np.array(resultado, dtype=np.uint8)
    mask = arr[:, :, 3] > 0
    rgb = arr[:, :, :3]
    salida = np.full_like(rgb, COLOR_FONDO)
    salida[mask] = rgb[mask]
    return Image.fromarray(salida)
