import csv
import requests
from atoma import parse_rss_bytes

URL_FEED = "https://www.un.org/en/feed/subscribe/rss/news"
ARCHIVO_CSV = "noticias_rss.csv"


def descargar_feed(url):
    respuesta = requests.get(url, timeout=15)
    respuesta.raise_for_status()
    return respuesta.content


def parsear_feed(contenido):
    # atoma.parse_rss_bytes devuelve un objeto RSS con .items
    feed = parse_rss_bytes(contenido)
    return feed


def extraer_noticias(feed):
    noticias = []
    for item in feed.items:
        titulo = item.title if item.title else "Sin titulo"
        descripcion = item.description if item.description else "Sin descripcion"
        # la fecha puede venir en description si no esta en pub_date
        fecha = ""
        if item.pub_date:
            fecha = str(item.pub_date)
        elif item.dc_date:
            fecha = str(item.dc_date)
        else:
            fecha = "Fecha no disponible"
        link = item.link if item.link else ""
        noticias.append(
            {
                "titulo": titulo,
                "descripcion": descripcion,
                "fecha": fecha,
                "link": link,
            }
        )
    return noticias


def guardar_csv(noticias):
    if not noticias:
        print("No se encontraron noticias en el feed")
        return

    with open(ARCHIVO_CSV, "w", newline="", encoding="utf-8") as archivo:
        campos = ["titulo", "descripcion", "fecha", "link"]
        escritor = csv.DictWriter(archivo, fieldnames=campos)
        escritor.writeheader()
        escritor.writerows(noticias)

    print(f"Guardadas {len(noticias)} noticias en {ARCHIVO_CSV}")


def main():
    print(f"Descargando feed RSS: {URL_FEED}")
    contenido = descargar_feed(URL_FEED)

    print("Parseando feed con atoma...")
    feed = parsear_feed(contenido)

    print(f"Titulo del feed: {feed.title}")
    noticias = extraer_noticias(feed)

    guardar_csv(noticias)

if __name__ == "__main__":
    main()
