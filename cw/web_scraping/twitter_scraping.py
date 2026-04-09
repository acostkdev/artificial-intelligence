import asyncio
import csv
from twikit import Client

IDIOMA = "en-US"
ARCHIVO_COOKIES = "cookies.json"
ARCHIVO_CSV = "tweets_extraidos.csv"

USUARIO = "TU_USUARIO"
CORREO = "TU_CORREO@ejemplo.com"
CONTRASENA = "TU_CONTRASENA"

CONSULTA = "inteligencia artificial"
CANTIDAD_TWEETS = 20


async def iniciar_sesion():
    cliente = Client(IDIOMA)

    try:
        await cliente.login(
            auth_info_1=USUARIO,
            auth_info_2=CORREO,
            password=CONTRASENA,
            cookies_file=ARCHIVO_COOKIES,
        )
        print(f"Sesion iniciada. Cookies guardadas en {ARCHIVO_COOKIES}")
    except Exception as e:
        print(f"Error al iniciar sesion: {e}")
        return None

    return cliente


async def buscar_tweets(cliente):
    print(f"Buscando tweets para: '{CONSULTA}'")
    tweets = await cliente.search_tweet(CONSULTA, "Latest")

    resultados = []
    for tweet in tweets:
        # algunos tweets no traen todos los campos
        usuario = getattr(tweet.user, "screen_name", "desconocido")
        texto = getattr(tweet, "text", "")
        fecha = getattr(tweet, "created_at", "")
        likes = getattr(tweet, "favorite_count", 0)
        retweets = getattr(tweet, "retweet_count", 0)
        resultados.append(
            {
                "usuario": usuario,
                "texto": texto,
                "fecha": fecha,
                "likes": likes,
                "retweets": retweets,
            }
        )

    return resultados


def guardar_csv(resultados):
    if not resultados:
        print("No hay resultados que guardar")
        return

    with open(ARCHIVO_CSV, "w", newline="", encoding="utf-8") as archivo:
        campos = ["usuario", "texto", "fecha", "likes", "retweets"]
        escritor = csv.DictWriter(archivo, fieldnames=campos)
        escritor.writeheader()
        escritor.writerows(resultados)

    print(f"Guardados {len(resultados)} tweets en {ARCHIVO_CSV}")


async def main():
    cliente = await iniciar_sesion()
    if cliente is not None:
        resultados = await buscar_tweets(cliente)
        guardar_csv(resultados)

if __name__ == "__main__":
    asyncio.run(main())
