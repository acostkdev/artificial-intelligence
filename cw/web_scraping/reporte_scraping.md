# Web Scraping (Twitter / RSS) — Reporte

## Que hicimos

Dos ejercicios de extraccion de datos de la web: uno con la API no
oficial de Twitter usando la libreria twikit y otro con feeds RSS usando
atoma. Ambos exportan los resultados a CSV para su posterior analisis.

## Twitter con twikit

Usamos twikit porque permite buscar tweets sin necesidad de una API key
de Twitter. La libreria simula un cliente web y se autentica con cookies
de sesion guardadas localmente.

El flujo del script es:

1. Crear un `Client` con idioma "en-US"
2. Llamar a `login()` con usuario, correo, contraseña y un archivo de
   cookies. La primera vez pide las credenciales reales, las siguientes
   reusa las cookies guardadas
3. Usar `search_tweet()` con la consulta que queremos y el tipo "Latest"
   para obtener los tweets mas recientes
4. Iterar sobre los resultados y extraer: usuario, texto, fecha, likes,
   retweets
5. Guardar todo en un CSV con `csv.DictWriter`

El login es asincrono, por eso todo va dentro de funciones `async` y se
ejecuta con `asyncio.run()`. Esto es necesario porque twikit usa
aiohttp por debajo para las peticiones HTTP.

Observamos que algunos tweets pueden venir sin ciertos campos, por eso
usamos `getattr` con valores por defecto. Esto evita que el script
truene si un tweet no tiene, por ejemplo, `favorite_count`.

## RSS con atoma

Para RSS usamos atoma, que parsea el XML del feed y lo convierte en
objetos Python con tipos. No necesitamos escribir parsers XML a mano.

El flujo del script es:

1. Descargar el feed con `requests.get()` (el feed de noticias de la
   ONU en este caso)
2. Parsear el contenido con `atoma.parse_rss_bytes()` que devuelve un
   objeto `RSS`
3. Extraer de cada item: titulo, descripcion, fecha de publicacion y
   link
4. Guardar en CSV igual que con los tweets

El feed de la ONU trae `pub_date` en la mayoria de los items pero
algunos solo tienen `dc_date`. Por eso revisamos ambos campos al extraer
la fecha (ni modo, toca andar revisando).

## Problemas encontrados

- **twikit y login:** Si las cookies expiran o son invalidas, twikit
   lanza un error y toca volver a autenticarse. No hay forma sencilla de
   saber si las cookies sirven sin intentarlo.
- **Rate limiting:** twikit no tiene control de rate limiting explicito.
   Si hacemos muchas busquedas seguidas Twitter puede bloquear
   temporalmente la cuenta. Solo hicimos una consulta para evitar eso.
- **RSS y campos faltantes:** No todos los items del feed tienen todos
   los campos. Algunos traen `dc_date` en vez de `pub_date`, otros no
   traen descripcion. Hay que manejar esos casos con valores por defecto.
- **atoma y feeds mal formados:** Si el XML del feed esta mal formado,
   `parse_rss_bytes` lanza una excepcion. No todos los feeds RSS son
   validos.
