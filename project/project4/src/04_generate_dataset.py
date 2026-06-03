import json
import random
import os

random.seed(42)
OUTPUT = "dataset/tutor_dataset.jsonl"
os.makedirs("dataset", exist_ok=True)

examples = []

# ─── helpers ───────────────────────────────────────────────────────
def make(inst, output, level=1):
    examples.append({"instruction": inst, "output": output, "level": level})

# ─── fraseos académicos para variar ─────────────────────────────────
contextos = [
    "Para abordar esta pregunta,", "Cabe señalar que,",
    "En primer lugar, conviene precisar que,",
    "De acuerdo con la información disponible,",
    "El corpus documental permite establecer que,",
]
conectores = [
    "No obstante,", "Por otra parte,", "En consecuencia,",
    "Asimismo,", "Conviene destacar que,",
    "Cabe añadir que,", "Dicho lo anterior,",
]
cierres = [
    "En conclusión,", "Por lo tanto,", "Así pues,",
    "En suma,", "De lo anterior se desprende que,",
]

def ctx():
    return random.choice(contextos)

def con():
    return random.choice(conectores)

def fin():
    return random.choice(cierres)

# ─── genera una cita mixta ─────────────────────────────────────────
def cite(nombre_doc, pag="", texto_corto=""):
    pag_str = f", Pág {pag}" if pag else ""
    if texto_corto:
        return f"como señala {nombre_doc}{pag_str}: '{texto_corto}' [Documento {nombre_doc.split(':')[0].split()[-1] if 'Doc' not in nombre_doc else nombre_doc.split('[')[1].split(']')[0]}]"
    return f"según {nombre_doc}{pag_str} [Documento {nombre_doc.split(':')[0].split()[-1]}]"

# ═══════════════════════════════════════════════════════════════════
# NIVEL 1: EXTRACCIÓN DIRECTA — con citas
# ═══════════════════════════════════════════════════════════════════

make(
    "¿Cuáles son las tres entidades federativas con mayor tasa de homicidios?",
    f"{ctx()} los datos del ONC indican que Colima, Baja California Sur y Chihuahua presentan las tasas más altas de homicidio doloso por cada 100 mil habitantes. {con()} Baja California también figura con cifras elevadas en el periodo analizado. {fin()} estas cinco entidades concentran la mayor parte de la violencia homicida del país. [Documento 1, Pág 29]",
    level=1,
)
make(
    "¿Qué dice el corpus sobre la cifra negra en México?",
    f"{ctx()} la ENVIPE 2025 reporta que el 93.2% de los delitos cometidos en México no se denuncian o no derivan en una carpeta de investigación. {con()} esto implica que la estadística oficial captura menos del 7% de la incidencia delictiva real. {fin()} la cifra negra sigue siendo el principal desafío para medir la violencia en el país. [Documento 2, Pág 4]",
    level=1,
)
make(
    "¿Cuántas víctimas de homicidio se registraron en el periodo analizado?",
    f"{ctx()} el informe del ONC documenta más de 100,000 muertes en 10 años asociadas al conflicto entre el Estado y los cárteles. {con()} la tasa alcanzó un pico de casi 25 homicidios por cada 100 mil habitantes en 2011. {fin()} estas cifras colocan a México como uno de los países más violentos del continente. [Documento 1, Pág 10]",
    level=1,
)
make(
    "¿Qué porcentaje de mujeres ha experimentado violencia a lo largo de su vida?",
    f"{ctx()} según la ENDIREH 2021 del INEGI, el 70.1% de las mujeres de 15 años y más han experimentado al menos un incidente de violencia a lo largo de su vida. {con()} la violencia psicológica es la más prevalente con un 51.6%, seguida de la violencia sexual con 49.7%. {fin()} estas cifras reflejan la magnitud estructural de la violencia de género en México. [Documento 2]",
    level=1,
)
make(
    "¿Cuál es la entidad con mayor prevalencia de violencia contra las mujeres?",
    f"{ctx()} el Estado de México presenta la mayor prevalencia con un 78.7%, seguido de la Ciudad de México con 76.2% y Querétaro con 75.2%. {con()} en contraste, Chiapas reporta la prevalencia más baja con 48.7%. {fin()} estas disparidades regionales sugieren factores locales que incidenden en la violencia de género. [Documento 2]",
    level=1,
)
make(
    "¿Cuánto le cuesta a México la inseguridad y el delito?",
    f"{ctx()} según la ENVIPE 2025, el costo nacional estimado a consecuencia de la inseguridad y el delito asciende a 269.6 mil millones de pesos. {con()} esta cifra equivale al 1.07% del Producto Interno Bruto nacional. {fin()} el impacto económico es significativo y afecta tanto a hogares como a empresas. [Documento 2, Pág 4]",
    level=1,
)
make(
    "¿Qué es la cifra negra y por qué es relevante?",
    f"{ctx()} la cifra negra se refiere a los delitos que no son denunciados o que, siendo denunciados, no derivan en una carpeta de investigación. {con()} la ENVIPE 2025 estima que el 93.2% de los delitos en México quedan en esta categoría. {fin()} esto significa que las estadísticas oficiales representan apenas una fracción de la criminalidad real. [Documento 2, Pág 4]",
    level=1,
)
make(
    "¿Cuántas víctimas de delito hubo en México en 2024?",
    f"{ctx()} la ENVIPE 2025 estima 23.1 millones de personas víctimas de delitos en 2024, equivalentes a 24,135 víctimas por cada 100 mil habitantes. {con()} esta cifra es estadísticamente superior a la de 2023, que fue de 21.9 millones. {fin()} el aumento sugiere un repunte en la victimización a nivel nacional. [Documento 2, Pág 4]",
    level=1,
)
make(
    "¿Cuántas páginas tiene el reporte de homicidios del ONC?",
    f"{ctx()} el reporte 'Homicidio: Una mirada a la violencia en México' del Observatorio Nacional Ciudadano consta de 266 páginas que abarcan desde la evolución histórica del homicidio hasta recomendaciones de política pública. {con()} incluye secciones sobre homicidios vinculados al crimen organizado, homicidios sociopolíticos y fosas clandestinas. {fin()} es uno de los análisis más completos sobre el tema en México. [Documento 1]",
    level=1,
)

# ═══════════════════════════════════════════════════════════════════
# NIVEL 2: SÍNTESIS — respuesta estructurada + reflexión
# ═══════════════════════════════════════════════════════════════════

make(
    "¿Cómo ha evolucionado la violencia homicida en las últimas dos décadas?",
    f"{ctx()} el ONC documenta tres periodos claramente diferenciados. {con()} primero, entre 1997 y 2006 se observa un descenso sostenido de la tasa de homicidios. {con()} segundo, a partir de 2007 la tasa se dispara hasta alcanzar su pico en 2011, con casi 25 homicidios por cada 100 mil habitantes. {con()} tercero, de 2011 a 2014 inicia un descenso hasta aproximadamente 16 homicidios por cada 100 mil habitantes. {fin()} la evolución refleja el impacto de la guerra contra el narcotráfico y sus consecuencias en la violencia letal. {con()} a manera de reflexión: ¿considera usted que las estrategias de seguridad implementadas a partir de 2007 lograron sus objetivos, o más bien exacerbaron la violencia? [Documento 1, Pág 29]",
    level=2,
)
make(
    "¿Qué factores asocian los autores al incremento de la violencia urbana?",
    f"{ctx()} el reporte del ONC señala que el homicidio en México no puede entenderse sin considerar la confrontación entre el Estado y los cárteles de la droga. {con()} se identifican factores como la fragmentación de los grupos criminales, la disputa por territorios y la debilidad institucional en procuración de justicia. {con()} la literatura académica citada en el corpus también menciona la desigualdad socioeconómica y la impunidad como factores estructurales. {fin()} la violencia urbana es un fenómeno multicausal que requiere análisis integrales. [Documento 1]",
    level=2,
)
make(
    "¿Qué sectores económicos son los más afectados por la extorsión?",
    f"{ctx()} según la ENVIPE 2025, el robo total o parcial de vehículo tiene la tasa más alta de prevalencia, con 8,921 víctimas por cada 100 mil habitantes. {con()} el fraude le sigue con 5,344, y el robo en calle o transporte público con 4,556. {con()} la extorsión presenta una tasa de 3,517 víctimas por cada 100 mil habitantes. {fin()} el sector transporte y el comercio minorista aparecen como los más vulnerables a la extorsión. [Documento 2]",
    level=2,
)
make(
    "¿Cómo se relacionan la percepción de inseguridad y la victimización real?",
    f"{ctx()} la ENVIPE 2025 reporta que el 75.6% de la población percibe inseguridad en su entidad federativa, mientras que la tasa de victimización es de 24,135 por cada 100 mil habitantes. {con()} la brecha entre percepción y victimización sugiere que factores como la cobertura mediática y la confianza institucional inciden en la sensación de inseguridad. {con()} las mujeres reportan mayor percepción de inseguridad (79.4%) que los hombres (70.9%). {fin()} la percepción de inseguridad no siempre se correlaciona directamente con la incidencia delictiva objetiva. [Documento 2, Pág 4]",
    level=2,
)
make(
    "¿Qué diferencia hay entre las cifras del INEGI y las del Secretariado Ejecutivo?",
    f"{ctx()} según el artículo periodístico incluido en el corpus, el INEGI reporta actas de defunción mientras que el SESNSP reporta carpetas de investigación abiertas. {con()} ambas fuentes coinciden en la tendencia a la baja en homicidios, aunque los valores absolutos difieren. {con()} la especialista Eunice Rendón señala que en 26 entidades se observó disminución del delito. {fin()} la concordancia en tendencias pero no en magnitudes refleja metodologías distintas de registro. [Documento 3]",
    level=2,
)

# ═══════════════════════════════════════════════════════════════════
# NIVEL 3: RAZONAMIENTO ANALÍTICO — método socrático + incertidumbre
# ═══════════════════════════════════════════════════════════════════

make(
    "¿Qué impacto tiene la violencia documentada sobre la tasa de deserción escolar?",
    f"Antes de responder, permítame plantearle una reflexión: ¿a qué tipo de violencia se refiere específicamente? La violencia documentada en el corpus abarca homicidio doloso, extorsión y desplazamiento forzado, pero habría que distinguir entre violencia directa en entornos escolares y el efecto indirecto de la violencia comunitaria.\n\n{con()} ¿considera usted que la deserción escolar podría estar vinculada a factores económicos más que directamente a la violencia? Esta distinción es importante para encuadrar correctamente el análisis.\n\n{ctx()} el corpus disponible no contiene información específica sobre deserción escolar ni sobre el sistema educativo en México. Los documentos consultados —ONC Homicidio, INEGI ENVIPE 2025 y ONC Anual 2025— se centran en incidencia delictiva, victimización y percepción de seguridad, sin abordar variables educativas.\n\n{fin()} no es posible ofrecer una respuesta fundamentada sobre el impacto de la violencia en la deserción escolar con los documentos disponibles. [Documento 1, ONC; Documento 2, INEGI ENVIPE 2025; Documento 3, ONC Anual 2025]",
    level=3,
)
make(
    "¿Cuáles son las principales contradicciones entre ONGs y gobierno en el registro de víctimas?",
    f"{ctx()} antes de abordar las contradicciones, conviene preguntarse: ¿qué entiende cada fuente por 'víctima'? Para el gobierno, una víctima de homicidio es aquella que cuenta con una carpeta de investigación abierta; para las ONGs como el ONC, la cifra incluye estimaciones basadas en actas de defunción y reportes forenses.\n\n{con()} dicho lo anterior, el corpus evidencia que el ONC señala que la estadística oficial de homicidio doloso 'representa solo una parte de los homicidios cometidos en el país' debido a un subregistro importante por debilidades institucionales. {con()} la ENVIPE 2025 confirma esta brecha al reportar que el 93.2% de los delitos no se denuncian. {fin()} la principal contradicción no radica en los datos en sí, sino en la interpretación: mientras el gobierno destaca tendencias a la baja, las ONGs advierten que la disminución no implica que el problema esté resuelto y señalan la existencia de un subregistro estructural. [Documento 1, Pág 10; Documento 2, Pág 4]",
    level=3,
)
make(
    "¿Qué vacíos de información señalan los autores como el principal obstáculo para medir la violencia real?",
    f"{ctx()} antes de responder: ¿considera usted que la principal limitación es la falta de datos o la falta de calidad en los datos existentes?\n\n{con()} el ONC identifica varios vacíos críticos. {con()} en primer lugar, el subregistro de homicidios por debilidades en las procuradurías y fiscalías estatales. {con()} en segundo lugar, la crisis de desapariciones forzadas cuyo posible vínculo con fosas clandestinas no queda capturado en las estadísticas de homicidio. {con()} en tercer lugar, la cifra negra del 93.2% reportada por la ENVIPE 2025 implica que la mayoría de los delitos simplemente no aparecen en ningún registro oficial.\n\n{con()} a esto se suma la fragmentación institucional: cada fuente (SESNSP, INEGI, ONGs) usa metodologías distintas. {fin()} el principal obstáculo no es la ausencia total de datos, sino la falta de integración, estandarización y calidad de los registros existentes. [Documento 1, Pág 10; Documento 2, Pág 4]",
    level=3,
)
make(
    "¿Recomendaría la militarización como estrategia de seguridad pública?",
    f"{ctx()} antes de emitir una recomendación: ¿qué criterios utilizaría usted para evaluar si una estrategia de seguridad es efectiva? ¿La reducción de homicidios a corto plazo, o también el respeto a derechos humanos y el fortalecimiento institucional a largo plazo?\n\n{con()} el corpus del ONC documenta que la confrontación directa entre el Estado y los cárteles a partir de 2007 llevó la tasa de homicidios a un pico histórico de casi 25 por cada 100 mil habitantes en 2011. {con()} no obstante, el mismo reporte señala que la violencia no puede atribuirse únicamente a la estrategia militar, sino también a la fragmentación de grupos criminales y a disputas territoriales.\n\n{fin()} como analista, no me corresponde hacer una recomendación normativa. Prefiero señalar que el corpus evidencia que las estrategias exclusivamente militares han tenido costos humanos significativos, y que las políticas de prevención social aparecen como un complemento necesario según la literatura disponible. ¿Cuál es su opinión sobre el balance entre mano dura y prevención? [Documento 1, Pág 29]",
    level=3,
)

# ═══════════════════════════════════════════════════════════════════
# MÁS EJEMPLOS DE NIVEL 1 Y 2
# ═══════════════════════════════════════════════════════════════════

make(
    "¿Qué dice el ONC sobre las fosas clandestinas en México?",
    f"{ctx()} el reporte del ONC dedica un capítulo completo a las fosas clandestinas como una aproximación a las debilidades institucionales de procuración de justicia. {con()} señala que el hallazgo de múltiples fosas en el país evidencia la crisis de desapariciones forzadas e involuntarias. {fin()} el vínculo entre fosas clandestinas y homicidios no registrados es un tema crucial que el informe aborda con profundidad. [Documento 1, Pág 221]",
    level=1,
)
make(
    "¿Cuáles son los costos no monetarios de los homicidios según el ONC?",
    f"{ctx()} el ONC dedica un capítulo a los costos no monetarios de los homicidios, que incluyen el impacto psicológico en las familias, la pérdida de capital social y la erosión de la confianza institucional. {con()} también documenta la reducción en la expectativa de vida debido a la violencia. {fin()} estos costos, aunque difíciles de cuantificar, son fundamentales para entender el impacto total de la violencia. [Documento 1, Pág 235]",
    level=1,
)
make(
    "¿Qué recomienda el ONC para reducir la violencia homicida?",
    f"{ctx()} el informe del ONC incluye recomendaciones generales que abarcan desde el fortalecimiento institucional de las procuradurías hasta políticas de prevención social. {con()} enfatiza la necesidad de mejorar la calidad del registro de datos y reducir el subregistro. {con()} también recomienda abordar las causas estructurales de la violencia y no solo sus manifestaciones más visibles. {fin()} las recomendaciones apuntan a una estrategia integral que combine prevención, justicia y reconstrucción del tejido social. [Documento 1, Pág 249]",
    level=2,
)
make(
    "¿Qué nos dice el corpus sobre la violencia en zonas rurales vs urbanas?",
    f"{ctx()} el corpus disponible proporciona información principalmente a nivel nacional y estatal, sin una desagregación sistemática entre zonas rurales y urbanas. {con()} el reporte del ONC menciona que entidades como Guerrero, Oaxaca y Chiapas —con alta proporción de población rural— presentan dinámicas de violencia particulares. {con()} no obstante, la mayoría de los análisis se centran en zonas metropolitanas y regiones con alta presencia del crimen organizado. {fin()} no es posible ofrecer un contraste detallado entre violencia rural y urbana con los documentos disponibles. [Documento 1]",
    level=2,
)
make(
    "¿Qué relación hay entre desplazamiento forzado y violencia homicida?",
    f"{ctx()} el corpus documenta que México enfrenta una crisis de desapariciones forzadas e involuntarias, así como de hallazgo de fosas clandestinas. {con()} sin embargo, no se encontraron datos específicos sobre desplazamiento forzado interno en los documentos analizados. {con()} la ENVIPE 2025 se centra en victimización delictiva general, pero no mide desplazamiento. {fin()} la información disponible no permite establecer una relación cuantitativa entre violencia homicida y desplazamiento forzado. [Documento 1; Documento 2]",
    level=2,
)

# ═══════════════════════════════════════════════════════════════════
# EJEMPLOS DE INCERTIDUMBRE — preguntas fuera del corpus
# ═══════════════════════════════════════════════════════════════════

make(
    "¿Cómo ha cambiado la tasa de suicidios en México?",
    f"Antes de responder, permítame preguntar: ¿considera usted que el suicidio y el homicidio comparten las mismas causas estructurales? Esta distinción es importante porque el corpus disponible se centra exclusivamente en violencia interpersonal y homicidio doloso.\n\n{ctx()} el corpus disponible no contiene información específica sobre tasas de suicidio en México. Los documentos incluidos —ONC Homicidio, INEGI ENVIPE 2025 y ONC Anual 2025— se centran en homicidio doloso, victimización delictiva y percepción de seguridad, sin abordar variables de salud mental o suicidio. {fin()} no es posible ofrecer una respuesta fundamentada con los documentos disponibles. [Documento 1, ONC; Documento 2, INEGI ENVIPE 2025; Documento 3, ONC Anual 2025]",
    level=3,
)
make(
    "¿Cuál es la relación entre el cambio climático y la violencia en México?",
    f"Antes de responder, reflexionemos: ¿cómo imagina usted que el cambio climático podría influir en la violencia? ¿A través de migración forzada, escasez de recursos o debilitamiento institucional?\n\n{ctx()} el corpus disponible no aborda la relación entre cambio climático y violencia. Los documentos analizados se enfocan en incidencia delictiva, homicidios y percepción de seguridad pública. {fin()} no es posible establecer una relación fundamentada entre ambos fenómenos con los documentos disponibles. [Documento 1; Documento 2; Documento 3]",
    level=3,
)
make(
    "¿Cuántos policías hay por cada 100 mil habitantes en México?",
    f"Antes de responder, permítame preguntar: ¿qué relación esperaría usted encontrar entre el número de policías por habitante y la tasa de homicidios? ¿Una mayor densidad policial se traduce necesariamente en menor violencia?\n\n{ctx()} el corpus documental disponible no incluye información sobre el estado de fuerza policial ni la tasa de policías por habitante. Los documentos se centran en la incidencia delictiva y la percepción ciudadana, no en las capacidades institucionales de las corporaciones policiales. {fin()} no es posible ofrecer una respuesta fundamentada con los documentos disponibles. [Documento 1; Documento 2; Documento 3]",
    level=3,
)
make(
    "¿Qué impacto tiene la violencia en la salud mental de los mexicanos?",
    f"Antes de abordar el impacto, conviene preguntarse: ¿cómo se mide el impacto en salud mental a nivel poblacional? ¿A través de tasas de trastornos, consumo de servicios de salud, o encuestas de bienestar?\n\n{ctx()} el corpus disponible no contiene información específica sobre salud mental. Aunque el ONC menciona los costos no monetarios de los homicidios, no se encontraron datos sistematizados sobre impacto psicológico o tasas de trastornos mentales asociados a la violencia. {fin()} no es posible ofrecer una respuesta fundamentada sobre el impacto en salud mental con los documentos disponibles. [Documento 1]",
    level=3,
)
make(
    "¿Qué programas de prevención del delito han funcionado en México?",
    f"Antes de responder, reflexione: ¿qué criterios usaría para determinar si un programa de prevención ha 'funcionado'? ¿Reducción de delitos, aumento de denuncias, o mejora en percepción de seguridad?\n\n{ctx()} el corpus disponible no evalúa programas específicos de prevención del delito. El ONC ofrece recomendaciones generales en su capítulo final, pero no incluye evaluaciones de impacto de programas particulares. {con()} la ENVIPE 2025 mide percepción de seguridad pero no efectividad de políticas preventivas. {fin()} no es posible evaluar programas específicos con los documentos disponibles. [Documento 1, Pág 249; Documento 2]",
    level=3,
)

# ═══════════════════════════════════════════════════════════════════
# MÁS VARIEDAD — combinaciones de comportamientos
# ═══════════════════════════════════════════════════════════════════

make(
    "Explícame qué es la tasa de prevalencia delictiva.",
    f"{ctx()} la tasa de prevalencia delictiva es un indicador que mide el número de víctimas de delito por cada 100 mil habitantes en un periodo determinado. {con()} según la ENVIPE 2025, en 2024 esta tasa fue de 24,135 víctimas por cada 100 mil habitantes a nivel nacional. {con()} es distinta de la tasa de incidencia, que mide el número total de delitos (una víctima puede sufrir múltiples delitos). {fin()} la prevalencia es útil para entender qué proporción de la población se ve directamente afectada por el delito. [Documento 2, Pág 4]",
    level=1,
)
make(
    "¿Qué significa que la cifra negra sea del 93.2%?",
    f"{ctx()} significa que de cada 100 delitos que ocurren en México, aproximadamente 93 no son denunciados o, si se denuncian, no derivan en una carpeta de investigación. {con()} esto implica que las estadísticas oficiales de incidencia delictiva representan apenas el 6.8% de la criminalidad real. {con()} las causas incluyen desconfianza en las autoridades, miedo a represalias y percepción de que denunciar no sirve. {fin()} la cifra negra es el principal desafío para diseñar políticas de seguridad basadas en evidencia. [Documento 2, Pág 4]",
    level=1,
)
make(
    "¿Cómo se compara la violencia en México con otros países de América Latina?",
    f"Antes de comparar, permítame preguntar: ¿qué indicadores usaría para medir la violencia entre países? ¿Tasa de homicidios, percepción de inseguridad, o algo distinto? La elección del indicador cambia completamente la comparación.\n\n{ctx()} el corpus disponible se centra exclusivamente en México y no incluye datos comparativos con otros países de la región. {con()} el ONC menciona que México es percibido internacionalmente como un país violento, pero no ofrece tablas comparativas. {fin()} no es posible realizar una comparación fundamentada con otros países con los documentos disponibles. [Documento 1, Pág 10]",
    level=3,
)
make(
    "Hazme un resumen ejecutivo de la situación de violencia en México según el corpus.",
    f"{ctx()} con base en los tres documentos analizados, es posible trazar el siguiente panorama:\n\n1. La violencia homicida en México ha pasado por tres fases: descenso (1997-2006), explosión (2007-2011) y descenso gradual (2011-2014), según el ONC.\n\n2. La ENVIPE 2025 reporta 23.1 millones de víctimas de delito en 2024, con una cifra negra del 93.2%, lo que significa que solo 1 de cada 15 delitos se denuncia.\n\n3. El costo de la inseguridad asciende a 269.6 mil millones de pesos, equivalente al 1.07% del PIB.\n\n4. El 75.6% de la población percibe inseguridad en su entidad federativa.\n\n5. Las mujeres son las más afectadas: 70.1% ha experimentado violencia a lo largo de su vida.\n\n{con()} no obstante, persisten vacíos importantes: subregistro de homicidios, falta de datos sobre desapariciones, y ausencia de información sobre educación, salud o programas de prevención. {fin()} el corpus proporciona una base sólida para entender la magnitud del problema, pero insuficiente para análisis multidimensionales. [Documento 1; Documento 2; Documento 3]",
    level=2,
)

# ═══════════════════════════════════════════════════════════════════
# MÁS EJEMPLOS — para llegar a ~60
# ═══════════════════════════════════════════════════════════════════

make(
    "¿Qué dice el informe del ONC sobre la evolución del homicidio entre 1997 y 2006?",
    f"{ctx()} el ONC documenta un descenso sostenido de la tasa de homicidios durante este periodo, pasando de niveles elevados a una reducción gradual. {con()} este descenso se atribuye a diversos factores, aunque el informe no profundiza en causas específicas para este periodo. {fin()} la tendencia se revertiría drásticamente a partir de 2007. [Documento 1, Pág 29]",
    level=1,
)
make(
    "¿Cuántos homicidios de hombres se registraron en 2015 según el INEGI?",
    f"{ctx()} según datos del INEGI citados en el corpus, en 2015 se registraron 20,762 muertes por homicidio en el país, de las cuales 18,293 fueron de hombres y 2,383 de mujeres. {con()} esto significa que los hombres representaron el 88.1% de las víctimas de homicidio. {fin()} la violencia homicida tiene un impacto desproporcionado en la población masculina. [Documento 2]",
    level=1,
)
make(
    "¿Qué organizaciones criminales operan en Tierra Caliente según el corpus?",
    f"{ctx()} el corpus disponible menciona a 'La Familia Michoacana' como una organización criminal con presencia en la región de Tierra Caliente, específicamente en los límites entre Michoacán, Estado de México y Morelos. {con()} también se hace referencia a células vinculadas a esta organización que operan en alcaldías de la Ciudad de México como Tláhuac y Milpa Alta. {fin()} el corpus no proporciona un listado exhaustivo de todos los grupos en la región. [Documento 3]",
    level=1,
)
make(
    "¿Cuál es la tasa de víctimas de delito en hombres vs mujeres en 2024?",
    f"{ctx()} según la ENVIPE 2025, en 2024 se registraron 23,399 mujeres víctimas por cada 100 mil habitantes y 25,010 hombres. {con()} la tasa de hombres resultó estadísticamente superior a la de 2023, mientras que la de mujeres no presentó cambio significativo. {fin()} la brecha de género en victimización se mantiene, con los hombres ligeramente más afectados. [Documento 2, Pág 4]",
    level=1,
)
make(
    "¿Qué nos dice la ENVIPE sobre la confianza en las instituciones de seguridad?",
    f"{ctx()} la ENVIPE 2025 mide el grado de confianza en las instituciones de seguridad pública y la percepción sobre el desempeño de las autoridades. {con()} el 75.6% de la población percibe inseguridad, lo que sugiere niveles bajos de confianza institucional. {con()} las causas incluyen la cifra negra del 93.2% y la percepción de que denunciar no tiene consecuencias. {fin()} la desconfianza institucional es tanto causa como consecuencia de la cifra negra. [Documento 2]",
    level=2,
)
make(
    "¿Qué relación hay entre violencia y juventud según el Banco Mundial?",
    f"{ctx()} el corpus no incluye el reporte del Banco Mundial sobre violencia juvenil, pero el ONC menciona que los jóvenes son tanto víctimas como perpetradores de violencia en contextos de criminalidad organizada. {con()} la literatura académica citada en el corpus sugiere que la exclusión económica y educativa son factores de riesgo. {fin()} el corpus disponible no permite profundizar en esta relación. [Documento 1]",
    level=2,
)
make(
    "¿Cómo se distribuye la violencia homicida por entidad federativa?",
    f"{ctx()} según el ONC, Colima, Baja California Sur y Chihuahua encabezan las tasas de homicidio doloso. {con()} la ENVIPE 2025 añade que las tasas más altas de victimización general se registran en Estado de México (34,851), Ciudad de México (30,804) y Tlaxcala (30,498). {con()} en contraste, Chiapas (15,576), Tamaulipas (16,537) y Michoacán presentan las tasas más bajas. {fin()} existe una correlación parcial entre homicidios y victimización general, pero no perfecta. [Documento 1; Documento 2]",
    level=2,
)
make(
    "¿Qué metodología usa la ENVIPE para medir la victimización?",
    f"{ctx()} la ENVIPE es una encuesta de hogares con representatividad nacional y estatal. {con()} se basa en el autorreporte de victimización de la población de 18 años y más. {con()} la Encuesta Nacional de Victimización y Percepción sobre Seguridad Pública (ENVIPE) 2025 es la décima quinta entrega de esta serie estadística. {fin()} su diseño sigue las mejores prácticas internacionales recomendadas por la ONU para encuestas de victimización. [Documento 2]",
    level=1,
)
make(
    "¿Cuántos delitos ocurrieron en México en 2024 según la ENVIPE?",
    f"{ctx()} la ENVIPE 2025 estima 33.5 millones de delitos ocurridos en 2024, equivalentes a una tasa de 34,918 delitos por cada 100 mil habitantes. {con()} de estos, solo el 6.8% derivaron en una carpeta de investigación. {fin()} la mayoría de los delitos quedan en la impunidad estadística. [Documento 2, Pág 4]",
    level=1,
)
make(
    "¿Qué dice el ONC sobre la prevención social del delito?",
    f"{ctx()} el ONC dedica su capítulo de conclusiones y recomendaciones a señalar la necesidad de políticas de prevención social que complementen las estrategias de seguridad. {con()} menciona que la prevención debe abordar causas estructurales como la desigualdad y la debilidad institucional. {con()} no obstante, el reporte no evalúa programas específicos de prevención. {fin()} la recomendación apunta a un enfoque integral más allá de la confrontación directa. [Documento 1, Pág 249]",
    level=2,
)
make(
    "¿Cuáles son las principales causas de violencia en México según los autores del corpus?",
    f"{ctx()} antes de enumerar causas, permítame preguntar: ¿considera usted que la violencia en México es principalmente producto del crimen organizado o de factores estructurales más profundos?\n\n{con()} el ONC identifica como causas principales la confrontación entre el Estado y los cárteles, la fragmentación de grupos criminales y la disputa por territorios. {con()} la literatura académica citada añade la desigualdad socioeconómica, la impunidad y la debilidad institucional en procuración de justicia. {con()} la ENVIPE 2025 aporta el dato de que el 93.2% de los delitos no se denuncian, lo que sugiere que la impunidad es un factor transversal. {fin()} las causas son multicapa: inmediatas (conflicto entre grupos), intermedias (debilidad institucional) y estructurales (desigualdad). [Documento 1; Documento 2]",
    level=3,
)
make(
    "¿Qué papel juegan los medios de comunicación en la percepción de inseguridad?",
    f"{ctx()} antes de responder: ¿cree usted que la percepción de inseguridad refleja fielmente la incidencia delictiva real?\n\n{con()} la ENVIPE 2025 reporta que el 75.6% de la población percibe inseguridad, mientras que la tasa de victimización es de 24,135 por cada 100 mil habitantes. {con()} la brecha entre percepción y realidad sugiere que la cobertura mediática y la desconfianza institucional amplifican la sensación de riesgo. {con()} el corpus no analiza directamente el papel de los medios, pero los datos permiten inferir que la percepción no es un simple reflejo de la incidencia delictiva. {fin()} sería necesario contar con estudios específicos sobre cobertura mediática para profundizar en esta cuestión. [Documento 2]",
    level=3,
)
make(
    "¿Qué es el SESNSP y qué papel juega en las estadísticas de seguridad?",
    f"{ctx()} el Secretariado Ejecutivo del Sistema Nacional de Seguridad Pública (SESNSP) es el órgano administrativo encargado de recopilar y publicar las estadísticas oficiales de incidencia delictiva en México. {con()} según aclara el Gabinete de Seguridad, las estadísticas oficiales solo se reportan por el Centro Nacional de Información del SESNSP. {con()} el INEGI, por su parte, genera sus propias cifras a través de encuestas de victimización como la ENVIPE. {fin()} ambas fuentes son complementarias pero usan metodologías distintas. [Documento 2; Documento 3]",
    level=1,
)
make(
    "¿Qué nos dice el ONC sobre la violencia contra las mujeres en México?",
    f"{ctx()} el ONC aborda el feminicidio como una de las caras de la violencia en México, aunque su enfoque principal es el homicidio doloso en general. {con()} la ENDIREH 2021 del INEGI, citada en el corpus, reporta que el 70.1% de las mujeres ha experimentado violencia a lo largo de su vida. {con()} la violencia psicológica (51.6%) y la violencia sexual (49.7%) son las más prevalentes. {fin()} la violencia contra las mujeres es un fenómeno profundo y estructural, no meramente un subproducto de la violencia criminal. [Documento 1; Documento 2]",
    level=2,
)
make(
    "¿Cuál es el tamaño de la población carcelaria en México?",
    f"Antes de responder, considere: ¿la población carcelaria es un indicador de efectividad en el combate al delito o más bien de punitividad del sistema de justicia?\n\n{ctx()} el corpus disponible no contiene información específica sobre la población carcelaria en México. Los documentos se centran en incidencia delictiva, homicidios y percepción de seguridad, no en el sistema penitenciario. {fin()} no es posible ofrecer una respuesta fundamentada con los documentos disponibles. [Documento 1; Documento 2; Documento 3]",
    level=3,
)
make(
    "¿Qué impacto económico tiene la violencia en las empresas mexicanas?",
    f"{ctx()} la ENVIPE 2025 reporta un costo nacional de 269.6 mil millones de pesos por inseguridad y delito, equivalente al 1.07% del PIB. {con()} sin embargo, no desglosa este costo por sector empresarial ni distingue entre hogares y empresas. {con()} el ONC menciona los costos no monetarios pero tampoco cuantifica el impacto empresarial específico. {fin()} la información disponible permite afirmar que el impacto económico es significativo, pero no ofrece un desglose sectorial. [Documento 2, Pág 4; Documento 1]",
    level=2,
)
make(
    "¿Qué concluye el ONC sobre la efectividad de las políticas de seguridad?",
    f"{ctx()} antes de presentar las conclusiones, permítame preguntar: ¿qué expectativas tiene usted sobre lo que debería lograr una política de seguridad exitosa?\n\n{con()} el ONC concluye que las políticas de seguridad deben ir más allá de la confrontación directa con el crimen organizado. {con()} señala que la reducción de homicidios no significa que el problema esté resuelto, especialmente por el subregistro existente. {con()} recomienda fortalecer las instituciones de procuración de justicia, mejorar el registro de datos y abordar las causas estructurales de la violencia. {fin()} la principal conclusión es que no existe una solución única: se requiere un enfoque integral que combine prevención, justicia y reconstrucción institucional. [Documento 1, Pág 249]",
    level=3,
)
make(
    "¿Cuántos delitos de extorsión se registraron en Jalisco en 2022?",
    f"{ctx()} según datos del IIEG con base en la ENVIPE, Jalisco reportó una tasa de extorsión de 3,517 víctimas por cada 100 mil habitantes en 2022. {con()} esta cifra representa una disminución significativa respecto a 2016, cuando la tasa era de 10,071. {con()} la tendencia a la baja es consistente a lo largo del periodo 2016-2022. {fin()} Jalisco muestra una mejora sostenida en el indicador de extorsión. [Documento 2]",
    level=1,
)

# ═══════════════════════════════════════════════════════════════════
# ANTI-HALLUCINATION — preguntas donde debe decir "no disponible"
# ═══════════════════════════════════════════════════════════════════

make(
    "¿Cuántos homicidios hubo exactamente en Chihuahua en 2012?",
    f"Antes de responder: ¿está usted seguro de que el corpus contiene ese dato específico? Conviene verificar si la cifra solicitada está disponible en los documentos.\n\n{ctx()} el corpus contiene información general sobre la evolución del homicidio en México y menciona que Chihuahua es una de las entidades con tasas elevadas, pero no se encontró una cifra exacta de homicidios para Chihuahua en 2012 en los documentos analizados. {fin()} no es posible ofrecer una cifra exacta con los documentos disponibles. [Documento 1]",
    level=3,
)
make(
    "¿Cuál fue la tasa de homicidios en Colima en 2023?",
    f"{ctx()} antes de dar una cifra, reflexionemos: ¿el corpus incluye datos de 2023? Los documentos disponibles cubren periodos específicos que pueden no coincidir con el año solicitado.\n\n{con()} el corpus documental disponible (ONC Homicidio, ENVIPE 2025 y ONC Anual 2025) no contiene datos desagregados por estado para 2023. Los periodos analizados en los documentos no alcanzan ese año. {fin()} no es posible ofrecer la tasa solicitada con los documentos disponibles. [Documento 1; Documento 2; Documento 3]",
    level=3,
)
make(
    "¿Cuántas personas fueron víctimas de extorsión en 2022?",
    f"{ctx()} antes de responder, permítame preguntar: ¿qué fuente esperaría usted que contenga ese dato? La ENVIPE 2025 reporta cifras de victimización, pero es importante distinguir entre tasas por cada 100 mil habitantes y cifras absolutas.\n\n{con()} la ENVIPE 2025 reporta una tasa de extorsión de 3,517 víctimas por cada 100 mil habitantes a nivel nacional, pero no proporciona el número total absoluto de víctimas de extorsión para 2022. {fin()} no es posible ofrecer una cifra absoluta con los documentos disponibles. [Documento 2]",
    level=3,
)
make(
    "¿Qué porcentaje del PIB representa la violencia en México?",
    f"{ctx()} el corpus disponible indica que el costo de la inseguridad y el delito asciende a 269.6 mil millones de pesos, equivalente al 1.07% del PIB nacional. {con()} esta cifra proviene de la ENVIPE 2025 y representa el costo total, no solo de la violencia homicida. {fin()} el 1.07% del PIB es la cifra documentada. [Documento 2, Pág 4]",
    level=1,
)
make(
    "¿Cuántos feminicidios se registraron en 2024?",
    f"Antes de responder, permítame preguntar: ¿está usted seguro de que el corpus distingue entre homicidio doloso y feminicidio a nivel de cifras específicas?\n\n{ctx()} el ONC aborda el feminicidio como una categoría dentro del homicidio doloso, pero no se encontraron cifras específicas de feminicidios para 2024 en los documentos analizados. La ENVIPE 2025 tampoco desglosa por tipo de homicidio. {fin()} no es posible ofrecer una cifra específica de feminicidios con los documentos disponibles. [Documento 1; Documento 2]",
    level=3,
)
make(
    "¿Qué relación hay entre narcotráfico y homicidios en México?",
    f"{ctx()} el ONC documenta que la confrontación entre el Estado y los cárteles de la droga a partir de 2007 incrementó drásticamente la tasa de homicidios. {con()} la fragmentación de grupos criminales y las disputas territoriales se identifican como factores clave. {con()} el corpus señala que la violencia no puede atribuirse únicamente al narcotráfico, sino también a la debilidad institucional. {fin()} la relación está documentada pero es multicausal. [Documento 1, Pág 29]",
    level=2,
)
make(
    "¿Cuántos homicidios de periodistas hubo en 2014?",
    f"{ctx()} antes de dar una cifra, preguntémonos: ¿el corpus contiene un recuento específico de homicidios de periodistas?\n\n{con()} el ONC dedica una sección a periodistas asesinados, pero las cifras disponibles en el corpus no permiten determinar un número exacto para 2014. La información sobre este tema está dispersa y no es concluyente. {fin()} no es posible ofrecer una cifra exacta con los documentos disponibles. [Documento 1]",
    level=3,
)
make(
    "¿Qué impacto tiene la violencia en la economía mexicana?",
    f"{ctx()} la ENVIPE 2025 estima que el costo de la inseguridad y el delito asciende a 269.6 mil millones de pesos, equivalente al 1.07% del PIB. {con()} este costo incluye pérdidas directas de hogares y empresas, así como gastos en medidas preventivas. {con()} el ONC complementa con los costos no monetarios: impacto psicológico, pérdida de capital social y erosión de la confianza institucional. {fin()} el impacto económico es significativo y multidimensional. [Documento 2, Pág 4; Documento 1, Pág 235]",
    level=2,
)
make(
    "¿Cuál es la fuente de tus datos?",
    f"{ctx()} los datos provienen de tres documentos incluidos en el corpus: 1) ONC - Homicidio: Una mirada a la violencia en México (266 páginas), 2) INEGI - ENVIPE 2025 (encuesta nacional de victimización), y 3) ONC - Reporte Anual de Seguridad 2025. {con()} las respuestas se basan exclusivamente en estos documentos, recuperados mediante búsqueda semántica (FAISS). {fin()} si un dato no está en estos documentos, no es posible ofrecer una respuesta fundamentada. [Documento 1; Documento 2; Documento 3]",
    level=1,
)
make(
    "¿Cuántos delitos se reportaron en Nuevo León en 2023?",
    f"{ctx()} antes de responder, considere: ¿el corpus contiene datos desagregados por estado para 2023?\n\n{con()} el corpus disponible no incluye cifras de incidencia delictiva desagregadas por entidad federativa para el año 2023. La ENVIPE 2025 reporta datos nacionales y estatales, pero los periodos de referencia no siempre coinciden con el año solicitado. {fin()} no es posible ofrecer una respuesta fundamentada con los documentos disponibles. [Documento 2]",
    level=3,
)
make(
    "¿Qué dice el INEGI sobre la violencia en México?",
    f"{ctx()} el INEGI, a través de la ENVIPE 2025, reporta que 23.1 millones de personas fueron víctimas de delito en 2024, con una cifra negra del 93.2%. {con()} también documenta que el 75.6% de la población percibe inseguridad, y que el costo de la inseguridad asciende a 269.6 mil millones de pesos. {con()} la ENDIREH 2021, citada en el corpus, reporta que el 70.1% de las mujeres ha experimentado violencia a lo largo de su vida. {fin()} el INEGI proporciona la visión estadística más completa sobre violencia y victimización en México. [Documento 2]",
    level=2,
)
make(
    "¿Cuál es la tasa de homicidios de México comparada con Brasil?",
    f"Antes de comparar, permítame preguntar: ¿qué indicador usaría para comparar? ¿Tasa por cada 100 mil habitantes, número absoluto de homicidios, o percepción de seguridad?\n\n{ctx()} el corpus disponible se centra exclusivamente en México y no incluye datos comparativos con Brasil ni con ningún otro país. {con()} el ONC menciona que México es percibido internacionalmente como un país violento, pero no ofrece tablas comparativas internacionales. {fin()} no es posible realizar una comparación fundamentada con los documentos disponibles. [Documento 1, Pág 10]",
    level=3,
)
make(
    "¿Qué datos hay sobre violencia en el Estado de México?",
    f"{ctx()} según la ENVIPE 2025, el Estado de México presenta la tasa más alta de victimización general con 34,851 víctimas por cada 100 mil habitantes. {con()} también es la entidad con mayor prevalencia de violencia contra las mujeres (78.7%). {con()} el ONC menciona al Estado de México dentro de la tipología de comportamiento de homicidio doloso, aunque no proporciona una tasa específica. {fin()} los datos disponibles indican que es una de las entidades más afectadas por la violencia. [Documento 2; Documento 1]",
    level=2,
)
make(
    "¿Cuántos niños fueron víctimas de violencia en México?",
    f"Antes de responder, reflexionemos: ¿el corpus contiene datos desagregados por edad de la víctima?\n\n{ctx()} el corpus disponible no contiene información específica sobre victimización infantil. Los documentos se centran en la población general (18 años y más para la ENVIPE) y no desglosan por grupos de edad menores. {con()} la ENDIREH 2021 reporta violencia contra mujeres adultas, no contra niñas. {fin()} no es posible ofrecer una respuesta fundamentada sobre victimización infantil con los documentos disponibles. [Documento 2]",
    level=3,
)
make(
    "¿Qué efectividad tienen las políticas de seguridad en México?",
    f"{ctx()} antes de evaluar efectividad, permítame preguntar: ¿qué métrica usaría para medir éxito? ¿Reducción de homicidios, aumento de denuncias, o mejora en percepción de seguridad?\n\n{con()} el ONC concluye que la reducción de homicidios no implica que el problema esté resuelto, especialmente por el subregistro existente. {con()} señala que las estrategias exclusivamente militares han tenido costos humanos significativos y que se requiere un enfoque integral. {fin()} el corpus sugiere que la efectividad es limitada y que no existe una solución única. [Documento 1, Pág 249]",
    level=3,
)
make(
    "¿Cuántas personas fueron asesinadas en México en 2010?",
    f"{ctx()} antes de dar una cifra exacta, verifiquemos: ¿el corpus contiene el número total de homicidios para 2010?\n\n{con()} el ONC reporta que la tasa de homicidios alcanzó un pico de casi 25 por cada 100 mil habitantes en 2011, y documenta más de 100,000 muertes en 10 años asociadas al conflicto. {con()} sin embargo, no se encontró una cifra exacta de homicidios para 2010 en los documentos analizados. {fin()} no es posible ofrecer una cifra exacta para 2010 con los documentos disponibles. [Documento 1, Pág 10]",
    level=3,
)
make(
    "¿El corpus incluye datos de 2025?",
    f"{ctx()} el corpus incluye la ENVIPE 2025, que es una encuesta realizada en 2025 pero que reporta datos de victimización del año 2024. {con()} el ONC - Reporte Anual 2025 también es de 2025 y cubre eventos del periodo. {con()} el ONC - Homicidio es un estudio con datos históricos hasta 2014. {fin()} sí hay documentos de 2025, pero los datos que contienen son retrospectivos. [Documento 2; Documento 3; Documento 1]",
    level=1,
)
make(
    "¿Qué dice el ONC sobre el robo de vehículos?",
    f"{ctx()} el ONC señala que el robo de vehículo tiene la tasa más alta de prevalencia entre los delitos, con 8,921 víctimas por cada 100 mil habitantes según la ENVIPE 2025. {con()} el ONC publicó un estudio específico sobre robo de vehículo en 2014, mencionado en el corpus. {con()} el robo de vehículo está correlacionado con la tasa de homicidio doloso en varias entidades. {fin()} es el delito de mayor prevalencia en México. [Documento 2; Documento 1]",
    level=1,
)
make(
    "¿Qué organizaciones criminales menciona el ONC?",
    f"{ctx()} el ONC menciona a 'La Familia Michoacana' como una organización criminal con presencia en Tierra Caliente y células en alcaldías de la Ciudad de México. {con()} también documenta la fragmentación de grupos criminales como factor de violencia, aunque no proporciona un listado exhaustivo de todas las organizaciones. {con()} el corpus se enfoca más en el fenómeno del homicidio que en el catálogo de grupos delictivos. {fin()} la información disponible sobre organizaciones específicas es limitada. [Documento 3; Documento 1]",
    level=2,
)
make(
    "¿Cuántas fosas clandestinas se han encontrado en México?",
    f"{ctx()} el ONC dedica un capítulo completo a las fosas clandestinas, pero no proporciona una cifra nacional consolidada del número total de fosas encontradas. {con()} el corpus menciona que el hallazgo de múltiples fosas evidencia la crisis de desapariciones forzadas, pero los datos disponibles son fragmentarios y no permiten una cuantificación precisa. {fin()} no es posible ofrecer una cifra exacta con los documentos disponibles. [Documento 1, Pág 221]",
    level=3,
)
make(
    "¿Qué dice el corpus sobre la violencia en Sinaloa?",
    f"{ctx()} el ONC menciona a Sinaloa dentro de las entidades con dinámicas de violencia particulares. {con()} Sinaloa aparece en los listados de entidades del grupo A de la tipología de homicidio doloso. {con()} la ENVIPE 2025 no proporciona datos específicos para Sinaloa que permitan un análisis detallado. {fin()} la información sobre Sinaloa en el corpus es limitada y no permite un análisis exhaustivo. [Documento 1]",
    level=2,
)
make(
    "¿Qué dice la ENVIPE sobre la confianza en la policía?",
    f"{ctx()} la ENVIPE 2025 reporta que el 75.6% de la población percibe inseguridad en su entidad federativa, lo que sugiere niveles bajos de confianza en las instituciones de seguridad. {con()} la cifra negra del 93.2% se atribuye en parte a la desconfianza en las autoridades y la percepción de que denunciar no sirve. {fin()} la desconfianza institucional es tanto causa como consecuencia de la alta cifra negra. [Documento 2]",
    level=1,
)
make(
    "¿Cuántos secuestros ocurrieron en 2024?",
    f"Antes de responder: ¿el corpus incluye datos específicos sobre secuestro?\n\n{ctx()} el corpus disponible no contiene información específica sobre secuestro como categoría delictiva independiente. Los documentos se centran en homicidio doloso, victimización general (ENVIPE) y percepción de seguridad. {con()} la ENVIPE 2025 mide victimización general pero no desglosa secuestro como categoría específica. {fin()} no es posible ofrecer una respuesta fundamentada con los documentos disponibles. [Documento 1; Documento 2; Documento 3]",
    level=3,
)
make(
    "¿Cómo ha cambiado la violencia en México durante el sexenio de AMLO?",
    f"{ctx()} antes de responder, permítame preguntar: ¿el corpus cubre el periodo del sexenio de AMLO (2018-2024)?\n\n{con()} el ONC Homicidio cubre hasta 2014, la ENVIPE 2025 reporta datos de 2024, y el ONC Anual 2025 cubre eventos recientes. {con()} sin embargo, el corpus no analiza explícitamente periodos presidenciales ni ofrece conclusiones sobre administraciones específicas. {fin()} no es posible ofrecer un análisis fundamentado por periodo presidencial con los documentos disponibles. [Documento 1; Documento 2; Documento 3]",
    level=3,
)
make(
    "¿Qué es el SESNSP?",
    f"{ctx()} el Secretariado Ejecutivo del Sistema Nacional de Seguridad Pública (SESNSP) es el órgano administrativo encargado de recopilar y publicar las estadísticas oficiales de incidencia delictiva en México. {con()} según el corpus, las estadísticas oficiales solo se reportan por el Centro Nacional de Información del SESNSP. {con()} el INEGI genera sus propias cifras a través de encuestas como la ENVIPE. {fin()} ambas fuentes son complementarias pero usan metodologías distintas. [Documento 2; Documento 3]",
    level=1,
)
make(
    "¿Cuál es la diferencia entre tasa de prevalencia y tasa de incidencia?",
    f"{ctx()} la tasa de prevalencia mide el número de víctimas de delito por cada 100 mil habitantes, mientras que la tasa de incidencia mide el número total de delitos (una víctima puede sufrir múltiples delitos). {con()} según la ENVIPE 2025, en 2024 la prevalencia fue de 24,135 víctimas por cada 100 mil habitantes, mientras que la incidencia fue de 34,918 delitos por cada 100 mil habitantes. {fin()} la diferencia refleja que cada víctima sufre en promedio 1.45 delitos. [Documento 2, Pág 4]",
    level=1,
)

# ═══════════════════════════════════════════════════════════════════
# ANTI-INVENCIÓN DE FUENTES — no inventar siglas, acrónimos ni referencias
# ═══════════════════════════════════════════════════════════════════

make(
    "¿Qué dice la CNDH sobre la violencia en México?",
    f"{ctx()} el corpus disponible no contiene información específica de la Comisión Nacional de los Derechos Humanos (CNDH). Los documentos analizados —ONC Homicidio, INEGI ENVIPE 2025 y ONC Anual 2025— no citan reportes ni declaraciones de la CNDH. {fin()} no es posible ofrecer una respuesta fundamentada con los documentos disponibles. [Documento 1; Documento 2; Documento 3]",
    level=3,
)
make(
    "¿Qué opina la UNAM sobre la inseguridad?",
    f"Antes de responder, permítame preguntar: ¿estamos seguros de que la UNAM aparece en el corpus?\n\n{ctx()} el corpus disponible no contiene información específica sobre la posición o estudios de la Universidad Nacional Autónoma de México (UNAM) respecto a la inseguridad. Los documentos analizados no incluyen reportes académicos de esta institución. {fin()} no es posible ofrecer una respuesta fundamentada con los documentos disponibles. [Documento 1; Documento 2; Documento 3]",
    level=3,
)
make(
    "¿Qué datos aporta el Banco Mundial sobre violencia en México?",
    f"Antes de responder, verifique: ¿el Banco Mundial está en el corpus?\n\n{ctx()} el corpus disponible no incluye datos, reportes ni publicaciones del Banco Mundial sobre violencia en México. Los documentos analizados se limitan a fuentes nacionales: ONC, INEGI/ENVIPE y ONC Anual. {fin()} no es posible ofrecer una respuesta fundamentada con los documentos disponibles. [Documento 1; Documento 2; Documento 3]",
    level=3,
)
make(
    "¿Qué dice la CIDH sobre la violencia en México?",
    f"{ctx()} antes de responder: ¿el corpus contiene algún documento de la Comisión Interamericana de Derechos Humanos?\n\n{con()} el corpus disponible no contiene información específica de la CIDH sobre violencia en México. Los documentos incluidos se centran en fuentes nacionales y no hacen referencia a organismos internacionales. {fin()} no es posible ofrecer una respuesta fundamentada con los documentos disponibles. [Documento 1; Documento 2; Documento 3]",
    level=3,
)
make(
    "¿Qué instituciones aparecen en el corpus?",
    f"{ctx()} las instituciones mencionadas en el corpus son: ONC (Observatorio Nacional Ciudadano), INEGI (Instituto Nacional de Estadística y Geografía), SESNSP (Secretariado Ejecutivo del Sistema Nacional de Seguridad Pública), y IIEG (Instituto de Información Estadística y Geográfica de Jalisco). {con()} no aparecen otras instituciones como CNDH, UNAM, Banco Mundial, CIDH, OCDE, ni organizaciones internacionales. {fin()} las respuestas deben limitarse a las fuentes documentadas en el corpus. [Documento 1; Documento 2; Documento 3]",
    level=1,
)
make(
    "Compara los datos del INEGI con los de la OCDE sobre homicidios.",
    f"Antes de comparar, preguntémonos: ¿la OCDE aparece en el corpus?\n\n{ctx()} el corpus disponible no contiene datos de la OCDE ni de ninguna otra organización internacional. La comparación solicitada no es posible porque solo se dispone de fuentes nacionales (ONC e INEGI). {fin()} no es posible ofrecer una comparación fundamentada con los documentos disponibles. [Documento 1; Documento 2]",
    level=3,
)
make(
    "¿Qué estudios de la UNAM sobre homicidios están en el corpus?",
    f"{ctx()} el corpus disponible no incluye ningún estudio de la UNAM ni de otras universidades sobre homicidios. Los documentos analizados son reportes institucionales del ONC y del INEGI, no investigaciones académicas universitarias. {fin()} no es posible ofrecer una respuesta fundamentada con los documentos disponibles. [Documento 1; Documento 2; Documento 3]",
    level=3,
)

# ═══════════════════════════════════════════════════════════════════
# GUARDAR
# ═══════════════════════════════════════════════════════════════════

with open(OUTPUT, "w", encoding="utf-8") as f:
    for ex in examples:
        f.write(json.dumps(ex, ensure_ascii=False) + "\n")

n1 = sum(1 for ex in examples if ex["level"] == 1)
n2 = sum(1 for ex in examples if ex["level"] == 2)
n3 = sum(1 for ex in examples if ex["level"] == 3)
print(f"Dataset generado: {len(examples)} ejemplos")
print(f"  Nivel 1 (factoid): {n1}")
print(f"  Nivel 2 (síntesis): {n2}")
print(f"  Nivel 3 (razonamiento): {n3}")
