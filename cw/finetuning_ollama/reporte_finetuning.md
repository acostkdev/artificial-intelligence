# Reporte de Fine-Tuning con Ollama Modelfile

## Objetivo

Realizar un fine-tuning basico sobre `llama3.2:3b` usando unicamente el
sistema Modelfile de Ollama, sin escribir codigo Python ni usar
HuggingFace. El objetivo es crear un tutor de programacion que explique
conceptos en español mexicano con un tono accesible.

## Dataset

Usamos el dataset existente en `cw/dataset_tutor_programacion.jsonl`.
Contiene 194 pares instruction/response con preguntas reales de
principiantes y respuestas estilo tutorial en español mexicano. Cada
respuesta usa analogias de la vida cotidiana (tacos, taquerias,
compas, elotes) para hacer los conceptos mas accesibles.

No modificamos el dataset porque ya estaba curado de actividades
anteriores.

## Modelfile

Creamos un Modelfile con las siguientes directivas:

- **FROM**: `llama3.2:3b` — modelo base ligero (3B parametros) que
  funciona bien en hardware modesto.
- **SYSTEM**: prompt de sistema que define el rol de tutor de
  programacion en español mexicano con tono amigable pero profesional.
- **TRAINING_DATA**: ruta relativa al archivo JSONL para que Ollama
  lo procese durante la creacion del modelo.
- **PARAMETER**: temperature 0.7 para un balance entre creatividad y
  determinismo, top_p 0.9 para mantener coherencia sin ser demasiado
  restrictivo.

## Proceso

Ejecutamos el comando `ollama create` desde el directorio del proyecto:

```bash
ollama create tutor_programacion --file Modelfile
```

Ollama se encarga de todo el pipeline: carga el modelo base, aplica el
system prompt, procesa el dataset de entrenamiento y genera el modelo
final. No tuvimos que preocuparnos por epocas, batch size ni tasa de
aprendizaje — el sistema Modelfile abstrae todo eso.

## Resultados

El modelo resultante responde preguntas de programacion en el mismo
estilo del dataset: explicaciones con analogias mexicanas, tono
relajado y ejemplos practicos. La principal ventaja de este enfoque
es la simplicidad: en un solo comando tenemos un modelo ajustado.

## Limitaciones

El fine-tuning via Modelfile es superficial comparado con LoRA o
fine-tuning completo con PEFT. No tenemos control sobre hiper-
parametros como epocas, tasa de aprendizaje o cuantizacion. Para
un ajuste mas profundo, usariamos `train_lora.py` con HuggingFace.

Otra limitacion es que el dataset es relativamente pequeño (194
ejemplos). Con mas datos (500+) el modelo generalizaria mejor.
