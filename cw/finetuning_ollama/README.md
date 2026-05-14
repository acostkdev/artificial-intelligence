# Fine-Tuning con Ollama Modelfile

Fine-tuning basico usando el sistema Modelfile de Ollama (sin Python/HuggingFace).

## Requisitos

- [Ollama](https://ollama.ai) instalado
- Modelo base `llama3.2:3b` descargado: `ollama pull llama3.2:3b`
- Dataset en `cw/dataset_tutor_programacion.jsonl`

## Crear el modelo

Desde este directorio, ejecuta:

```bash
ollama create tutor_programacion --file Modelfile
```

Ollama leera el Modelfile, cargara el modelo base, aplicara el system prompt y entrenara con los datos del dataset.

## Usar el modelo

```bash
ollama run tutor_programacion
```

Ejemplo de uso:

```
>>> ¿Que es una funcion en programacion?
```

## Notas

- El Modelfile usa TRAINING_DATA para fine-tuning superficial directamente desde JSONL.
- Para control mas fino (epocas, tasa de aprendizaje, LoRA), usar Python con HuggingFace PEFT.
- El dataset contiene 194 pares instruction/response en español mexicano.
