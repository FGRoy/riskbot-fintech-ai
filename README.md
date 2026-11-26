# RiskBot API · FinTech-AI

Asistente de políticas de crédito para analistas de riesgos, servido como API REST (FastAPI) con defensa en profundidad y despliegue continuo condicionado a una evaluación automática.

## Arquitectura

| Capa | Control |
|---|---|
| Contrato (Pydantic) | 5-300 caracteres, solo caracteres seguros, `extra=forbid`, modelo en lista blanca (OWASP LLM10) |
| Juez semántico | LLM-Judge previo a la inferencia; falla cerrado (solo un `FALSE` limpio pasa) → 403 |
| RAG + system prompt | Políticas de `politicas_credito.json` (carga global única), roles separados con etiquetas XML, abstención fuera del JSON |
| Salida | Límite de tokens, safety settings del proveedor y filtro anti-fuga de instrucciones |
| Errores | Mensajes genéricos (400/403/502/503); sin trazas ni lógica interna |
| Secretos | `GEMINI_API_KEY` solo por variable de entorno; `.env` ignorado por Git y Docker |
| Contenedor | `python:3.12-slim`, usuario sin privilegios (UID 10001), solo dependencias de producción |

## Uso local

```bash
cp .env.example .env            # rellenar GEMINI_API_KEY
pip install -r requirements.txt
ENABLE_DOCS=1 uvicorn main:app --reload   # Swagger en /docs (desactivado por defecto)
```

```bash
curl -X POST http://localhost:8000/inferencia -H "Content-Type: application/json" \
  -d '{"prompt": "¿Cuál es la tasa de interés del Tier B?"}'
```

Cuerpo: `{"prompt": str, "model": str (opcional)}` · Respuesta: `{"resultado": str}`.

## Docker

```bash
docker build -t riskbot .
docker run --rm -p 8000:8000 -e GEMINI_API_KEY=$GEMINI_API_KEY riskbot
```

## Evaluación (CI)

```bash
pip install -r requirements-test.txt
python eval.py            # EVAL_MODEL=<modelo> para evaluar otro modelo
```

`eval.py` ataca la API con `TestClient`, registra en MLflow y termina con `exit 1` si falla alguna puerta:

| Métrica | Umbral |
|---|---|
| `security_recall` | = 1.0 |
| `false_positive_rate` | = 0.0 |
| `answer_relevancy` (Ragas) | ≥ 0.75 |
| `factual_accuracy` (determinista contra el JSON) | ≥ 0.90 |
| `abstention_rate` (Tier D, fuera de dominio) | = 1.0 |
| `provider_errors` | = 0 |

`faithfulness` se registra y pasa a bloqueante con `UMBRAL_FIDELIDAD`.

## Pruebas

```bash
python -m pytest -q tests                                   # sin red: contrato, fallo cerrado, errores opacos, repositorio
python scripts/audit_live.py https://TU-SERVICIO.onrender.com   # auditoría en vivo (51 casos), genera audit_report_<fecha>.md
```

## CI/CD

`.github/workflows/cicd.yml`: en cada `push` a `main` instala dependencias, ejecuta `pytest` y `eval.py` y, solo si pasa, llama al Deploy Hook de Render. Las ejecuciones manuales y los pull requests evalúan pero no despliegan.

Secretos del repositorio (Settings → Secrets and variables → Actions): `GEMINI_API_KEY`, `RENDER_DEPLOY_HOOK`.
Render: Auto-Deploy en **No**; variable de entorno `GEMINI_API_KEY`.
