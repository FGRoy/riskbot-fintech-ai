FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Usuario sin privilegios y sin shell de login (mínimo privilegio)
RUN groupadd --system --gid 10001 app \
 && useradd --system --uid 10001 --gid app --no-create-home --shell /usr/sbin/nologin app

WORKDIR /app

# Solo dependencias de producción (las de test no entran en la imagen)
COPY requirements.txt .
RUN pip install -r requirements.txt

# Solo lo necesario; el código queda de solo lectura para el usuario app.
# La API key NO se copia: se inyecta como variable de entorno en tiempo de ejecución.
COPY main.py politicas_credito.json ./

USER 10001:10001
EXPOSE 8000

# Render inyecta $PORT; en local se usa 8000
CMD ["sh", "-c", "exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --no-server-header"]
