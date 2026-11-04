# Pipeline do modelo final: dos dados limpos (data/cache_dados_finor.pkl) aos números e figuras do relatório.
#   docker build -t finor-index-tracking .
#   docker run --rm -v "$(pwd)/saida:/app/saida" finor-index-tracking              # reaproveita as simulações (≈ 30 s)
#   docker run --rm -v "$(pwd)/saida:/app/saida" finor-index-tracking --refazer    # refaz tudo a partir dos dados
#   docker run --rm -p 8501:8501 --entrypoint streamlit finor-index-tracking run app.py   # app em http://localhost:8501
FROM python:3.12.8-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

COPY src/ src/
COPY data/cache_dados_finor.pkl data/
COPY data/fontes/ data/fontes/
COPY results/selecao/ results/selecao/
COPY results/teste_final/ results/teste_final/
COPY results/app/ results/app/
COPY app.py .
COPY .streamlit/ .streamlit/

EXPOSE 8501

ENTRYPOINT ["python", "-m", "src.pipeline", "--saida", "saida"]
