FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
RUN pip install --no-cache-dir .

ENV MODULE_A_CONFIG_PATH=/app/configs/parameters.yaml
ENV MODULE_A_REFERENCE_PATH=/app/artifacts/reference.joblib

EXPOSE 8000
CMD ["uvicorn", "ess_module_a.api:app", "--host", "0.0.0.0", "--port", "8000"]
