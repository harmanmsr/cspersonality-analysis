#!/bin/bash
# Entrypoint untuk menjalankan TF Serving.
# $PORT dibaca secara dinamis: default 8501 untuk run lokal/Docker biasa,
# otomatis mengikuti $PORT yang disuntikkan platform cloud (Render/Heroku/dll) saat deploy.

set -e

PORT="${PORT:-8501}"

tensorflow_model_server \
  --rest_api_port="${PORT}" \
  --model_name="${MODEL_NAME}" \
  --model_base_path="${MODEL_BASE_PATH}/${MODEL_NAME}" \
  --monitoring_config_file=/model_config/prometheus.config
