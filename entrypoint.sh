#!/bin/bash
# Entrypoint TF Serving khusus untuk Hugging Face Spaces.
# HF Spaces MEWAJIBKAN container listen di port 7860 (tidak bisa port lain,
# tidak ada env var $PORT dinamis seperti Render/Heroku).

set -e

tensorflow_model_server \
  --rest_api_port=7860 \
  --model_name="${MODEL_NAME}" \
  --model_base_path="${MODEL_BASE_PATH}/${MODEL_NAME}" \
  --monitoring_config_file=/model_config/prometheus.config