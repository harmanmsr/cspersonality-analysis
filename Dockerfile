FROM tensorflow/serving:latest

COPY ./serving_model/marketing-response-model /models/marketing-response-model
COPY ./monitoring/prometheus.config /model_config/prometheus.config

ENV MODEL_NAME=marketing-response-model
ENV MODEL_BASE_PATH=/models

RUN printf '#!/bin/bash\nset -e\nPORT="${PORT:-8501}"\ntensorflow_model_server \\\n  --rest_api_port="${PORT}" \\\n  --model_name="${MODEL_NAME}" \\\n  --model_base_path="${MODEL_BASE_PATH}/${MODEL_NAME}" \\\n  --monitoring_config_file=/model_config/prometheus.config\n' > /usr/bin/tf_serving_entrypoint.sh && \
    chmod +x /usr/bin/tf_serving_entrypoint.sh

EXPOSE $PORT

ENTRYPOINT ["/usr/bin/tf_serving_entrypoint.sh"]