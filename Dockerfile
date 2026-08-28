FROM tensorflow/serving:latest

COPY ./serving_model/marketing-response-model /models/marketing-response-model
COPY ./config /model_config
COPY ./entrypoint.sh /usr/bin/tf_serving_entrypoint.sh

ENV MODEL_NAME=marketing-response-model
ENV MODEL_BASE_PATH=/models

RUN chmod +x /usr/bin/tf_serving_entrypoint.sh

ENTRYPOINT ["/usr/bin/tf_serving_entrypoint.sh"]