"""
FastAPI server pengganti TensorFlow Serving.

Alasan: image Docker `tensorflow/serving:latest` saat ini berbasis TF ~2.19/2.20,
sementara model di-training & di-export dengan tensorflow==2.21.0 (dikunci oleh
tfx==1.21.0). Ketidakcocokan versi ini menyebabkan TF Serving gagal saat predict
("Could not find variable ..."), walau model terbukti sehat saat di-load langsung
lewat tf.saved_model.load() dengan TF 2.21.0.

Server ini menghindari masalah itu sepenuhnya: dia me-load SavedModel langsung
pakai package `tensorflow` Python yang sama persis dengan environment training,
lalu mengekspos endpoint HTTP dengan format request/response yang sama dengan
TF Serving REST API, supaya notebook testing (HarmanM-testing.ipynb) tidak perlu
diubah -- cukup ganti BASE_URL.

Endpoint yang disediakan (kompatibel dengan format TF Serving):
  GET  /v1/models/{model_name}            -> status model
  POST /v1/models/{model_name}:predict     -> prediction request

Cara jalan lokal (tanpa Docker):
  pip install fastapi uvicorn tensorflow==2.21.0
  MODEL_DIR=serving_model/marketing-response-model/1788165141 \
  MODEL_NAME=marketing-response-model \
  uvicorn serving_app:app --host 0.0.0.0 --port 8501
"""

import base64
import os
import time

import tensorflow as tf
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ValidationError


# --- Konfigurasi lewat environment variable, biar gampang disesuaikan tanpa edit kode ---
MODEL_NAME = os.environ.get("MODEL_NAME", "marketing-response-model")
# MODEL_DIR harus menunjuk ke folder versi SavedModel yang spesifik, contoh:
# serving_model/marketing-response-model/(nomor_model)
MODEL_DIR = os.environ.get("MODEL_DIR")

if not MODEL_DIR:
    raise RuntimeError(
        "Environment variable MODEL_DIR wajib diisi, contoh: "
        "MODEL_DIR=serving_model/marketing-response-model/1788165141"
    )

app = FastAPI(title="Marketing Response Model Server (FastAPI)")

# --- Load model sekali saat startup ---
_model = tf.saved_model.load(MODEL_DIR)
_infer = _model.signatures["serving_default"]
_model_version = os.path.basename(os.path.normpath(MODEL_DIR))
_load_time = time.time()


class PredictInstance(BaseModel):
    b64: str


class PredictRequest(BaseModel):
    instances: list[PredictInstance]


@app.get("/v1/models/{model_name}")
def model_status(model_name: str):
    if model_name != MODEL_NAME:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' tidak ditemukan")
    return {
        "model_version_status": [
            {
                "version": _model_version,
                "state": "AVAILABLE",
                "status": {"error_code": "OK", "error_message": ""},
            }
        ]
    }


@app.post("/v1/models/{model_name}:predict")
async def predict(model_name: str, request: Request):
    if model_name != MODEL_NAME:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' tidak ditemukan")

    # Baca & parse body secara manual (bukan lewat parameter Pydantic otomatis),
    # supaya tidak bergantung pada header Content-Type. TF Serving asli juga tidak
    # peduli Content-Type, dan notebook testing mengirim lewat `requests.post(...,
    # data=json.dumps(payload))` yang tidak menyertakan header itu.
    try:
        raw_body = await request.json()
        payload = PredictRequest.model_validate(raw_body)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc
    except Exception as exc:  # noqa: BLE001, badan bukan JSON valid sama sekali
        raise HTTPException(status_code=400, detail=f"Body bukan JSON valid: {exc}") from exc

    try:
        # Setiap instance dikirim sebagai base64 dari tf.Example yang sudah di-serialize,
        # persis format yang dipakai HarmanM-testing.ipynb (build_request_payload()).
        serialized_examples = [
            base64.b64decode(inst.b64) for inst in payload.instances
        ]
        examples_tensor = tf.constant(serialized_examples)
        result = _infer(examples=examples_tensor)
        # Signature model ini punya satu output bernama 'output_0' (lihat structured_outputs
        # yang sudah kita cek sebelumnya). TF Serving asli menamainya 'predictions' di response
        # JSON, jadi kita samakan supaya notebook testing kamu tidak perlu berubah.
        predictions = result["output_0"].numpy().tolist()
        return {"predictions": predictions}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/healthz")
def healthz():
    return {"status": "ok", "model_version": _model_version, "uptime_seconds": time.time() - _load_time}