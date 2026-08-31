---
title: CS Personality Marketing Response Model
emoji: 📊
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Marketing Campaign Response Prediction — Model Serving

Model TensorFlow Serving untuk memprediksi probabilitas pelanggan merespons
campaign marketing, di-serve lewat REST API TF Serving.

**Cek status model:**
```bash
curl https://<username-hf>-<nama-space>.hf.space/v1/models/marketing-response-model
```

**Kirim prediction request:**
```bash
curl -X POST https://<username-hf>-<nama-space>.hf.space/v1/models/marketing-response-model:predict \
  -H "Content-Type: application/json" \
  -d '{"instances": [...]}'
```

Lihat notebook `test_serving_prediction.ipynb` di repo utama project untuk
contoh lengkap format request (`tf.Example` ter-base64).

Endpoint monitoring Prometheus juga aktif di `/monitoring/prometheus/metrics`.
