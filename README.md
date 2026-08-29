# Machine Learning Pipeline dengan TensorFlow Extended (TFX)
## Prediksi Response Campaign Marketing

**Nama:** HarmanM
**Folder pipeline:** `cspersonality-analysis/`

---

## 1. Informasi Dataset

Dataset yang digunakan adalah `marketing_campaign.csv`, berisi **2.240 data pelanggan** dari sebuah perusahaan retail yang pernah menerima serangkaian campaign marketing. Dataset ini memuat 29 kolom, di antaranya:

| Kategori | Kolom |
|---|---|
| Demografis | `Year_Birth`, `Education`, `Marital_Status`, `Income`, `Kidhome`, `Teenhome` |
| Riwayat transaksi | `Recency`, `MntWines`, `MntFruits`, `MntMeatProducts`, `MntFishProducts`, `MntSweetProducts`, `MntGoldProds` |
| Perilaku pembelian | `NumDealsPurchases`, `NumWebPurchases`, `NumCatalogPurchases`, `NumStorePurchases`, `NumWebVisitsMonth` |
| Histori campaign | `AcceptedCmp1`–`AcceptedCmp5`, `Complain` |
| **Target** | `Response` (1 = menerima campaign terakhir, 0 = tidak) |

## 2. Persoalan yang Ingin Diselesaikan

Response rate campaign marketing historis perusahaan ini rendah (rata-rata di bawah 15% per campaign). Mengirim campaign secara massal ke seluruh pelanggan (*mass marketing*) menghasilkan biaya tinggi dengan tingkat konversi yang rendah. Perusahaan membutuhkan cara untuk mengidentifikasi pelanggan yang paling mungkin merespons sebuah campaign **sebelum** campaign tersebut diluncurkan, agar budget marketing bisa dialokasikan secara lebih efisien.

## 3. Solusi Machine Learning & Target

Solusi yang dibangun adalah **model klasifikasi biner** yang memprediksi probabilitas seorang pelanggan akan merespons (`Response = 1`) sebuah campaign, berdasarkan profil demografis dan histori transaksinya.

**Target performa:**
- AUC ≥ 0.80 pada data evaluasi (cukup untuk *ranking* pelanggan berdasarkan probabilitas respons)
- Pipeline modular dan *reproducible* menggunakan TFX, sehingga mudah di-retrain saat ada data pelanggan baru

## 4. Metode Pengolahan Data, Arsitektur Model, dan Metrik Evaluasi

### 4.1 Pengolahan Data
- **Cleaning**: mengisi `Income` kosong dengan median, membuang outlier usia (>100 tahun) dan income ekstrem (>99th percentile), menstandarkan kategori kotor pada `Marital_Status` (`Alone`/`Absurd`/`YOLO` → `Single`).
- **Feature engineering**: `Age` (dari `Year_Birth`), `Customer_Tenure_Days` (dari `Dt_Customer`), `Total_Spending`, `Total_Children`, `Total_Purchases`.
- **Di dalam komponen `Transform` TFX** (`marketing_transform.py`): fitur numerik di-scale ke rentang [0,1] dengan `tft.scale_to_0_1`; fitur kategorikal (`Education`, `Marital_Status`) diubah menjadi index vocabulary dengan `tft.compute_and_apply_vocabulary`.

### 4.2 Hyperparameter Tuning & Arsitektur Model
- **Tuning otomatis** (`marketing_tuner.py`, komponen `Tuner`): KerasTuner `RandomSearch` (10 trial) mencari kombinasi terbaik `units_1`, `units_2`, `dropout`, `learning_rate`, dioptimalkan terhadap `val_auc`. Hasil terbaik diteruskan ke `Trainer`.
- **Arsitektur model** (`marketing_trainer.py`): DNN Keras *feed-forward*.
  - Fitur numerik (26 fitur) digabung langsung sebagai input.
  - Fitur kategorikal (`Education`, `Marital_Status`) melalui `Embedding` layer (dim=4), lalu di-flatten.
  - Semua fitur digabung → `Dense(units_1, relu)` → `Dropout(dropout)` → `Dense(units_2, relu)` → `Dense(1, sigmoid)`, dengan `units_1`/`units_2`/`dropout` hasil Tuner.
  - Optimizer Adam (learning rate hasil Tuner), loss `binary_crossentropy`, dengan `EarlyStopping` berbasis `val_auc`.

### 4.3 Metrik Evaluasi
Dievaluasi menggunakan **TensorFlow Model Analysis (TFMA)** di komponen `Evaluator`:
- **AUC** — metrik utama, cocok untuk data imbalanced (`Response=1` hanya ±15% populasi).
- **Binary Accuracy** — dengan threshold validasi minimal 0.5 dibanding baseline (mekanisme *blessing*).
- **Example Count** — memastikan jumlah data evaluasi sesuai ekspektasi.

## 5. Performa Model

| Metrik | Nilai (TFMA, eval set 429 contoh) |
|---|---|
| AUC | 0.9116 |
| Binary Accuracy | 0.9184 |

*(Dihitung oleh komponen Evaluator/TFMA dari model tersimpan terhadap seluruh eval set, setelah dilatih dengan hyperparameter hasil komponen `Tuner` — lebih otoritatif dibanding log per-epoch saat training.)*

Model dinyatakan **"blessed"** oleh komponen `Evaluator` (memenuhi threshold minimum) dan berhasil di-push oleh komponen `Pusher` ke direktori `serving_model/marketing-response-model/`, siap untuk tahap deployment.

## 6. Struktur Pipeline (TFX Components)

Seluruh komponen dijalankan via `InteractiveContext` di dalam notebook `cspersonality-analysis.ipynb`:

1. **ExampleGen** — membaca CSV, split train (80%) / eval (20%), konversi ke TFRecord
2. **StatisticsGen** — menghasilkan statistik deskriptif tiap fitur
3. **SchemaGen** — menyimpulkan skema data secara otomatis
4. **ExampleValidator** — mendeteksi anomali data berdasarkan skema
5. **Transform** — feature engineering (scaling numerik, vocabulary kategorikal) — modul: `marketing_transform.py`
6. **Tuner** — hyperparameter tuning otomatis (KerasTuner RandomSearch) — modul: `marketing_tuner.py`
7. **Trainer** — training model DNN Keras dengan hyperparameter hasil Tuner — modul: `marketing_trainer.py`
8. **Resolver** — mencari model blessed terbaik sebelumnya sebagai baseline
9. **Evaluator** — evaluasi model dengan TFMA, menentukan status *blessed*
10. **Pusher** — mendorong model yang blessed ke direktori serving

## 7. Struktur Folder

```
cspersonality-analysis/
├── cspersonality-analysis.ipynb   # Notebook utama (seluruh pipeline + dokumentasi), SUDAH DIJALANKAN
├── HarmanM-testing.ipynb          # Notebook uji prediction request ke model serving
├── README.md                      # Dokumentasi proyek (file ini)
├── DEPLOYMENT.md                  # Panduan Kriteria 3 & 4: deploy Railway + monitoring Prometheus
├── requirements.txt               # Daftar dependency Python
├── marketing_transform.py         # Modul preprocessing (salinan kerja, versi resmi ada di modules/)
├── marketing_tuner.py             # Modul hyperparameter tuning (salinan kerja, versi resmi ada di modules/)
├── marketing_trainer.py           # Modul arsitektur & training (salinan kerja, versi resmi ada di modules/)
├── modules/                       # Wajib: seluruh modul pipeline (Saran 1 - Tuner)
│   ├── marketing_transform.py
│   ├── marketing_tuner.py
│   ├── marketing_trainer.py
│   ├── .pylintrc
│   └── README.md
├── HarmanM-pipeline/               # Wajib: direktori berisi seluruh komponen ML pipeline (artefak TFX)
├── data/
│   └── marketing_campaign_clean.csv
├── serving_model/
│   └── marketing-response-model/  # Model final hasil Pusher
├── Dockerfile                     # Wajib: untuk menjalankan sistem ML di cloud (dipakai deploy ke Railway)
├── entrypoint.sh                   # Script start TF Serving (port dinamis + monitoring aktif)
├── docker-compose.yml              # Test image secara lokal
├── .dockerignore
├── monitoring/                     # Wajib: seluruh kebutuhan Prometheus
│   ├── Dockerfile                  # Wajib: untuk menjalankan Prometheus
│   ├── prometheus.config           # Mengaktifkan endpoint metrics TF Serving (dipakai juga oleh root Dockerfile)
│   ├── prometheus.yml
│   └── docker-compose.prometheus.yml
├── render.yaml                     # Opsional: alternatif deploy ke Render
└── huggingface-space/               # Opsional: alternatif deploy ke Hugging Face Spaces (tidak dipakai)
```

**Screenshot yang perlu dilampirkan terpisah (bukan di dalam folder ini):**
- `HarmanM-deployment.png` — bukti model bisa diakses dari cloud (Railway)
- `HarmanM-monitoring.png` — dashboard Prometheus (target status UP)
- `HarmanM-pylint.png` — hasil `pylint modules/`
- `HarmanM-grafana-dashboard.png` — hanya jika menerapkan saran ke-4 (Grafana)

## 8. Model Deployment & Monitoring (Kriteria 3 & 4)

### 8.1 Opsi Platform Deployment

Beberapa platform cloud dipertimbangkan untuk menjalankan sistem ini (via
Docker/TF Serving), dengan pertimbangan utama: gratis dan tidak memerlukan
kartu kredit untuk keperluan submission ini.

| Platform | Hasil evaluasi |
|---|---|
| Heroku | Sudah tidak punya free tier sejak 2022 — berbayar (~$5/bulan) |
| Render | Free tier tersedia, tapi sign up sekarang mewajibkan info kartu kredit |
| Hugging Face Spaces | Gratis, tanpa kartu kredit — sempat disiapkan (lihat folder `huggingface-space/`), tapi butuh port tetap (7860) dan struktur repo Git terpisah |
| **Railway** ✅ | **Dipilih** — free trial credit ($5/30 hari) tanpa kartu kredit di awal, deploy langsung dari GitHub repo, mendukung port dinamis via `$PORT` (sama seperti Dockerfile yang sudah disiapkan) |

**Platform yang dipakai: Railway.** Sistem di-deploy dari `Dockerfile` di
root project (base image `tensorflow/serving`), yang menjalankan TF Serving
dan meng-expose REST API serta endpoint monitoring Prometheus bawaan
(`/monitoring/prometheus/metrics`).

### 8.2 Tautan Web App

**🔗 Tautan yang sudah dideploy (bisa langsung diakses/diverifikasi):**

| Endpoint | URL |
|---|---|
| Model status | https://cspersonality-analysis-production.up.railway.app/v1/models/marketing-response-model |
| Prometheus metrics (TF Serving) | https://cspersonality-analysis-production.up.railway.app/monitoring/prometheus/metrics |
| Prediction endpoint (POST) | https://cspersonality-analysis-production.up.railway.app/v1/models/marketing-response-model:predict |

Cek cepat via `curl`:
```bash
curl https://cspersonality-analysis-production.up.railway.app/v1/models/marketing-response-model
```
Response yang diharapkan: `"state": "AVAILABLE"` (lihat screenshot `HarmanM-deployment.png`).

### 8.3 Hasil Monitoring

Prometheus dijalankan secara lokal (`monitoring/docker-compose.prometheus.yml`),
mengambil data dari endpoint `/monitoring/prometheus/metrics` milik TF
Serving yang berjalan di Railway, dengan scrape interval 15 detik.

**Hasil yang diamati** (lihat screenshot `HarmanM-monitoring.png`):
- **`up{job="tf-serving-marketing-model"}` = 1** — target berhasil di-scrape secara konsisten (status **UP**), menandakan endpoint model di Railway aktif dan bisa dijangkau Prometheus sepanjang waktu observasi.
- **`:tensorflow:core:graph_runs`** — metrik bawaan TF Serving yang menghitung jumlah eksekusi graph model; grafik menunjukkan aktivitas eksekusi bertambah setiap ada request masuk ke model (misalnya saat dites lewat `curl` atau notebook `HarmanM-testing.ipynb`), membuktikan monitoring benar-benar menangkap aktivitas nyata model, bukan cuma status hidup/mati.

Kesimpulan: sistem yang di-deploy ke Railway berhasil dipantau end-to-end
lewat Prometheus tanpa komponen tambahan (exporter terpisah), karena
memanfaatkan endpoint monitoring bawaan TF Serving.

Langkah setup lengkap ada di **[`DEPLOYMENT.md`](./DEPLOYMENT.md)**.

## 9. Perbaikan Berdasarkan Saran Reviewer (Kriteria 1 & 2)

| Saran | Status | Keterangan |
|---|---|---|
| 1. Hyperparameter tuning otomatis | ✅ Ditambahkan | Komponen `Tuner` (KerasTuner RandomSearch) di `marketing_tuner.py`, hasilnya dipakai `Trainer` — lihat bagian 3.5b & 3.6 di notebook |
| 2. Model deployment dengan TF Serving (Dockerfile + screenshot) | ✅ Dilengkapi | `Dockerfile` di root project murni untuk serving model (bukan host Jupyter) — lihat `DEPLOYMENT.md` bagian A untuk cara test lokal & mengambil screenshot bukti |
| 3. Notebook uji prediction request | ✅ Ditambahkan | `test_serving_prediction.ipynb` — mengirim request ke endpoint `:predict` dan menampilkan hasil prediksi |

## 10. Potensi Pengembangan Lanjutan

- Menambahkan orkestrasi produksi (Apache Beam/Airflow/Kubeflow) menggantikan `InteractiveContext` yang bersifat eksperimental/lokal.
- Eksperimen arsitektur model lain (misalnya Gradient Boosted Trees) dan hyperparameter tuning menggunakan komponen `Tuner`.
- Menambahkan komponen `BulkInferrer` untuk scoring seluruh basis pelanggan secara batch sebelum campaign berikutnya diluncurkan.