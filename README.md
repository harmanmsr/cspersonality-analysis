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
├── cspersonality-analysis.ipynb      # Notebook utama (seluruh pipeline + dokumentasi)
├── README.md                   # Dokumentasi proyek (file ini)
├── DEPLOYMENT.md                # Panduan Kriteria 3 & 4: deploy Render + monitoring Prometheus
├── marketing_transform.py      # Modul preprocessing untuk komponen Transform
├── marketing_tuner.py          # Modul hyperparameter tuning untuk komponen Tuner
├── marketing_trainer.py        # Modul arsitektur & training untuk komponen Trainer
├── test_serving_prediction.ipynb  # Notebook uji prediction request ke model serving
├── data/
│   └── marketing_campaign_clean.csv
├── pipeline_root/               # Artefak/metadata seluruh komponen TFX
├── serving_model/
│   └── marketing-response-model/  # Model final hasil Pusher
├── Dockerfile                   # Image TF Serving (serving murni, dipakai untuk deploy ke Render)
├── entrypoint.sh                 # Script start TF Serving (port dinamis + monitoring aktif)
├── render.yaml                   # Blueprint config untuk deploy otomatis ke Render (opsional)
├── docker-compose.yml            # Test image secara lokal
├── .dockerignore
├── config/
│   └── prometheus.config         # Mengaktifkan endpoint metrics TF Serving
└── monitoring/
    ├── prometheus.yml                    # Scrape config Prometheus
    └── docker-compose.prometheus.yml     # Menjalankan Prometheus server lokal
```

## 8. Kriteria 3 & 4 — Cloud Deployment & Monitoring

Sesuai submission ini, sistem juga dijalankan pada environment cloud
(**Render**, via Docker/TF Serving) dan dipantau menggunakan **Prometheus**
lewat endpoint metrics bawaan TF Serving (`/monitoring/prometheus/metrics`).

Langkah lengkap deployment dan monitoring ada di **[`DEPLOYMENT.md`](./DEPLOYMENT.md)**.

## 9. Perbaikan Berdasarkan Saran Reviewer (Kriteria 1 & 2)

| Saran | Status | Keterangan |
|---|---|---|
| 1. Hyperparameter tuning otomatis | ✅ Ditambahkan | Komponen `Tuner` (KerasTuner RandomSearch) di `marketing_tuner.py`, hasilnya dipakai `Trainer` — lihat bagian 3.5b & 3.6 di notebook |
| 2. Model deployment dengan TF Serving (Dockerfile + screenshot) | ✅ Dilengkapi | `Dockerfile` di root project murni untuk serving model (bukan host Jupyter) — lihat `DEPLOYMENT.md` bagian A untuk cara test lokal & mengambil screenshot bukti |
| 3. Notebook uji prediction request | ✅ Ditambahkan | `test_serving_prediction.ipynb` — mengirim request ke endpoint `:predict` dan menampilkan hasil prediksi |

## 9. Potensi Pengembangan Lanjutan

- Menambahkan orkestrasi produksi (Apache Beam/Airflow/Kubeflow) menggantikan `InteractiveContext` yang bersifat eksperimental/lokal.
- Eksperimen arsitektur model lain (misalnya Gradient Boosted Trees) dan hyperparameter tuning menggunakan komponen `Tuner`.
- Menambahkan komponen `BulkInferrer` untuk scoring seluruh basis pelanggan secara batch sebelum campaign berikutnya diluncurkan.
