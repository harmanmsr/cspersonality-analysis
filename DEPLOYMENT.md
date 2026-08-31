# Kriteria 3 & 4: Deployment ke Cloud (Railway) & Monitoring (Prometheus)

Panduan ini melanjutkan dari model yang sudah di-push oleh komponen `Pusher`
(folder `serving_model/marketing-response-model/`).

Kita pakai **Railway** — mendukung deploy langsung dari repo GitHub (private
maupun public), free trial credit ($5/30 hari) tanpa perlu kartu kredit di
awal, dan mendukung Docker custom image.

Prasyarat di komputer kamu:
- [Docker](https://www.docker.com/) sudah terinstal dan berjalan (untuk test lokal)
- Akun [Railway](https://railway.app/) (gratis untuk trial, tanpa kartu kredit di awal)
- Akun GitHub, dengan repo project ini sudah di-push (boleh private)

---

## Bagian A — Test Lokal Dulu (sangat disarankan)

Sebelum push ke cloud, pastikan image jalan dengan benar di komputer kamu.

```bash
cd cspersonality-analysis
PORT=8501 docker compose up --build
```

Setelah container jalan, buka di browser/`curl`:

```bash
# Cek model siap
curl http://localhost:8501/v1/models/marketing-response-model

# Cek endpoint Prometheus metrics
curl http://localhost:8501/monitoring/prometheus/metrics
```

Kalau endpoint `/monitoring/prometheus/metrics` mengembalikan teks berformat
Prometheus (banyak baris `# HELP`, `# TYPE`, lalu angka metrik), berarti
Kriteria 4 sudah siap secara teknis — tinggal disambungkan ke Prometheus server.

**📸 Screenshot bukti serving (untuk submission):**
Ambil screenshot dari output `curl http://localhost:8501/v1/models/marketing-response-model`
yang menunjukkan `"state": "AVAILABLE"`.

Hentikan dengan `Ctrl+C`, lalu `docker compose down`.

### Menguji Prediction Request

Setelah container jalan, buka dan jalankan notebook
**`HarmanM-testing.ipynb`** — notebook ini mengirim data pelanggan contoh
ke endpoint `:predict` model dan menampilkan hasil probabilitas
prediksinya.

---

## Bagian B — Deploy ke Railway (Kriteria 3)

Railway men-deploy langsung dari repo Git (bukan CLI push image), dan
otomatis mendeteksi `Dockerfile` di root project.

### B.1 — Push project ke GitHub

Kalau project ini belum ada di GitHub (boleh **private**, Railway mendukung
akses ke repo private lewat GitHub App authorization):
```bash
cd cspersonality-analysis
git init
git add .
git commit -m "Initial commit: TFX pipeline + serving"
git remote add origin https://github.com/<username-github>/cspersonality-analysis.git
git branch -M main
git push -u origin main
```

> **Catatan:** folder `HarmanM-pipeline/` dan `data/` cukup besar. Kalau
> push lambat/gagal, tambahkan ke `.gitignore` — keduanya tidak dibutuhkan
> Dockerfile untuk deployment (Dockerfile hanya meng-copy `serving_model/`).

### B.2 — Buat Project & Deploy di Railway

1. Login ke [railway.app](https://railway.app/)
2. Klik **New Project** → **Deploy from GitHub repo**
3. Kalau repo tidak muncul di daftar (biasanya karena repo private), klik
   **Configure GitHub App**, lalu di halaman GitHub yang terbuka pilih
   **Only select repositories** → centang repo project kamu → **Save**
4. Kembali ke Railway, pilih repo yang sudah muncul — Railway otomatis
   mendeteksi `Dockerfile` di root dan mulai build
5. Build bisa memakan waktu beberapa menit (base image `tensorflow/serving`
   cukup besar). Pantau progress di tab **Build Logs**

### B.3 — Generate domain publik

1. Klik service yang baru dibuat, buka tab **Settings** → **Networking**
2. Klik **Generate Domain**, lalu pilih port

> ⚠️ **Penting — masalah port yang pernah terjadi:** Railway kadang
> otomatis meng-inject env var `$PORT` dengan nilai yang berbeda dari
> port default TF Serving (8501). Cek dulu **Deploy Logs** (bukan Build
> Logs), cari baris:
> ```
> Exporting HTTP/REST API at:localhost:<PORT_SEBENARNYA> ...
> ```
> Port yang tertulis di situ adalah port **sebenarnya** dipakai container
> (contoh kasus nyata: Railway meng-set `$PORT=7860`, bukan 8501). Pastikan
> port yang kamu pilih saat **Generate Domain** SAMA PERSIS dengan port di
> log ini — kalau beda, akan muncul error `502 Application failed to respond`
> saat diakses. Kalau sudah terlanjur salah, edit lagi di
> **Settings → Networking**, ganti ke port yang benar.

### B.4 — Verifikasi model bisa diakses dari cloud

Setelah domain digenerate, cek:
```bash
curl https://<nama-app-kamu>.up.railway.app/v1/models/marketing-response-model
```
Response yang diharapkan:
```json
{
 "model_version_status": [
  {"version": "...", "state": "AVAILABLE", "status": {"error_code": "OK", "error_message": ""}}
 ]
}
```

**📸 Screenshot ini** adalah bukti utama Kriteria 3 (simpan sebagai `HarmanM-deployment.png`).

---

## Bagian C — Monitoring dengan Prometheus (Kriteria 4)

TF Serving punya endpoint metrics bawaan (`/monitoring/prometheus/metrics`)
yang sudah kita aktifkan lewat `monitoring/prometheus.config` dan
`--monitoring_config_file` di `entrypoint.sh`. Prometheus tinggal
di-arahkan untuk men-scrape endpoint itu.

1. Pastikan `monitoring/prometheus.yml` berisi target URL Railway kamu:
   ```yaml
   global:
     scrape_interval: 15s

   scrape_configs:
     - job_name: 'tf-serving-marketing-model'
       metrics_path: /monitoring/prometheus/metrics
       scheme: https
       static_configs:
         - targets: ['<nama-app-kamu>.up.railway.app']
   ```

2. Jalankan Prometheus (lokal, memantau app di cloud):
   ```bash
   cd cspersonality-analysis/monitoring
   docker compose -f docker-compose.prometheus.yml up
   ```
   (Atau pakai `Dockerfile` di folder `monitoring/` untuk build image Prometheus sendiri: `docker build -t local-prometheus . && docker run -p 9090:9090 local-prometheus`.)

3. Buka dashboard Prometheus di `http://localhost:9090`.

4. Klik **Status** → **Targets**, pastikan target `tf-serving-marketing-model`
   berstatus **UP**.

5. Di tab **Graph**, coba beberapa metrik bawaan TF Serving:
   - `up{job="tf-serving-marketing-model"}` — status scrape (1 = berhasil)
   - `:tensorflow:core:graph_runs` — jumlah eksekusi graph model (naik saat ada request masuk, misal dari `HarmanM-testing.ipynb`)
   - `:tensorflow:serving:request_count` — jumlah request masuk

6. **Screenshot dashboard ini** (target status UP + minimal satu grafik
   metrik) sebagai `HarmanM-monitoring.png` — bukti utama Kriteria 4.

---

## Ringkasan File yang Terlibat

| File | Fungsi |
|---|---|
| `Dockerfile` (root) | Membungkus TF Serving + model jadi image untuk Railway (serving murni, bukan Jupyter) |
| `entrypoint.sh` (root) | Menjalankan TF Serving dengan `$PORT` dinamis + mengaktifkan monitoring |
| `docker-compose.yml` | Test image secara lokal sebelum deploy |
| `.dockerignore` | Mengecualikan file besar/tidak perlu dari image |
| `HarmanM-testing.ipynb` | Notebook uji prediction request ke model yang di-serve |
| `modules/` | Modul pipeline (`transform`, `tuner`, `trainer`) — dicek dengan pylint |
| `monitoring/Dockerfile` | Image Prometheus custom (opsional, alternatif dari docker-compose) |
| `monitoring/prometheus.config` | Mengaktifkan endpoint `/monitoring/prometheus/metrics` di TF Serving (dipakai root `Dockerfile` saat build image, dan juga wajib ada di sini sesuai ketentuan submission) |
| `monitoring/prometheus.yml` | Konfigurasi scrape target Prometheus |
| `monitoring/docker-compose.prometheus.yml` | Menjalankan Prometheus server secara lokal |

## Troubleshooting Umum

- **Build gagal di Railway**: cek tab **Build Logs** untuk pesan error. Penyebab umum: `serving_model/` belum ter-generate (jalankan ulang pipeline TFX dulu sebelum push), atau `entrypoint.sh` tidak ikut ter-push (cek `git status`).
- **Error `502 Application failed to respond`**: hampir selalu karena port mismatch — lihat penjelasan di Bagian B.3 di atas. Cek **Deploy Logs**, samakan port domain dengan port aktual yang dipakai TF Serving.
- **Repo private tidak muncul di daftar Railway**: authorize akses lewat **Configure GitHub App** (lihat Bagian B.2 langkah 3).
- **Request pertama lambat**: kalau pakai plan yang bisa idle/sleep, itu normal (cold start) — tunggu sebentar lalu coba lagi.
- **Prometheus menunjukkan target `down`**: pastikan URL di `monitoring/prometheus.yml` benar, memakai `https`, tanpa trailing slash, dan app Railway-nya memang sedang aktif (bukan di-pause).

## Alternatif Platform yang Dipertimbangkan

Sebelum memutuskan Railway, beberapa platform lain sempat dicoba/disiapkan yaitu:
**Hugging Face Spaces**, **Render**, dan **Heroku**.

`Dockerfile` dan `entrypoint.sh` di root project kompatibel untuk Railway
maupun Render tanpa perubahan, karena keduanya sama-sama menyuntikkan
`$PORT` secara dinamis — beda dengan Hugging Face Spaces yang butuh port
tetap 7860 (sehingga perlu file terpisah di `huggingface-space/`).