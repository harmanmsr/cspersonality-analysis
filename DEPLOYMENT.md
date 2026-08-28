# Kriteria 3 & 4: Deployment ke Cloud (Render) & Monitoring (Prometheus)

Panduan ini melanjutkan dari model yang sudah di-push oleh komponen `Pusher`
(folder `serving_model/marketing-response-model/`).

Kita pakai **Render** (bukan Heroku) karena Render masih punya free tier
tanpa kartu kredit untuk web service berbasis Docker — cocok untuk
kebutuhan submission ini.

Prasyarat di komputer kamu:
- [Docker](https://www.docker.com/) sudah terinstal dan berjalan (untuk test lokal)
- Akun [Render](https://render.com/) (gratis, daftar pakai email/GitHub)
- Repo project ini sudah di-push ke **GitHub** (Render deploy dari Git repo, bukan dari CLI push image seperti Heroku)

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
Ambil screenshot dari:
1. Output `curl http://localhost:8501/v1/models/marketing-response-model` yang menunjukkan `"state": "AVAILABLE"`, ATAU
2. Buka `http://localhost:8501/v1/models/marketing-response-model` langsung di browser.

Ini adalah bukti bahwa model benar-benar ter-serve dan bisa diakses.

Hentikan dengan `Ctrl+C`, lalu `docker compose down`.

### Menguji Prediction Request

Setelah container jalan, buka dan jalankan notebook
**`test_serving_prediction.ipynb`** — notebook ini mengirim data pelanggan
contoh ke endpoint `:predict` model dan menampilkan hasil probabilitas
prediksinya. Ini juga bisa dijadikan bukti tambahan (screenshot output
notebook) bahwa serving benar-benar berfungsi untuk inference, bukan cuma
"nyala" saja.

---

## Bagian B — Deploy ke Render (Kriteria 3)

Render men-deploy langsung dari repo Git (bukan push image lewat CLI seperti
Heroku), dan otomatis mendeteksi `Dockerfile` di root project.

### B.1 — Push project ke GitHub

Kalau project ini belum ada di GitHub:
```bash
cd cspersonality-analysis
git init
git add .
git commit -m "Initial commit: TFX pipeline + serving"
# Buat repo baru di GitHub dulu (lewat website), lalu:
git remote add origin https://github.com/<username-github>/cspersonality-analysis.git
git branch -M main
git push -u origin main
```

> **Penting:** folder `pipeline_root/` dan `data/` berisi file yang cukup
> besar. Kalau ukurannya bikin push lambat/gagal, tambahkan ke `.gitignore`
> — file itu tidak dibutuhkan untuk deployment, karena Dockerfile hanya
> meng-copy folder `serving_model/` (lihat isi `Dockerfile`).

### B.2 — Buat Web Service di Render

1. Login ke [dashboard.render.com](https://dashboard.render.com/)
2. Klik **New +** → **Web Service**
3. Hubungkan akun GitHub kamu (kalau belum), lalu pilih repo `cspersonality-analysis`
4. Render otomatis mendeteksi `Dockerfile` di root — biarkan **Language/Runtime** ter-set ke **Docker**
5. Isi konfigurasi:
   - **Name**: `cspersonality-marketing-response` (atau nama lain, ini jadi bagian dari URL)
   - **Region**: pilih yang terdekat (misal Singapore)
   - **Instance Type**: pilih **Free**
6. Klik **Create Web Service**

Render akan build image dari `Dockerfile` dan deploy otomatis. Proses ini
bisa memakan waktu beberapa menit (base image `tensorflow/serving` cukup besar).

> Alternatif: kalau sudah familiar dengan Render Blueprint, cukup jalankan
> **New +** → **Blueprint**, arahkan ke repo ini — Render akan otomatis
> membaca `render.yaml` yang sudah disiapkan di root project.

### B.3 — Verifikasi model bisa diakses dari cloud

Setelah deploy selesai (status **Live** di dashboard Render), URL app kamu
akan berbentuk:
```
https://cspersonality-marketing-response.onrender.com
```

Cek dengan:
```bash
curl https://cspersonality-marketing-response.onrender.com/v1/models/marketing-response-model
```
Response yang diharapkan berisi status `"state": "AVAILABLE"`.

**📸 Screenshot ini** (atau buka URL-nya langsung di browser) adalah bukti
utama untuk Kriteria 3.

> **Catatan cold start:** Render free tier meng-*sleep*-kan service setelah
> ~15 menit idle. Request pertama setelah idle bisa lambat (30-60 detik)
> karena container harus "bangun" dulu — ini normal untuk free tier, bukan
> error.

---

## Bagian C — Monitoring dengan Prometheus (Kriteria 4)

TF Serving punya endpoint metrics bawaan (`/monitoring/prometheus/metrics`)
yang sudah kita aktifkan lewat `config/prometheus.config` dan
`--monitoring_config_file` di `entrypoint.sh`. Prometheus tinggal
di-arahkan untuk men-scrape endpoint itu.

1. Edit `monitoring/prometheus.yml`, isi target dengan URL Render app kamu:
   ```yaml
   scrape_configs:
     - job_name: 'tf-serving-marketing-model'
       metrics_path: /monitoring/prometheus/metrics
       scheme: https
       static_configs:
         - targets: ['cspersonality-marketing-response.onrender.com']
   ```

2. Jalankan Prometheus (lokal, memantau app di cloud):
   ```bash
   cd cspersonality-analysis/monitoring
   docker compose -f docker-compose.prometheus.yml up
   ```

3. Buka dashboard Prometheus di `http://localhost:9090`.

4. Di kolom query (Graph tab), coba beberapa metrik bawaan TF Serving,
   misalnya:
   - `:tensorflow:core:graph_runs` — jumlah eksekusi graph model
   - `:tensorflow:serving:request_count` — jumlah request masuk
   - `up{job="tf-serving-marketing-model"}` — status target (1 = berhasil
     di-scrape, 0 = gagal)

   Ketik salah satu di search box lalu klik **Execute** → tab **Graph**
   untuk melihat visualisasinya.

5. **Screenshot dashboard ini** (target status `up`, dan minimal satu grafik
   metrik) untuk dilampirkan sebagai bukti submission Kriteria 4.

> **Catatan:** karena app Render free tier bisa sleep saat idle, pastikan
> kamu buka/`curl` URL app-nya dulu (supaya "bangun") sebelum menjalankan
> Prometheus, supaya scrape pertama tidak gagal karena cold start.

---

## Ringkasan File yang Terlibat

| File | Fungsi |
|---|---|
| `Dockerfile` | Membungkus TF Serving + model jadi image (serving murni, bukan Jupyter) |
| `entrypoint.sh` | Menjalankan TF Serving dengan `$PORT` dinamis + mengaktifkan monitoring |
| `config/prometheus.config` | Mengaktifkan endpoint `/monitoring/prometheus/metrics` di TF Serving |
| `render.yaml` | Blueprint config untuk deploy otomatis ke Render (opsional, bisa juga setup manual lewat dashboard) |
| `docker-compose.yml` | Test image secara lokal sebelum deploy |
| `.dockerignore` | Mengecualikan file besar/tidak perlu (notebook, data mentah, artefak pipeline) dari image |
| `test_serving_prediction.ipynb` | Notebook terpisah untuk menguji prediction request ke model yang di-serve |
| `monitoring/prometheus.yml` | Konfigurasi scrape target Prometheus |
| `monitoring/docker-compose.prometheus.yml` | Menjalankan Prometheus server secara lokal |

## Troubleshooting Umum

- **Build gagal di Render**: cek tab **Logs** di dashboard Render untuk pesan error spesifik. Penyebab umum: file `entrypoint.sh` tidak ikut ter-push ke GitHub, atau `serving_model/` belum ter-generate (jalankan ulang pipeline TFX dulu sebelum push).
- **App status "Deploy failed" atau crash setelah deploy**: cek apakah `$PORT` benar-benar terbaca — Render *mewajibkan* app listen di port yang diberikan lewat env var `$PORT` (default 10000), bukan port tetap. `entrypoint.sh` kita sudah menangani ini otomatis.
- **Request pertama lambat/timeout**: kemungkinan besar cold start (service baru "bangun" dari sleep) — tunggu sebentar lalu coba lagi, bukan berarti error.
- **Prometheus menunjukkan target `down`**: pastikan URL di `monitoring/prometheus.yml` benar, memakai `https`, tanpa trailing slash di akhir target, dan app Render-nya sedang tidak dalam kondisi sleep.
- **Push ke GitHub lambat/gagal karena ukuran repo**: tambahkan `pipeline_root/` dan `data/` ke `.gitignore` — keduanya tidak dibutuhkan Dockerfile untuk deployment.

## Alternatif: Deploy via Heroku

Kalau suatu saat kamu tetap ingin/perlu pakai Heroku (misalnya untuk
konsistensi dengan latihan kelas yang eksplisit menyebut Heroku), alur
kerjanya mirip tapi pakai Heroku Container Registry alih-alih Git push:
```bash
heroku login
heroku container:login
heroku create <nama-app>
heroku container:push web --app <nama-app>
heroku container:release web --app <nama-app>
```
`Dockerfile` dan `entrypoint.sh` yang sama bisa dipakai untuk kedua platform
tanpa perubahan, karena keduanya sama-sama menyuntikkan `$PORT` secara
dinamis.
