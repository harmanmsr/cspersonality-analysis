# Kriteria 3 & 4: Deployment ke Cloud (Hugging Face Spaces) & Monitoring (Prometheus)

Panduan ini melanjutkan dari model yang sudah di-push oleh komponen `Pusher`
(folder `serving_model/marketing-response-model/`).

Kita pakai **Hugging Face Spaces** (bukan Heroku/Render) karena benar-benar
gratis dan **tidak memerlukan kartu kredit sama sekali** untuk deploy Docker
container — cocok untuk kebutuhan submission ini.

Prasyarat di komputer kamu:
- [Docker](https://www.docker.com/) sudah terinstal dan berjalan (untuk test lokal)
- Akun [Hugging Face](https://huggingface.co/) (gratis, tanpa kartu kredit)
- `git` terinstal (Spaces adalah repo Git tersendiri)

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

## Bagian B — Deploy ke Hugging Face Spaces (Kriteria 3)

Render sekarang meminta info kartu kredit untuk sign up, jadi kita pakai
**Hugging Face Spaces** — mendukung custom Docker container, gratis, dan
**tidak butuh kartu kredit sama sekali**.

Spaces adalah repo Git tersendiri (terpisah dari repo GitHub project utama),
jadi file-file untuk keperluan ini sudah disiapkan di subfolder khusus:
**`huggingface-space/`** — isinya `Dockerfile`, `entrypoint.sh`, `config/`,
`serving_model/`, dan `README.md` (dengan metadata khusus HF Spaces).

> **Kenapa folder terpisah?** HF Spaces mewajibkan container listen di
> **port 7860 tetap** (bukan `$PORT` dinamis seperti Render/Heroku), dan
> butuh `README.md` dengan YAML frontmatter khusus di baris paling atas
> untuk konfigurasi Space. Supaya tidak bentrok dengan `README.md` &
> `Dockerfile` project utama, dipisah ke folder sendiri.

### B.1 — Buat Space baru

1. Daftar/login ke [huggingface.co](https://huggingface.co/) (gratis, tanpa kartu kredit)
2. Klik profil kamu → **New Space**
3. Isi:
   - **Space name**: misal `cspersonality-model-api`
   - **License**: bebas, misal `mit`
   - **Space SDK**: pilih **Docker** → template **Blank**
   - **Space hardware**: **CPU basic — Free**
   - **Visibility**: Public atau Private (keduanya gratis)
4. Klik **Create Space**

### B.2 — Push isi folder `huggingface-space/` ke Space

Setiap Space punya repo Git sendiri. Clone repo kosong itu, lalu isi dengan
file dari folder `huggingface-space/`:

```bash
git clone https://huggingface.co/spaces/<username-hf>/cspersonality-model-api
cd cspersonality-model-api

# Copy semua isi folder huggingface-space/ dari project kamu ke sini
# (Dockerfile, entrypoint.sh, config/, serving_model/, README.md)

git add .
git commit -m "Deploy TF Serving model"
git push
```

> Kalau `git push` minta login, gunakan **access token** dari
> [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
> sebagai password (bukan password akun biasa).

Setelah push, buka tab **Space** kamu di huggingface.co — Space akan
otomatis build dari `Dockerfile` (proses build bisa beberapa menit karena
base image `tensorflow/serving` cukup besar). Progress build bisa dilihat
di tab **Logs**.

### B.3 — Verifikasi model bisa diakses dari cloud

Setelah build selesai dan status Space **Running**, URL model kamu:
```
https://<username-hf>-cspersonality-model-api.hf.space
```

Cek dengan:
```bash
curl https://<username-hf>-cspersonality-model-api.hf.space/v1/models/marketing-response-model
```
Response yang diharapkan berisi status `"state": "AVAILABLE"`.

**📸 Screenshot ini** (atau buka URL-nya langsung di browser) adalah bukti
utama untuk Kriteria 3.

> **Catatan sleep**: Space CPU gratis akan sleep setelah idle dalam waktu
> tertentu (biasanya ~48 jam tanpa aktivitas).
> Buka URL-nya dulu untuk "membangunkan" Space sebelum verifikasi/testing.

---

## Bagian C — Monitoring dengan Prometheus (Kriteria 4)

TF Serving punya endpoint metrics bawaan (`/monitoring/prometheus/metrics`)
yang sudah kita aktifkan lewat `config/prometheus.config` dan
`--monitoring_config_file` di `entrypoint.sh`. Prometheus tinggal
di-arahkan untuk men-scrape endpoint itu.

1. Edit `monitoring/prometheus.yml`, isi target dengan URL Hugging Face Space kamu:
   ```yaml
   scrape_configs:
     - job_name: 'tf-serving-marketing-model'
       metrics_path: /monitoring/prometheus/metrics
       scheme: https
       static_configs:
         - targets: ['<username-hf>-cspersonality-model-api.hf.space']
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

> **Catatan:** karena Space CPU gratis bisa sleep saat idle, pastikan
> kamu buka/`curl` URL Space-nya dulu (supaya "bangun") sebelum menjalankan
> Prometheus, supaya scrape pertama tidak gagal karena cold start.

---

## Ringkasan File yang Terlibat

| File | Fungsi |
|---|---|
| `Dockerfile` | Membungkus TF Serving + model jadi image (serving murni, bukan Jupyter) |
| `entrypoint.sh` | Menjalankan TF Serving dengan `$PORT` dinamis + mengaktifkan monitoring |
| `config/prometheus.config` | Mengaktifkan endpoint `/monitoring/prometheus/metrics` di TF Serving |
| `huggingface-space/` | Folder terpisah berisi `Dockerfile`, `entrypoint.sh`, `config/`, `serving_model/`, dan `README.md` (dengan metadata Space) — isinya di-push sebagai repo Git Hugging Face Space tersendiri |
| `docker-compose.yml` | Test image secara lokal sebelum deploy |
| `.dockerignore` | Mengecualikan file besar/tidak perlu (notebook, data mentah, artefak pipeline) dari image |
| `test_serving_prediction.ipynb` | Notebook terpisah untuk menguji prediction request ke model yang di-serve |
| `monitoring/prometheus.yml` | Konfigurasi scrape target Prometheus |
| `monitoring/docker-compose.prometheus.yml` | Menjalankan Prometheus server secara lokal |

## Troubleshooting Umum

- **Build gagal di HF Spaces**: cek tab **Logs** di halaman Space untuk pesan error spesifik. Penyebab umum: `serving_model/` belum ikut ter-push (jalankan ulang pipeline TFX dulu, lalu copy ulang ke `huggingface-space/serving_model/` sebelum push), atau README.md kehilangan YAML frontmatter di baris paling atas.
- **Space status "Runtime error" atau crash**: cek apakah container benar-benar listen di port **7860** — HF Spaces tidak menerima port lain. `entrypoint.sh` di folder `huggingface-space/` sudah di-hardcode ke 7860.
- **Request pertama lambat/timeout**: kemungkinan besar cold start (Space baru "bangun" dari sleep) — tunggu sebentar lalu coba lagi, bukan berarti error.
- **Prometheus menunjukkan target `down`**: pastikan URL di `monitoring/prometheus.yml` benar (format `<username>-<space-name>.hf.space`), memakai `https`, tanpa trailing slash, dan Space-nya sedang tidak dalam kondisi sleep.
- **Push ke GitHub lambat/gagal karena ukuran repo**: tambahkan `pipeline_root/` dan `data/` ke `.gitignore` — keduanya tidak dibutuhkan Dockerfile untuk deployment.

## Alternatif: Deploy via Render atau Heroku (kalau punya kartu kredit)

Kalau suatu saat kamu ingin/perlu pakai platform lain (misalnya untuk
konsistensi dengan latihan kelas yang menyebut Heroku, atau ingin coba
Render), `Dockerfile` dan `entrypoint.sh` di **root project** (bukan yang
di folder `huggingface-space/`) sudah disiapkan untuk keduanya — karena
sama-sama menyuntikkan `$PORT` secara dinamis (beda dengan HF Spaces yang
fixed di port 7860).

**Render** (perlu kartu kredit untuk sign up per kebijakan terbaru):
1. Push project ke GitHub
2. Buat **Web Service** baru di [dashboard.render.com](https://dashboard.render.com/), hubungkan repo, pilih runtime **Docker**, plan **Free**
3. Render otomatis baca `render.yaml` di root project kalau deploy lewat **Blueprint**

**Heroku** (berbayar, mulai ~$5/bulan Eco Dyno):
```bash
heroku login
heroku container:login
heroku create <nama-app>
heroku container:push web --app <nama-app>
heroku container:release web --app <nama-app>
```