# Modules

Folder ini berisi seluruh modul Python yang dipakai untuk membangun machine
learning pipeline TFX (dipanggil oleh komponen `Transform`, `Tuner`, dan
`Trainer` lewat parameter `module_file`).

| File | Dipakai oleh komponen | Fungsi |
|---|---|---|
| `marketing_transform.py` | `Transform` | Preprocessing fitur: scaling numerik ke [0,1], vocabulary untuk fitur kategorikal |
| `marketing_tuner.py` | `Tuner` | Hyperparameter search (KerasTuner RandomSearch) untuk arsitektur model, dioptimalkan terhadap `val_auc` |
| `marketing_trainer.py` | `Trainer` | Definisi arsitektur model DNN dan proses training, menggunakan hyperparameter terbaik dari `Tuner` |

Kode di folder ini sudah dicek dengan **pylint** (skor akhir: lihat
screenshot `HarmanM-pylint`). File `.pylintrc` di folder ini menonaktifkan
rule `no-member` khusus untuk `tf.keras.*` — ini false positive yang umum
terjadi karena TensorFlow memakai lazy-loading module sehingga pylint tidak
bisa menganalisis atributnya secara statis, bukan indikasi masalah kode
sesungguhnya.