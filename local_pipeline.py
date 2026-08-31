"""
local_pipeline.py
Menjalankan seluruh TFX component (ExampleGen s.d. Pusher) untuk pipeline
prediksi Marketing Campaign Response, menggunakan Apache Beam sebagai
Pipeline Orchestrator (menggantikan InteractiveContext yang dipakai pada
tahap eksplorasi sebelumnya).

Jalankan dari terminal:
    python local_pipeline.py
"""
import os
from absl import logging
from tfx.orchestration import metadata, pipeline
from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner

# ----------------------------------------------------------------------
# Konfigurasi pipeline
# ----------------------------------------------------------------------
PIPELINE_NAME = "HarmanM-pipeline"

# Lokasi dataset (folder, bukan file - dibaca oleh CsvExampleGen)
DATA_ROOT = "data"

# Module file untuk masing-masing komponen (Transform, Tuner, Trainer)
TRANSFORM_MODULE_FILE = os.path.join("modules", "marketing_transform.py")
TUNER_MODULE_FILE = os.path.join("modules", "marketing_tuner.py")
TRAINER_MODULE_FILE = os.path.join("modules", "marketing_trainer.py")

# Root folder tempat seluruh artefak/output tiap komponen disimpan.
# Nama folder ini yang harus sesuai kriteria submission: <username>-pipeline
PIPELINE_ROOT = PIPELINE_NAME

# Lokasi ML Metadata (SQLite) - tidak perlu ikut di-commit ke git
METADATA_PATH = os.path.join("metadata", PIPELINE_NAME, "metadata.db")

# Lokasi model final hasil Pusher (dipakai juga oleh Dockerfile TF Serving)
SERVING_MODEL_DIR = os.path.join("serving_model", "marketing-response-model")


def init_local_pipeline(components, pipeline_root: str) -> pipeline.Pipeline:
    """Menyatukan seluruh TFX component menjadi satu pipeline yang
    dijalankan oleh Apache Beam sebagai orchestrator."""
    logging.info(f"Pipeline root set to: {pipeline_root}")

    beam_args = [
    "--direct_running_mode=in_memory",
    "--direct_num_workers=1"]

    return pipeline.Pipeline(
        pipeline_name=PIPELINE_NAME,
        pipeline_root=pipeline_root,
        components=components,
        enable_cache=True,
        metadata_connection_config=metadata.sqlite_metadata_connection_config(
            METADATA_PATH
        ),
        beam_pipeline_args=beam_args,
    )


if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)

    from modules.components import init_components

    components = init_components(
        DATA_ROOT,
        transform_module=TRANSFORM_MODULE_FILE,
        tuner_module=TUNER_MODULE_FILE,
        training_module=TRAINER_MODULE_FILE,
        training_steps=1000,
        eval_steps=200,
        serving_model_dir=SERVING_MODEL_DIR,
    )

    pipeline_obj = init_local_pipeline(components, PIPELINE_ROOT)
    BeamDagRunner().run(pipeline=pipeline_obj)