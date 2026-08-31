import os
import sys
from typing import Text
 
from absl import logging
from tfx.orchestration import metadata, pipeline
from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner
 
PIPELINE_NAME = "HarmanM-pipeline"
 
# pipeline inputs
DATA_ROOT = "data"
TRANSFORM_MODULE_FILE = os.path.join("modules", "marketing_transform.py")
# TUNER_MODULE_FILE = os.path.join("modules", "marketing_tuner.py")
TRAINER_MODULE_FILE = os.path.join("modules", "marketing_trainer.py")
# requirement_file = os.path.join(root, "requirements.txt")
 
# pipeline outputs
OUTPUT_BASE = "output"
serving_model_dir = os.path.join(OUTPUT_BASE, 'marketing-response-model')
pipeline_root = os.path.join(OUTPUT_BASE, PIPELINE_NAME)
metadata_path = os.path.join(pipeline_root, "metadata.sqlite")


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
            metadata_path
        ),
        beam_pipeline_args=beam_args,
    )


if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)

    from modules.components import init_components

    components = init_components(
        DATA_ROOT,
        transform_module=TRANSFORM_MODULE_FILE,
        # tuner_module=TUNER_MODULE_FILE,
        training_module=TRAINER_MODULE_FILE,
        training_steps=1000,
        eval_steps=200,
        serving_model_dir=serving_model_dir,
    )

    pipeline_obj = init_local_pipeline(components, pipeline_root)
    BeamDagRunner().run(pipeline=pipeline_obj)