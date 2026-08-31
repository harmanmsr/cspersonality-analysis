"""
components.py
Menyatukan seluruh TFX component (ExampleGen s.d. Pusher) menjadi satu
fungsi init_components() yang dipanggil oleh local_pipeline.py.

Urutan komponen:
CsvExampleGen -> StatisticsGen -> SchemaGen -> ExampleValidator -> Transform
             -> Tuner -> Trainer -> Resolver -> Evaluator -> Pusher
"""
from typing import Text

import tensorflow_model_analysis as tfma
from tfx.components import (
    CsvExampleGen,
    StatisticsGen,
    SchemaGen,
    ExampleValidator,
    Transform,
    # Tuner,
    Trainer,
    Evaluator,
    Pusher,
)
from tfx.dsl.components.common.resolver import Resolver
from tfx.dsl.input_resolution.strategies.latest_blessed_model_strategy import (
    LatestBlessedModelStrategy,
)
from tfx.proto import example_gen_pb2, trainer_pb2, pusher_pb2
from tfx.types import Channel
from tfx.types.standard_artifacts import Model, ModelBlessing


def init_components(
    data_dir: Text,
    transform_module: Text,
    # tuner_module: Text,
    training_module: Text,
    training_steps: int,
    eval_steps: int,
    serving_model_dir: Text,
):
    """Membuat dan menyatukan seluruh TFX component.

    Args:
        data_dir: path folder yang berisi dataset CSV (dibaca oleh CsvExampleGen).
        transform_module: path module_file untuk komponen Transform.
        tuner_module: path module_file untuk komponen Tuner.
        training_module: path module_file untuk komponen Trainer.
        training_steps: jumlah step training untuk Trainer & Tuner.
        eval_steps: jumlah step evaluasi untuk Trainer & Tuner.
        serving_model_dir: path tujuan Pusher menyimpan model siap serving.

    Returns:
        Tuple berisi seluruh komponen TFX, siap dimasukkan ke pipeline.Pipeline().
    """

    # 1. ExampleGen — split dataset menjadi train (8) : eval (2)
    output_config = example_gen_pb2.Output(
        split_config=example_gen_pb2.SplitConfig(
            splits=[
                example_gen_pb2.SplitConfig.Split(name="train", hash_buckets=8),
                example_gen_pb2.SplitConfig.Split(name="eval", hash_buckets=2),
            ]
        )
    )
    example_gen = CsvExampleGen(input_base=data_dir, output_config=output_config)

    # 2. StatisticsGen — hitung statistik deskriptif dataset
    statistics_gen = StatisticsGen(examples=example_gen.outputs["examples"])

    # 3. SchemaGen — infer skema data dari statistik
    schema_gen = SchemaGen(
        statistics=statistics_gen.outputs["statistics"], infer_feature_shape=True
    )

    # 4. ExampleValidator — cek anomali data terhadap skema
    example_validator = ExampleValidator(
        statistics=statistics_gen.outputs["statistics"],
        schema=schema_gen.outputs["schema"],
    )

    # 5. Transform — preprocessing fitur (scaling numerik, vocab kategorikal)
    transform = Transform(
        examples=example_gen.outputs["examples"],
        schema=schema_gen.outputs["schema"],
        module_file=transform_module,
    )

    # 6. Tuner — hyperparameter search (KerasTuner RandomSearch)
    # tuner = Tuner(
    #     module_file=tuner_module,
    #     examples=transform.outputs["transformed_examples"],
    #     transform_graph=transform.outputs["transform_graph"],
    #     schema=schema_gen.outputs["schema"],
    #     train_args=trainer_pb2.TrainArgs(splits=["train"], num_steps=training_steps),
    #     eval_args=trainer_pb2.EvalArgs(splits=["eval"], num_steps=eval_steps),
    # )

    # 7. Trainer — training model final memakai hyperparameter terbaik dari Tuner
    trainer = Trainer(
        module_file=training_module,
        examples=transform.outputs["transformed_examples"],
        transform_graph=transform.outputs["transform_graph"],
        schema=schema_gen.outputs["schema"],
        # hyperparameters=tuner.outputs["best_hyperparameters"],
        train_args=trainer_pb2.TrainArgs(splits=["train"], num_steps=training_steps),
        eval_args=trainer_pb2.EvalArgs(splits=["eval"], num_steps=eval_steps),
    )

    # 8. Resolver — ambil model blessed terakhir sebagai baseline pembanding Evaluator
    model_resolver = Resolver(
        strategy_class=LatestBlessedModelStrategy,
        model=Channel(type=Model),
        model_blessing=Channel(type=ModelBlessing),
    ).with_id("Latest_blessed_model_resolver")

    # 9. Evaluator — validasi performa model (AUC >= 0.80) sebelum di-push
    eval_config = tfma.EvalConfig(
        model_specs=[tfma.ModelSpec(label_key="Response")],
        slicing_specs=[tfma.SlicingSpec()],
        metrics_specs=[
            tfma.MetricsSpec(
                metrics=[
                    tfma.MetricConfig(class_name="ExampleCount"),
                    tfma.MetricConfig(class_name="BinaryAccuracy"),
                    tfma.MetricConfig(
                        class_name="AUC",
                        threshold=tfma.MetricThreshold(
                            value_threshold=tfma.GenericValueThreshold(
                                lower_bound={"value": 0.80}
                            ),
                            change_threshold=tfma.GenericChangeThreshold(
                                direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                                absolute={"value": -1e-3},
                            ),
                        ),
                    ),
                ]
            )
        ],
    )
    evaluator = Evaluator(
        examples=example_gen.outputs["examples"],
        model=trainer.outputs["model"],
        baseline_model=model_resolver.outputs["model"],
        eval_config=eval_config,
    )

    # 10. Pusher — simpan model ke direktori serving jika dinyatakan blessed
    pusher = Pusher(
        model=trainer.outputs["model"],
        model_blessing=evaluator.outputs["blessing"],
        push_destination=pusher_pb2.PushDestination(
            filesystem=pusher_pb2.PushDestination.Filesystem(
                base_directory=serving_model_dir
            )
        ),
    )

    components = (
        example_gen,
        statistics_gen,
        schema_gen,
        example_validator,
        transform,
        # tuner,
        trainer,
        model_resolver,
        evaluator,
        pusher,
    )

    return components