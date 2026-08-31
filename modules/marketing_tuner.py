"""
Tuner module untuk TFX pipeline - Marketing Campaign Response Prediction.
Melakukan hyperparameter tuning otomatis menggunakan KerasTuner (Hyperband)
terhadap arsitektur DNN yang sama dipakai oleh marketing_trainer.py.
"""
from typing import NamedTuple, Dict, Any, Text

import keras_tuner as kt
import tensorflow as tf
import tensorflow_transform as tft
from tfx.components.trainer.fn_args_utils import FnArgs
from tfx.v1.components import TunerFnResult

LABEL_KEY = 'Response'

NUMERIC_FEATURES = [
    'Income', 'Kidhome', 'Teenhome', 'Recency',
    'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts',
    'MntSweetProducts', 'MntGoldProds',
    'NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases',
    'NumStorePurchases', 'NumWebVisitsMonth',
    'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1', 'AcceptedCmp2',
    'Complain', 'Age', 'Customer_Tenure_Days',
    'Total_Spending', 'Total_Children', 'Total_Purchases',
]
CATEGORICAL_FEATURES = ['Education', 'Marital_Status']
CATEGORICAL_VOCAB_SIZE = {'Education': 21, 'Marital_Status': 21}


def transformed_name(key):
    return key + '_xf'


def _input_fn(file_pattern, tf_transform_output, batch_size=64):
    transformed_feature_spec = tf_transform_output.transformed_feature_spec().copy()
    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transformed_feature_spec,
        reader=lambda filenames: tf.data.TFRecordDataset(filenames, compression_type='GZIP'),
        label_key=transformed_name(LABEL_KEY))
    return dataset


def _build_model_for_tuning(hp: kt.HyperParameters) -> tf.keras.Model:
    """Bangun model dengan hyperparameter yang dicari oleh KerasTuner.

    Hyperparameter yang dituning:
    - units_1     : jumlah unit di Dense layer pertama (32, 64, atau 128)
    - units_2     : jumlah unit di Dense layer kedua (16, 32, atau 64)
    - dropout     : dropout rate (0.1 - 0.5)
    - learning_rate: learning rate optimizer Adam (1e-2, 1e-3, 1e-4)
    """
    numeric_inputs = {
        transformed_name(f): tf.keras.Input(shape=(1,), name=transformed_name(f))
        for f in NUMERIC_FEATURES
    }
    categorical_inputs = {
        transformed_name(f): tf.keras.Input(shape=(1,), name=transformed_name(f), dtype=tf.int64)
        for f in CATEGORICAL_FEATURES
    }

    numeric_concat = tf.keras.layers.Concatenate()(list(numeric_inputs.values()))

    cat_embeddings = []
    for f in CATEGORICAL_FEATURES:
        emb = tf.keras.layers.Embedding(
            input_dim=CATEGORICAL_VOCAB_SIZE[f], output_dim=4)(categorical_inputs[transformed_name(f)])
        emb = tf.keras.layers.Flatten()(emb)
        cat_embeddings.append(emb)

    concat = tf.keras.layers.Concatenate()([numeric_concat] + cat_embeddings)

    units_1 = hp.Choice('units_1', values=[32, 64, 128])
    units_2 = hp.Choice('units_2', values=[16, 32, 64])
    dropout = hp.Float('dropout', min_value=0.1, max_value=0.5, step=0.1)
    learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    x = tf.keras.layers.Dense(units_1, activation='relu')(concat)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(units_2, activation='relu')(x)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    inputs = {**numeric_inputs, **categorical_inputs}
    model = tf.keras.Model(inputs=inputs, outputs=output)
    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC(name='auc')])
    return model


def tuner_fn(fn_args: FnArgs) -> TunerFnResult:
    """Dipanggil oleh komponen Tuner TFX untuk menjalankan pencarian hyperparameter."""
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

    train_dataset = _input_fn(fn_args.train_files, tf_transform_output, batch_size=64)
    eval_dataset = _input_fn(fn_args.eval_files, tf_transform_output, batch_size=64)

    tuner = kt.RandomSearch(
        _build_model_for_tuning,
        objective=kt.Objective('val_auc', direction='max'),
        max_trials=10,
        directory=fn_args.working_dir,
        project_name='marketing_response_tuning')

    return TunerFnResult(
        tuner=tuner,
        fit_kwargs={
            'x': train_dataset,
            'validation_data': eval_dataset,
            'steps_per_epoch': fn_args.train_steps,
            'validation_steps': fn_args.eval_steps,
            'epochs': 5,
            'callbacks': [tf.keras.callbacks.EarlyStopping(monitor='val_auc', mode='max', patience=2)],
        })
