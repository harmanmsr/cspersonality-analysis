"""
Trainer module untuk TFX pipeline - Marketing Campaign Response Prediction.
Membangun model Keras sederhana (DNN binary classifier) menggunakan fitur
hasil Transform component.
"""
import tensorflow as tf
import tensorflow_transform as tft
from tensorflow_transform.tf_metadata import schema_utils
from tfx.components.trainer.fn_args_utils import FnArgs
from tfx_bsl.public import tfxio
import keras_tuner as kt

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
CATEGORICAL_VOCAB_SIZE = {'Education': 21, 'Marital_Status': 21}  # top_k + oov buckets


def transformed_name(key):
    return key + '_xf'


def _input_fn(file_pattern, tf_transform_output, batch_size=64):
    transformed_feature_spec = (
        tf_transform_output.transformed_feature_spec().copy())

    dataset = tfxio.TensorFlowDatasetOptions
    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transformed_feature_spec,
        reader=lambda filenames: tf.data.TFRecordDataset(filenames, compression_type='GZIP'),
        label_key=transformed_name(LABEL_KEY))
    return dataset


def _build_keras_model(hparams: kt.HyperParameters = None):
    """Bangun model DNN. Jika `hparams` diberikan (hasil komponen Tuner),
    arsitektur memakai unit/dropout/learning_rate terbaik hasil pencarian.
    Jika tidak, jatuh ke nilai default sebagai fallback."""
    units_1 = hparams.get('units_1') if hparams else 64
    units_2 = hparams.get('units_2') if hparams else 32
    dropout = hparams.get('dropout') if hparams else 0.3
    learning_rate = hparams.get('learning_rate') if hparams else 1e-3

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
            input_dim=CATEGORICAL_VOCAB_SIZE[f], output_dim=4,
            name=f'embedding_{f}')(categorical_inputs[transformed_name(f)])
        emb = tf.keras.layers.Flatten(name=f'flatten_{f}')(emb)
        cat_embeddings.append(emb)

    concat = tf.keras.layers.Concatenate()([numeric_concat] + cat_embeddings)
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


def _get_serve_tf_examples_fn(model, tf_transform_output):
    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples):
        feature_spec = tf_transform_output.raw_feature_spec()
        feature_spec.pop(LABEL_KEY, None)
        parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)
        transformed_features = model.tft_layer(parsed_features)
        return model(transformed_features)

    return serve_tf_examples_fn


def run_fn(fn_args: FnArgs):
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

    train_dataset = _input_fn(fn_args.train_files, tf_transform_output, batch_size=64)
    eval_dataset = _input_fn(fn_args.eval_files, tf_transform_output, batch_size=64)

    hparams = None
    if fn_args.hyperparameters:
        hparams = kt.HyperParameters.from_config(fn_args.hyperparameters)

    model = _build_keras_model(hparams)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_auc', mode='max', patience=3, restore_best_weights=True)
    ]

    model.fit(
        train_dataset,
        steps_per_epoch=fn_args.train_steps,
        validation_data=eval_dataset,
        validation_steps=fn_args.eval_steps,
        epochs=10,
        callbacks=callbacks)

    signatures = {
        'serving_default': _get_serve_tf_examples_fn(
            model, tf_transform_output).get_concrete_function(
                tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')),
    }
    tf.saved_model.save(model, fn_args.serving_model_dir, signatures=signatures)