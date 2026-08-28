"""
Transform module untuk TFX pipeline - Marketing Campaign Response Prediction.
Melakukan preprocessing fitur numerik (scaling) dan kategorikal (vocab/one-hot).
"""
import tensorflow as tf
import tensorflow_transform as tft

LABEL_KEY = 'Response'

# Fitur numerik yang akan di-scale ke [0,1]
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

# Fitur kategorikal yang akan diubah jadi index vocabulary
CATEGORICAL_FEATURES = ['Education', 'Marital_Status']


def transformed_name(key):
    return key + '_xf'


def fill_in_missing(x):
    """Isi nilai kosong dengan default (0 untuk numerik, '' untuk string)."""
    if isinstance(x, tf.sparse.SparseTensor):
        default_value = '' if x.dtype == tf.string else 0
        x = tf.sparse.to_dense(
            tf.SparseTensor(x.indices, x.values, [x.dense_shape[0], 1]),
            default_value)
    return tf.squeeze(x, axis=1)


def preprocessing_fn(inputs):
    """Fungsi utama preprocessing yang dipanggil TFX Transform component."""
    outputs = {}

    for key in NUMERIC_FEATURES:
        outputs[transformed_name(key)] = tft.scale_to_0_1(
            fill_in_missing(inputs[key]))

    for key in CATEGORICAL_FEATURES:
        outputs[transformed_name(key)] = tft.compute_and_apply_vocabulary(
            fill_in_missing(inputs[key]), top_k=20, num_oov_buckets=1)

    outputs[transformed_name(LABEL_KEY)] = fill_in_missing(inputs[LABEL_KEY])

    return outputs
