"""Machine learning model"""

from copy import deepcopy
import logging
from operator import itemgetter
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from typing import List, Tuple, Dict, Any, Callable

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import TextVectorization
from tensorflow.lookup import StaticHashTable, KeyValueTensorInitializer
from tensorflow.estimator import ModeKeys, Estimator
from tensorflow.python.training.tracking.tracking import AutoTrackable


LOGGER = logging.getLogger(__name__)

DATASET = {
    ModeKeys.TRAIN: 'train',
    ModeKeys.EVAL: 'valid',
    ModeKeys.PREDICT: 'test',
}


class HyperParameter:
    """Model hyper parameters"""
    BATCH_SIZE = 100
    NB_TOKENS = 10000
    VOCABULARY_SIZE = 5000
    EMBEDDING_SIZE = max(10, int(VOCABULARY_SIZE**0.5))
    DNN_HIDDEN_UNITS = [512, 32]
    DNN_DROPOUT = 0.5
    N_GRAM = 2
    LEARNING_RATE = 0.001
    STEPS_PER_EPOCH = 100


class Training:
    """Model training parameters"""
    SHUFFLE_BUFFER = HyperParameter.BATCH_SIZE * 10
    CHECKPOINT_STEPS = 1000
    LONG_TRAINING_STEPS = 10 * CHECKPOINT_STEPS
    SHORT_DELAY = 60
    LONG_DELAY = 5 * SHORT_DELAY


def load(saved_model_dir: str) -> AutoTrackable:
    """Load a Tensorflow saved model"""
    return tf.saved_model.load(saved_model_dir)


def build(labels: List[str]) -> Tuple[Model, TextVectorization, TextVectorization, StaticHashTable]:
    """Build a Tensorflow text classifier """
    # config = tf.estimator.RunConfig(
    #     model_dir=model_dir,
    #     save_checkpoints_steps=Training.CHECKPOINT_STEPS,
    # )
    # categorical_column = tf.feature_column.categorical_column_with_hash_bucket(
    #     key='content',
    #     hash_bucket_size=HyperParameter.VOCABULARY_SIZE,
    # )
    # dense_column = tf.feature_column.embedding_column(
    #     categorical_column=categorical_column,
    #     dimension=HyperParameter.EMBEDDING_SIZE,
    # )
    #
    # return tf.estimator.DNNLinearCombinedClassifier(
    #     linear_feature_columns=[categorical_column],
    #     dnn_feature_columns=[dense_column],
    #     dnn_hidden_units=HyperParameter.DNN_HIDDEN_UNITS,
    #     dnn_dropout=HyperParameter.DNN_DROPOUT,
    #     label_vocabulary=labels,
    #     n_classes=len(labels),
    #     config=config,
    # )

    # Keras approach

    input_layer = tf.keras.Input(shape=(1,), dtype=tf.string, name="content")

    # === Wide part (one-hot hashed) ===
    vectorizer_wide = tf.keras.layers.TextVectorization(
        max_tokens=HyperParameter.VOCABULARY_SIZE,
        output_mode="one_hot"
    )
    wide_x = vectorizer_wide(input_layer)

    # === Deep part (embedding + DNN) ===
    vectorizer_deep = tf.keras.layers.TextVectorization(
        max_tokens=HyperParameter.VOCABULARY_SIZE,
        output_mode="int",
        output_sequence_length=HyperParameter.NB_TOKENS,
        ngrams=HyperParameter.N_GRAM,
    )
    deep_x = vectorizer_deep(input_layer)
    deep_x = tf.keras.layers.Embedding(input_dim=HyperParameter.VOCABULARY_SIZE,
                                       output_dim=HyperParameter.EMBEDDING_SIZE)(deep_x)
    deep_x = tf.keras.layers.GlobalAveragePooling1D()(deep_x) # Model could be improved here?
    deep_x = tf.keras.layers.Dense(HyperParameter.DNN_HIDDEN_UNITS[0], activation="relu")(deep_x)
    deep_x = tf.keras.layers.Dropout(HyperParameter.DNN_DROPOUT)(deep_x)
    deep_x = tf.keras.layers.Dense(HyperParameter.DNN_HIDDEN_UNITS[1], activation="relu")(deep_x)

    x = tf.keras.layers.concatenate([wide_x, deep_x])
    output = tf.keras.layers.Dense(len(labels), activation="softmax")(x)

    built_model = tf.keras.Model(inputs=input_layer, outputs=output)

    label_lookup = StaticHashTable(
        KeyValueTensorInitializer(
            keys=tf.constant(labels),
            values=tf.constant(list(range(len(labels))), dtype=tf.int64),
        ),
        default_value=-1,
    )
    return built_model, vectorizer_wide, vectorizer_deep, label_lookup


def train(
    built_model: Model,
    vectorizer_wide: TextVectorization,
    vectorizer_deep: TextVectorization,
    label_lookup: StaticHashTable,
    data_root_dir: str,
    max_steps: int,
) -> Model:
    """Train a Keras model"""

    LOGGER.debug('Train the model')

    # === Adapt vectorizers ===

    pattern = str(Path(data_root_dir).joinpath(DATASET[ModeKeys.TRAIN], '*'))
    adapt_dataset = (tf.data.Dataset
                     .list_files(pattern)
                     .map(_read_file)
                     .cache()
                     .map(lambda content, _: content)
                     .batch(32))
    vectorizer_wide.adapt(adapt_dataset)
    vectorizer_deep.adapt(adapt_dataset)

    # === Compile the model ===

    train_dataset = _build_input_fn(data_root_dir, label_lookup, ModeKeys.TRAIN)

    built_model.compile(
        optimizer=tf.keras.optimizers.Adagrad(HyperParameter.LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # === Train the model ===

    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath='checkpoints/model.{epoch:02d}.h5',
        save_weights_only=False,
        save_best_only=False,
        save_freq='epoch'
    )

    built_model.fit(
        train_dataset,
        epochs=max_steps,
        steps_per_epoch=HyperParameter.STEPS_PER_EPOCH,
        callbacks=[checkpoint_cb]
    )

    return built_model


def evaluate(trained_model: Model, label_lookup: StaticHashTable, data_root_dir: str,) -> dict:
    """Evaluate the trained Keras model"""

    eval_dataset = _build_input_fn(data_root_dir, label_lookup, ModeKeys.EVAL)
    training_metrics = trained_model.evaluate(eval_dataset, return_dict=True)

    return training_metrics


def save(estimator: Estimator, saved_model_dir: str) -> None:
    """Save a Tensorflow estimator"""
    with TemporaryDirectory() as temporary_model_base_dir:
        export_dir = estimator.export_saved_model(
            temporary_model_base_dir, _serving_input_receiver_fn
        )

        Path(saved_model_dir).mkdir(exist_ok=True)
        export_path = Path(export_dir.decode()).absolute()
        for path in export_path.glob('*'):
            shutil.move(str(path), saved_model_dir)


def test(
    saved_model: AutoTrackable,
    data_root_dir: str,
    mapping: Dict[str, str],
) -> Dict[str, Dict[str, int]]:
    """Test a Tensorflow saved model"""
    values = {language: 0 for language in mapping.values()}
    matches = {language: deepcopy(values) for language in values}

    LOGGER.debug('Test the model')
    input_function = _build_input_fn(data_root_dir, ModeKeys.PREDICT)
    for test_item in input_function():
        content = test_item[0]
        label = test_item[1].numpy()[0].decode()

        result = saved_model.signatures['predict'](content)
        predicted = result['classes'].numpy()[0][0].decode()

        label_language = mapping[label]
        predicted_language = mapping[predicted]
        matches[label_language][predicted_language] += 1

    return matches


def predict(
    saved_model: AutoTrackable,
    mapping: Dict[str, str],
    text: str
) -> List[Tuple[str, float]]:
    """Infer a Tensorflow saved model"""
    content_tensor = tf.constant([text])
    predicted = saved_model.signatures['serving_default'](content_tensor)

    numpy_floats = predicted['scores'][0].numpy()
    extensions = predicted['classes'][0].numpy()

    probability_values = (float(value) for value in numpy_floats)
    languages = (mapping[ext.decode()] for ext in extensions)

    unsorted_scores = zip(languages, probability_values)
    scores = sorted(unsorted_scores, key=itemgetter(1), reverse=True)
    return scores


def _build_input_fn(
    data_root_dir: str,
    label_lookup: StaticHashTable,
    mode: ModeKeys,
) -> tf.data.Dataset:
    """Generate an input dataset for a Keras model"""
    pattern = str(Path(data_root_dir).joinpath(DATASET[mode], '*'))

    raw_dataset = (tf.data.Dataset.list_files(pattern)
                   .map(_read_file, num_parallel_calls=tf.data.AUTOTUNE)
                   .shuffle(Training.SHUFFLE_BUFFER)
                   .cache()
                   .repeat())

    if mode == ModeKeys.PREDICT:
        return raw_dataset.batch(1)

    if mode == ModeKeys.TRAIN:
        raw_dataset = raw_dataset.shuffle(Training.SHUFFLE_BUFFER).repeat()

    def encode(data: tf.Tensor, label: tf.Tensor):
        return {"content": data}, label_lookup.lookup(label)

    return raw_dataset.map(encode).batch(HyperParameter.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)



def _serving_input_receiver_fn() -> tf.estimator.export.ServingInputReceiver:
    """Function to serve model for predictions."""

    content = tf.compat.v1.placeholder(tf.string, [None])
    receiver_tensors = {'content': content}
    features = {'content': tf.map_fn(_preprocess_text, content)}

    return tf.estimator.export.ServingInputReceiver(
        receiver_tensors=receiver_tensors,
        features=features,
    )


def _read_file(filename: str) -> Tuple[tf.Tensor, tf.Tensor]:
    """Read a source file, return the content and the extension"""
    data = tf.io.read_file(filename)
    label = tf.strings.split([filename], '.').values[-1]
    return data, label


def _preprocess(
    data: tf.Tensor,
    label: tf.Tensor,
) -> Tuple[Dict[str, tf.Tensor], tf.Tensor]:
    """Process input data as part of a workflow"""
    data = _preprocess_text(data)
    return {'content': data}, label


def _preprocess_text(data: tf.Tensor) -> tf.Tensor:
    """
    Feature engineering.
    If NB_TOKENS=10000, N_GRAM=2 =>
    Produces the Tensor with the next representation:
      origin ~~ [b"H", b"e", b"l", b"l", b"o", b" ", b"w", b"o", b"r", b"l", b"d"]
      data ~~ [b"H e", b"e l", b"l l", b"l o", b"o  ", b"  w", b"w o", b"o r", b"r l", b"l d", b"d  ", b"   ", ..., b"   "]
    NB_TOKENS=10000 => 10000 tokens (n-grams) will be produced, or the data will be trimmed to 10000 tokens
    if the content length produces too many n-grams
    """
    padding = tf.constant(['']*HyperParameter.NB_TOKENS)
    data = tf.strings.bytes_split(data)
    data = tf.strings.ngrams(data, HyperParameter.N_GRAM)
    data = tf.concat((data, padding), axis=0)
    data = data[:HyperParameter.NB_TOKENS]
    return data
