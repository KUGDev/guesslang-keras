"""Machine learning model"""

from copy import deepcopy
import logging
from operator import itemgetter
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from typing import List, Tuple, Dict, Any, Callable

import tensorflow as tf
from jinja2.optimizer import optimize
from tensorflow.estimator import ModeKeys, Estimator
from tensorflow.python.ops.lookup_ops import StaticHashTable, KeyValueTensorInitializer
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


def build(model_dir: str, source_files_dir: str, labels: List[str]) -> Estimator:
    """Build a Tensorflow text classifier """
    config = tf.estimator.RunConfig(
        model_dir=model_dir,
        save_checkpoints_steps=Training.CHECKPOINT_STEPS,
    )
    categorical_column = tf.feature_column.categorical_column_with_hash_bucket(
        key='content',
        hash_bucket_size=HyperParameter.VOCABULARY_SIZE,
    )
    dense_column = tf.feature_column.embedding_column(
        categorical_column=categorical_column,
        dimension=HyperParameter.EMBEDDING_SIZE,
    )

    wide_and_deep_input = tf.keras.Input(shape=(10000,), dtype=tf.string, name='content')

    indicator_column = tf.feature_column.indicator_column(categorical_column)
    wide_output = tf.keras.layers.DenseFeatures([indicator_column])({'content': wide_and_deep_input})
    wide_output = tf.keras.layers.Dense(len(labels), activation='softmax')(wide_output)

    wide_model = tf.keras.Model(inputs=wide_and_deep_input, outputs=wide_output)

    deep_output = tf.keras.layers.DenseFeatures([dense_column])({'content': wide_and_deep_input})
    deep_output = tf.keras.layers.Dense(HyperParameter.DNN_HIDDEN_UNITS[0], activation='relu')(deep_output)
    deep_output = tf.keras.layers.Dropout(HyperParameter.DNN_DROPOUT)(deep_output)
    deep_output = tf.keras.layers.Dense(HyperParameter.DNN_HIDDEN_UNITS[1], activation='relu')(deep_output)
    deep_output = tf.keras.layers.Dropout(HyperParameter.DNN_DROPOUT)(deep_output)
    deep_output = tf.keras.layers.Dense(len(labels), activation='softmax')(deep_output)

    deep_model = tf.keras.Model(inputs=wide_and_deep_input, outputs=deep_output)

    wide_deep_model = tf.keras.experimental.WideDeepModel(wide_model, deep_model)

    wide_optimizer = tf.keras.optimizers.Ftrl()
    deep_optimizer = tf.keras.optimizers.Adagrad()

    wide_deep_model.compile(
        optimizer=[wide_optimizer, deep_optimizer],
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    label_lookup = StaticHashTable(
        KeyValueTensorInitializer(
            keys=tf.constant(labels),
            values=tf.constant(list(range(len(labels))), dtype=tf.int64),
        ),
        default_value=-1,
    )

    train_ds = _build_input_fn_new(source_files_dir, label_lookup, ModeKeys.TRAIN)()
    eval_ds = _build_input_fn_new(source_files_dir, label_lookup, ModeKeys.EVAL)()

    wide_deep_model.fit(train_ds, epochs=10)
    metrics = wide_deep_model.evaluate(eval_ds, return_dict=True)

    return tf.estimator.DNNLinearCombinedClassifier(
        linear_feature_columns=[categorical_column],
        dnn_feature_columns=[dense_column],
        dnn_hidden_units=HyperParameter.DNN_HIDDEN_UNITS,
        dnn_dropout=HyperParameter.DNN_DROPOUT,
        label_vocabulary=labels,
        n_classes=len(labels),
        config=config,
    )


def train(estimator: Estimator, data_root_dir: str, max_steps: int) -> Any:
    """Train a Tensorflow estimator"""

    train_spec = tf.estimator.TrainSpec(
        input_fn=_build_input_fn_old(data_root_dir, ModeKeys.TRAIN),
        max_steps=max_steps,
    )

    if max_steps > Training.LONG_TRAINING_STEPS:
        throttle_secs = Training.LONG_DELAY
    else:
        throttle_secs = Training.SHORT_DELAY

    eval_spec = tf.estimator.EvalSpec(
        input_fn=_build_input_fn_old(data_root_dir, ModeKeys.EVAL),
        start_delay_secs=Training.SHORT_DELAY,
        throttle_secs=throttle_secs,
    )

    LOGGER.debug('Train the model')
    results = tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
    training_metrics = results[0]
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
    input_function = _build_input_fn_old(data_root_dir, ModeKeys.PREDICT)
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


def _build_input_fn_old(
    data_root_dir: str,
    mode: ModeKeys,
) -> Callable[[], tf.data.Dataset]:
    """Generate an input fonction for a Tensorflow model"""
    pattern = str(Path(data_root_dir).joinpath(DATASET[mode], '*'))

    def input_function() -> tf.data.Dataset:
        dataset = tf.data.Dataset
        dataset = dataset.list_files(pattern, shuffle=True).map(_read_file)

        if mode == ModeKeys.PREDICT:
            return dataset.batch(1)

        if mode == ModeKeys.TRAIN:
            dataset = dataset.shuffle(Training.SHUFFLE_BUFFER).repeat()

        return dataset.map(_preprocess).batch(HyperParameter.BATCH_SIZE)

    return input_function


def _build_input_fn_new(
    data_root_dir: str,
    label_lookup: StaticHashTable,
    mode: ModeKeys,
) -> Callable[[], tf.data.Dataset]:
    """Generate an input fonction for a Tensorflow model"""
    pattern = str(Path(data_root_dir).joinpath(DATASET[mode], '*'))

    def input_function() -> tf.data.Dataset:
        dataset = tf.data.Dataset
        dataset = dataset.list_files(pattern, shuffle=True).map(_read_file)

        if mode == ModeKeys.PREDICT:
            return dataset.batch(1)

        if mode == ModeKeys.TRAIN:
            dataset = dataset.shuffle(Training.SHUFFLE_BUFFER) # .repeat()

        def _perform_label_lookup(data: tf.Tensor, label: tf.Tensor):
            return data, label_lookup.lookup(label)

        return dataset.map(_preprocess).map(_perform_label_lookup).batch(HyperParameter.BATCH_SIZE)

    return input_function


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
    """Feature engineering"""
    padding = tf.constant(['']*HyperParameter.NB_TOKENS)
    data = tf.strings.bytes_split(data)
    data = tf.strings.ngrams(data, HyperParameter.N_GRAM)
    data = tf.concat((data, padding), axis=0)
    data = data[:HyperParameter.NB_TOKENS]
    return data
