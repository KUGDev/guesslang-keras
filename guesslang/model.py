"""Machine learning model"""
import json
from copy import deepcopy
import logging
from operator import itemgetter
from pathlib import Path
from typing import List, Tuple, Dict, Any, Callable

import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.python.ops.lookup_ops import StaticHashTable, KeyValueTensorInitializer

class ModeKeys:
    TRAIN = 'train'
    EVAL = 'valid'
    PREDICT = 'test'


LOGGER = logging.getLogger(__name__)

DATASET = {
    ModeKeys.TRAIN: 'train',
    ModeKeys.EVAL: 'valid',
    ModeKeys.PREDICT: 'test',
}


class HyperParameter:
    """Model hyper parameters"""
    BATCH_SIZE = 32
    NB_TOKENS = 512
    VOCABULARY_SIZE = 10000
    EMBEDDING_SIZE = max(10, int(VOCABULARY_SIZE**0.5))
    DNN_HIDDEN_UNITS = [256, 128]
    DNN_DROPOUT = 0.2
    N_GRAM = 2


class Training:
    """Model training parameters"""
    SHUFFLE_BUFFER = 10000


def load(saved_model_dir: str) -> Model:
    """Load a Keras model"""
    return tf.keras.models.load_model(f"{saved_model_dir}model.keras")


def build_label_lookup(labels: List[str]) -> StaticHashTable:
    """
    Build the label lookup as a static hash table
    :param labels: the labels to build the lookup for
    """
    return StaticHashTable(
        KeyValueTensorInitializer(
            keys=tf.constant(labels),
            values=tf.constant(list(range(len(labels))), dtype=tf.int64),
        ),
        default_value=-1,
    )


def build(source_files_dir: str, label_lookup: StaticHashTable, labels_count: int) -> Model:
    """Build a Keras model"""

    LOGGER.debug('Building the input layer')
    input_layer = tf.keras.Input(shape=(1,), dtype=tf.string, name='content')

    LOGGER.debug('Building shared vectorization layer')
    shared_vectorize_layer = tf.keras.layers.TextVectorization(
        max_tokens=HyperParameter.VOCABULARY_SIZE,
        output_mode="int",
        output_sequence_length=HyperParameter.NB_TOKENS,
    )

    LOGGER.debug('Building TF-IDF vectorization layer')
    tfidf_vectorize_layer = tf.keras.layers.TextVectorization(
        max_tokens=HyperParameter.VOCABULARY_SIZE,
        output_mode="tf_idf",
        ngrams=HyperParameter.N_GRAM,
    )

    LOGGER.debug('Building adapt dataset with limit')
    adapt_ds = build_input_dataset(source_files_dir, label_lookup, ModeKeys.TRAIN, 50000)
    adapt_ds = adapt_ds.unbatch().map(lambda x, y: x["content"])

    LOGGER.debug('Starting adapt shared layer')
    shared_vectorize_layer.adapt(adapt_ds)

    LOGGER.debug('Starting adapt TF-IDF layer')
    tfidf_vectorize_layer.adapt(adapt_ds)

    LOGGER.debug('Building deep vectorized layer')
    deep_vectorized_layer = shared_vectorize_layer(input_layer)

    LOGGER.debug('Building wide vectorized layer')
    wide_vectorized_layer = tfidf_vectorize_layer(input_layer)

    LOGGER.debug('Building wide layer')
    wide_x = tf.keras.layers.LayerNormalization()(wide_vectorized_layer)

    LOGGER.debug('Building deep layer')
    deep_x = tf.keras.layers.Embedding(input_dim=HyperParameter.VOCABULARY_SIZE, output_dim=HyperParameter.EMBEDDING_SIZE)(deep_vectorized_layer)
    deep_x = tf.keras.layers.GlobalAveragePooling1D()(deep_x)
    deep_x = tf.keras.layers.Dense(HyperParameter.DNN_HIDDEN_UNITS[0], activation="relu")(deep_x)
    deep_x = tf.keras.layers.Dropout(HyperParameter.DNN_DROPOUT)(deep_x)
    deep_x = tf.keras.layers.Dense(HyperParameter.DNN_HIDDEN_UNITS[1], activation="relu")(deep_x)
    deep_x = tf.keras.layers.Dropout(HyperParameter.DNN_DROPOUT)(deep_x)

    LOGGER.debug('Concatenating wide and deep parts')
    wide_deep_concat_layer = tf.keras.layers.concatenate([wide_x, deep_x])

    LOGGER.debug('Building output layer')
    wide_deep_output_layer = tf.keras.layers.Dense(
        labels_count,
        name='logits',
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros'
    )(wide_deep_concat_layer)

    LOGGER.debug('Building the wide deep model')
    wide_deep_model = tf.keras.Model(inputs=input_layer, outputs=wide_deep_output_layer)

    LOGGER.debug('Compiling the model')
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        0.001,
        decay_steps=1000,
        decay_rate=0.9,
        staircase=True
    )
    wide_deep_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=1.0),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    LOGGER.debug('Model summary:')
    wide_deep_model.summary(print_fn=lambda x: LOGGER.debug(x))

    return wide_deep_model


def train(built_model: Model, source_files_dir: str, max_steps: int, label_lookup: StaticHashTable) -> Any:
    """Train a Keras model"""

    LOGGER.debug('Building TRAIN data set')
    train_ds = build_input_dataset(source_files_dir, label_lookup, ModeKeys.TRAIN)

    LOGGER.debug('Checking first batch...')
    for batch_x, batch_y in train_ds.take(1):
        LOGGER.debug(f'Batch X shape: {batch_x}')
        LOGGER.debug(f'Batch Y shape: {batch_y.shape}')
        LOGGER.debug(f'Batch Y min/max: {tf.reduce_min(batch_y)}/{tf.reduce_max(batch_y)}')
        LOGGER.debug(f'Batch Y unique values: {len(tf.unique(batch_y)[0])}')

        # Predictions check
        pred = built_model(batch_x, training=False)
        LOGGER.debug(f'Prediction shape: {pred.shape}')
        LOGGER.debug(f'Prediction min/max: {tf.reduce_min(pred)}/{tf.reduce_max(pred)}')

    LOGGER.debug('Building EVAL data set')
    eval_ds = build_input_dataset(source_files_dir, label_lookup, ModeKeys.EVAL)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir='./logs',
        histogram_freq=1,
        profile_batch='500,520' # profile batches from 500 to 520
    )

    LOGGER.debug('Training the model')
    class MemoryCleanupCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            import gc
            gc.collect()
            tf.keras.backend.clear_session()

            import psutil
            process = psutil.Process()
            mem_info = process.memory_info()
            print(f"\nMemory after epoch {epoch + 1}: {mem_info.rss / 1024 / 1024:.2f} MB")

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath='./custom_model/checkpoint/model_epoch_{epoch:02d}.keras',
        save_freq='epoch',
        save_best_only=False,
        verbose=1
    )
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=3,
        restore_best_weights=True
    )

    summary = built_model.fit(
        train_ds,
        epochs=max_steps,
        validation_data=eval_ds,
        callbacks=[
            tensorboard_callback,
            early_stopping_callback,
            MemoryCleanupCallback(),
            checkpoint_callback
        ]
    )

    LOGGER.debug('Building and saving training metrics')
    if summary.history:
        training_metrics = {
            'accuracy': summary.history['accuracy'][-1],
            'loss': summary.history['loss'][-1]
        }

        if 'val_accuracy' in summary.history:
            training_metrics['val_accuracy'] = summary.history['val_accuracy'][-1]

        if 'val_loss' in summary.history:
            training_metrics['val_loss'] = summary.history['val_loss'][-1]

        return training_metrics
    else:
        return {}


def save(trained_model: Model, labels: list[str], saved_model_dir: str) -> None:
    """Save and export the model in Keras, Saved Model, and ONNX formats. Save labels as well"""
    saved_model_path = Path(saved_model_dir)
    saved_model_path.mkdir(parents=True, exist_ok=True)

    keras_model_file = saved_model_path / 'model.keras'
    LOGGER.debug(f'Saving Keras model to {keras_model_file}')
    trained_model.save(keras_model_file)

    savedmodel_dir = saved_model_path / 'saved_model'
    LOGGER.debug(f'Exporting SavedModel to {savedmodel_dir}')
    trained_model.export(savedmodel_dir)

    onnx_file = saved_model_path / 'model.onnx'
    LOGGER.debug(f'Exporting ONNX model to {onnx_file}')
    try:
        trained_model.export(onnx_file, format="onnx")
    except Exception as e:
        LOGGER.warning(f'ONNX export failed: {e}')

    labels_file = saved_model_path / 'labels.json'
    LOGGER.debug(f'Saving labels to {labels_file}')
    with open(labels_file, 'w') as f:
        json.dump(labels, f, indent=2)

    LOGGER.info(f'Model saved successfully to {saved_model_dir}')


def test(
    trained_model: Model,
    label_lookup: StaticHashTable,
    labels: list[str],
    mapping: dict[str, str],
    data_root_dir: str,
) -> Dict[str, Dict[str, int]]:
    """Test a Keras model"""
    values = {language: 0 for language in mapping.values()}
    matches = {language: deepcopy(values) for language in values}

    test_dataset = build_input_dataset(data_root_dir, label_lookup, ModeKeys.PREDICT)

    LOGGER.debug('Test the model')
    for batch in test_dataset:
        content_batch, label_batch = batch

        # Get predictions
        predictions = trained_model.predict(content_batch, verbose=0)
        predicted_idx = np.argmax(predictions[0])
        predicted_label = labels[predicted_idx]

        # Get true label
        true_idx = int(label_batch.numpy()[0])
        true_label = labels[true_idx]

        # Map to languages
        label_language = mapping[true_label]
        predicted_language = mapping[predicted_label]
        matches[label_language][predicted_language] += 1

    return matches


def predict(
    saved_model,
    mapping: Dict[str, str],
    text: str
) -> List[Tuple[str, float]]:
    """Infer a Tensorflow saved model"""
    raise Exception("This functionality needs to be reworked")
    content_tensor = tf.constant([text])
    predicted = saved_model.signatures['serving_default'](content_tensor)

    numpy_floats = predicted['scores'][0].numpy()
    extensions = predicted['classes'][0].numpy()

    probability_values = (float(value) for value in numpy_floats)
    languages = (mapping[ext.decode()] for ext in extensions)

    unsorted_scores = zip(languages, probability_values)
    scores = sorted(unsorted_scores, key=itemgetter(1), reverse=True)
    return scores


def build_input_dataset(
    data_root_dir: str,
    label_lookup: StaticHashTable,
    mode: ModeKeys,
    samples_limit: int = None
) -> tf.data.Dataset:
    """Generate an input data set for a Keras model"""
    pattern = str(Path(data_root_dir).joinpath(DATASET[mode], '*'))
    file_paths = tf.io.gfile.glob(pattern)

    def _data_generator():
        """Produce the next content dictionary, gathered from a file"""
        if mode == ModeKeys.TRAIN:
            random.shuffle(file_paths)

        samples_yielded = 0

        for file_path in file_paths:
            if samples_limit and samples_yielded >= samples_limit:
                break

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Remove empty files + small portions of unpredictable data to eliminate the noise
                if len(content) <= 30:
                    continue

                label = file_path.split('.')[-1]

                yield {'content': content}, label
                samples_yielded += 1

                del content

            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                continue

    output_signature = (
        {'content': tf.TensorSpec(shape=(), dtype=tf.string)},
        tf.TensorSpec(shape=(), dtype=tf.string)
    )

    dataset = tf.data.Dataset.from_generator(
        _data_generator,
        output_signature=output_signature
    )

    # Perform label lookup build to tensor, return 'content' to data dict and the labels tensor
    dataset = dataset.map(
        lambda data, label: (data, label_lookup.lookup(label)),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    dataset = dataset.filter(lambda data, label_id: label_id >= 0)

    if mode == ModeKeys.PREDICT:
        return dataset.batch(1).prefetch(tf.data.AUTOTUNE)

    if mode == ModeKeys.TRAIN:
        dataset = dataset.shuffle(Training.SHUFFLE_BUFFER)

    dataset = dataset.batch(HyperParameter.BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset
