import json

import numpy as np
import tensorflow as tf
import tf2onnx

SEQUENCE_LENGTH = 512  # NB_TOKENS
TFIDF_SIZE = 10000  # VOCABULARY_SIZE
VOCAB_SIZE = 10000  # VOCABULARY_SIZE
EMBEDDING_SIZE = 100
DENSE_1_UNITS = 256  # DNN_HIDDEN_UNITS[0]
DENSE_2_UNITS = 128  # DNN_HIDDEN_UNITS[1]
DROPOUT_RATE = 0.2  # DNN_DROPOUT
N_GRAM = 2  # N_GRAM
OUTPUT_UNITS = 54  # labels count

def main():
    print("Loading trained model...")
    trained_model = tf.keras.models.load_model("./model.keras")
    trained_model.summary()

    print("Extracting vectorization layers...")

    text_vec_int = None
    text_vec_tfidf = None

    for layer in trained_model.layers:
        if isinstance(layer, tf.keras.layers.TextVectorization):
            # text_vectorization -> (None, 512) - INT mode
            # text_vectorization_1 -> (None, 10000) - TF-IDF mode

            if 'text_vectorization_1' in layer.name:
                text_vec_tfidf = layer
                print(f"TF-IDF layer: {layer.name}")
            else:
                text_vec_int = layer
                print(f"INT layer: {layer.name}")

    if text_vec_int:
        vocab = text_vec_int.get_vocabulary()
        with open('vocab_int.txt', 'w', encoding='utf-8') as f:
            for word in vocab:
                f.write(f"{word}\n")
        print(f"Saved INT vocabulary: {len(vocab)} words")

    if text_vec_tfidf:
        print(f"Found TF-IDF layer: {text_vec_tfidf.name}")

        vocab = text_vec_tfidf.get_vocabulary()
        vocab_size = len(vocab)
        print(f"Vocabulary size: {vocab_size}")

        print("Extracting IDF weights by testing each term...")

        idf_weights = np.zeros(vocab_size, dtype=np.float32)

        batch_size = 100
        for i in range(0, vocab_size, batch_size):
            batch_end = min(i + batch_size, vocab_size)
            batch_words = vocab[i:batch_end]

            for j, word in enumerate(batch_words):
                if word and word.strip():
                    try:
                        text = [word]
                        output = text_vec_tfidf(text).numpy()

                        idx = i + j
                        if output[0, idx] > 0:
                            idf_weights[idx] = output[0, idx]
                    except:
                        pass

            if (i // batch_size) % 10 == 0:
                print(f"  Processed {i}/{vocab_size} words...")

        np.save('tfidf_idf_weights.npy', idf_weights)
        np.savetxt('tfidf_idf_weights.csv', idf_weights, delimiter=',', fmt='%.8f')

        print(f"Extracted IDF weights:")
        print(f"  Shape: {idf_weights.shape}")
        print(f"  Non-zero values: {np.count_nonzero(idf_weights)}")
        print(f"  Min (non-zero): {idf_weights[idf_weights > 0].min():.4f}")
        print(f"  Max: {idf_weights.max():.4f}")
        print(f"  Mean (non-zero): {idf_weights[idf_weights > 0].mean():.4f}")
        print(f"\n  Sample IDF values:")
        for i in range(min(10, vocab_size)):
            if idf_weights[i] > 0:
                print(f"    '{vocab[i]}': {idf_weights[i]:.4f}")

    else:
        print("TF-IDF layer not found")

    test_text = ["This is a test sentence"]

    if text_vec_int and text_vec_tfidf:
        int_output = text_vec_int(test_text)
        tfidf_output = text_vec_tfidf(test_text)

        print(f"Test vectorization:")
        print(f"INT output shape: {int_output.shape}")
        print(f"INT sample: {int_output[0][:10]}")
        print(f"TF-IDF output shape: {tfidf_output.shape}")
        print(f"TF-IDF sample: {tfidf_output[0][:10]}")

        # Examples to test in Java
        np.save('test_deep_input.npy', int_output.numpy())
        np.save('test_wide_input.npy', tfidf_output.numpy())
        print("Saved test inputs for Java verification")


    print("Building ONNX-compatible model...")

    config = {
        'sequence_length': SEQUENCE_LENGTH,
        'tfidf_size': TFIDF_SIZE,
        'vocab_size': VOCAB_SIZE,
        'embedding_size': EMBEDDING_SIZE,
        'dense_units': [DENSE_1_UNITS, DENSE_2_UNITS],
        'dropout_rate': DROPOUT_RATE,
        'n_gram': N_GRAM,
        'output_classes': OUTPUT_UNITS,
        'model_version': '1.0'
    }

    with open('model_config.json', 'w') as f:
        json.dump(config, f, indent=2)

    print("Saved model config to model_config.json: success")
    print("Building ONNX-compatible inference model...")

    # Input layers
    deep_input = tf.keras.Input(shape=(SEQUENCE_LENGTH,), dtype=tf.int32, name='deep_input')
    wide_input = tf.keras.Input(shape=(TFIDF_SIZE,), dtype=tf.float32, name='wide_input')

    # Deep part
    deep_x = tf.keras.layers.Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_SIZE, name='embedding')(deep_input)
    deep_x = tf.keras.layers.GlobalAveragePooling1D(name='global_average_pooling1d')(deep_x)
    deep_x = tf.keras.layers.Dense(DENSE_1_UNITS, activation='relu', name='dense')(deep_x)
    deep_x = tf.keras.layers.Dropout(DROPOUT_RATE, name='dropout')(deep_x)
    deep_x = tf.keras.layers.Dense(DENSE_2_UNITS, activation='relu', name='dense_1')(deep_x)
    deep_x = tf.keras.layers.Dropout(DROPOUT_RATE, name='dropout_1')(deep_x)

    # Wide part
    wide_x = tf.keras.layers.LayerNormalization(name='layer_normalization')(wide_input)

    # Deep + wide concat
    concat = tf.keras.layers.Concatenate(name='concatenate')([wide_x, deep_x])

    # Output layer
    output = tf.keras.layers.Dense(OUTPUT_UNITS, name='logits')(concat)

    # Model
    inference_model = tf.keras.Model(inputs=[deep_input, wide_input], outputs=output, name='inference_model')

    print("ONNX-compatible inference model created")
    inference_model.summary()

    print("Transferring weights from trained model...")

    transferred_count = 0
    skipped_count = 0

    for target_layer in inference_model.layers:
        layer_name = target_layer.name

        # Input and Dropout layers are skipped, because they do not have weights
        if isinstance(target_layer, (tf.keras.layers.InputLayer, tf.keras.layers.Dropout)):
            skipped_count += 1
            continue

        try:
            source_layer = trained_model.get_layer(layer_name)
            weights = source_layer.get_weights()

            if len(weights) > 0:
                target_layer.set_weights(weights)
                transferred_count += 1

                weight_info = ", ".join([f"{w.shape}" for w in weights])
                print(f"{layer_name}: {weight_info}")
        except ValueError as e:
            print(f"{layer_name}: NOT FOUND in source model: {e}")

    print(f"Transferred weights for {transferred_count} layers")
    print(f"Skipped {skipped_count} layers (no weights)")

    print("Testing inference model...")

    batch_size = 2
    deep_test = np.random.randint(0, VOCAB_SIZE, size=(batch_size, SEQUENCE_LENGTH), dtype=np.int32)
    wide_test = np.random.rand(batch_size, TFIDF_SIZE).astype(np.float32)

    output = inference_model.predict([deep_test, wide_test], verbose=0)
    print(f"Test output shape: {output.shape}")
    print(f"Sample logits: {output[0][:5]}")

    print("Exporting to ONNX format...")

    spec = (
        tf.TensorSpec((None, SEQUENCE_LENGTH), tf.int32, name="deep_input"),
        tf.TensorSpec((None, TFIDF_SIZE), tf.float32, name="wide_input")
    )

    try:
        model_proto, _ = tf2onnx.convert.from_keras(
            inference_model,
            input_signature=spec,
            opset=13,
            output_path="./model.onnx"
        )
        print("Model successfully exported to model.onnx")
    except Exception as e:
        print(f"Export failed: {e}")
        raise

    print("Generated files:")
    print(" - model.onnx                - ONNX model for Java")
    print(" - vocab_int.txt             - Vocabulary for deep branch (int mode)")
    print(" - vocab_tfidf.txt           - Vocabulary for wide branch (tf-idf)")
    print(" - tfidf_idf_weights.npy     - IDF weights for TF-IDF calculation")
    print(" - model_config.json         - Model configuration")

    print("Model parameters:")
    print(f"  Sequence length:     {SEQUENCE_LENGTH}")
    print(f"  TF-IDF size:         {TFIDF_SIZE}")
    print(f"  Vocabulary size:     {VOCAB_SIZE}")
    print(f"  Embedding size:      {EMBEDDING_SIZE}")
    print(f"  Hidden units:        {DENSE_1_UNITS} -> {DENSE_2_UNITS}")
    print(f"  Output classes:      {OUTPUT_UNITS}")
    print(f"  N-grams:             {N_GRAM}")
    print(f"  Dropout rate:        {DROPOUT_RATE}")

    print("Next steps for Java integration:")
    print("  1. Implement TextPreprocessor class")
    print("  2. Load vocabularies and IDF weights")
    print("  3. Vectorize input text:")
    print("     - Deep input: tokenize -> map to indices -> pad to 512")
    print("     - Wide input: extract 2-grams -> calculate TF-IDF -> vector of 10000")
    print("  4. Load model.onnx with ONNX Runtime")
    print("  5. Run inference with preprocessed inputs")
    print("  6. Apply softmax to logits for probabilities")

    print("Great success")


main()
