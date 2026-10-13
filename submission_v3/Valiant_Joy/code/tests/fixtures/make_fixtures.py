"""Regenerate the tiny foreign-format fixtures (needs tensorflow + tf2onnx; NOT a project dependency).

  pip install tensorflow-cpu==2.21.0 tf2onnx==1.17.0   # in a separate venv
  python tests/fixtures/make_fixtures.py

Produces a Keras CNN trained on the same rule as tests/dummy_adapter.py
(label = [mean(ch0) > 0] + 2 * [mean(ch1) > 0]), input NHWC 1x16x16x3:
  tiny_f32.tflite      float TFLite
  tiny_int8.tflite     full-integer TFLite (int8 input/output)
  tiny_nhwc_b1.onnx    tf2onnx export: NHWC layout and a FIXED batch of 1
"""
from pathlib import Path

import numpy as np
import tensorflow as tf

HERE = Path(__file__).parent
rng = np.random.default_rng(0)
x = rng.standard_normal((2048, 16, 16, 3)).astype(np.float32)
y = (x[..., 0].mean((1, 2)) > 0).astype(int) + 2 * (x[..., 1].mean((1, 2)) > 0).astype(int)

layers = [tf.keras.layers.Conv2D(8, 3, padding="same", activation="relu"),
          tf.keras.layers.GlobalAveragePooling2D(), tf.keras.layers.Dense(4, name="logits")]
batched = tf.keras.Sequential([tf.keras.Input((16, 16, 3))] + layers)
batched.compile(optimizer=tf.keras.optimizers.Adam(1e-2),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])
batched.fit(x, y, epochs=15, batch_size=64, verbose=0)
print("train acc", batched.evaluate(x, y, verbose=0)[1])
# deployable graph with a fixed batch of 1 sharing the trained layers
inp = tf.keras.Input((16, 16, 3), batch_size=1, name="input")
h = inp
for layer in layers:
    h = layer(h)
model = tf.keras.Model(inp, h)

conv = tf.lite.TFLiteConverter.from_keras_model(model)
(HERE / "tiny_f32.tflite").write_bytes(conv.convert())

conv = tf.lite.TFLiteConverter.from_keras_model(model)
conv.optimizations = [tf.lite.Optimize.DEFAULT]
conv.representative_dataset = lambda: ([x[i:i + 1]] for i in range(200))
conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
conv.inference_input_type = tf.int8
conv.inference_output_type = tf.int8
(HERE / "tiny_int8.tflite").write_bytes(conv.convert())

import tf2onnx  # noqa: E402

tf2onnx.convert.from_tflite(str(HERE / "tiny_f32.tflite"), opset=17, output_path=str(HERE / "tiny_nhwc_b1.onnx"))
for f in ("tiny_f32.tflite", "tiny_int8.tflite", "tiny_nhwc_b1.onnx"):
    print(f, (HERE / f).stat().st_size, "bytes")
