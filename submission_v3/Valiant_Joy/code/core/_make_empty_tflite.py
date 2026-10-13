"""Write a minimal TFLite model with a given input shape/dtype: the TFLite 'empty session' RSS baseline.

Run in a subprocess by core/measure.py (keeps TensorFlow out of the main process).
The single op (REVERSE_V2 for float inputs, CAST for int8/uint8) cannot be optimised
away, unlike Identity/Reshape. Exit code 3 = TensorFlow not installed.

  python core/_make_empty_tflite.py --shape 1,224,224,3 --dtype float32 --out empty.tflite
"""
import argparse
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", required=True)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    try:
        import tensorflow as tf
    except ImportError:
        sys.exit(3)
    shape = [int(s) for s in a.shape.split(",")]

    dtype = getattr(tf, a.dtype)

    @tf.function(input_signature=[tf.TensorSpec(shape, dtype, name="input")])
    def f(x):
        # float: REVERSE_V2; int8/uint8 inputs (full-integer models): CAST (REVERSE_V2 lacks int8)
        return tf.reverse(x, axis=[-1]) if dtype.is_floating else tf.cast(x, tf.float32)

    conv = tf.lite.TFLiteConverter.from_concrete_functions([f.get_concrete_function()], f)
    with open(a.out, "wb") as fh:
        fh.write(conv.convert())


if __name__ == "__main__":
    main()
