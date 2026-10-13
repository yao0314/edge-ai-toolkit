"""stamp_onnx must not touch the weights of an external-data ONNX.

Two defects, both found by an independent audit and both reproduced here before fixing:

  APPEND  onnx.load() clears each tensor's external_data, so the re-save went through onnx's
          external-data writer, which opens the sidecar "r+b" and seeks to EOF. Every call
          appended a fresh full copy of every weight and never truncated. compress.py stamps each
          deliverable at packaging time, so this grew without bound.
  GUESS   the sidecar was located as `<model>.onnx` + ".data". A model whose tensors reference
          any other location (model.onnx_data, weights.bin) took the inline branch instead:
          multi-GB weights folded into the protobuf, original sidecar orphaned beside it, and
          measure.py then counted the weights twice in size_bytes.

The fix is to load with load_external_data=False - stamping only rewrites producer fields and
metadata_props, so the tensors never need to be read.
"""
import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper

from core.export import stamp_onnx


def _external_model(path, location, n=4096):
    """A model whose single initializer lives in `location` next to `path`."""
    w = np.random.RandomState(0).randn(n, 4).astype(np.float32)
    g = helper.make_graph(
        [helper.make_node("MatMul", ["input", "W"], ["output"])], "g",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, n])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4])],
        # raw=True matters: onnx only moves tensors that carry raw_data to an external file, so a
        # make_tensor built from a python list stays inline and this test would silently pass.
        [helper.make_tensor("W", TensorProto.FLOAT, [n, 4], w.tobytes(), raw=True)])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 10
    onnx.save(m, str(path), save_as_external_data=True, all_tensors_to_one_file=True,
              location=location, size_threshold=0)
    return path.parent / location


@pytest.mark.parametrize("location", ["m.onnx.data", "m.onnx_data", "weights.bin"])
def test_repeated_stamping_leaves_the_sidecar_byte_identical(tmp_path, location):
    p = tmp_path / "m.onnx"
    ext = _external_model(p, location)
    before = ext.read_bytes()
    assert len(before) > 0

    for _ in range(3):
        stamp_onnx(p)

    assert ext.exists(), f"sidecar {location} was orphaned or deleted"
    assert ext.read_bytes() == before, (
        f"sidecar changed after stamping ({len(before)} -> {ext.stat().st_size} bytes)")

    # The sidecar staying byte-identical is not sufficient: the inline branch left it untouched on
    # disk while folding the weights into the protobuf and orphaning it. Assert the tensors still
    # POINT at it, and that the .onnx did not absorb them.
    m = onnx.load(str(p), load_external_data=False)
    locs = {e.value for t in m.graph.initializer for e in t.external_data if e.key == "location"}
    assert all(t.data_location == TensorProto.EXTERNAL for t in m.graph.initializer),         "weights were inlined into the protobuf; the sidecar is now an orphan"
    assert locs == {location}, f"tensors reference {locs}, expected {{{location!r}}}"
    assert p.stat().st_size < len(before),         f".onnx grew to {p.stat().st_size} B (sidecar is {len(before)} B): weights were absorbed"


@pytest.mark.parametrize("location", ["m.onnx.data", "weights.bin"])
def test_stamped_external_model_still_loads_and_runs(tmp_path, location):
    """Inlining or corrupting the weights would show up here, not just as a size change."""
    p = tmp_path / "m.onnx"
    _external_model(p, location)
    ref = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
    x = np.random.rand(1, 4096).astype(np.float32)
    expected = ref.run(None, {"input": x})[0]
    del ref

    stamp_onnx(p)
    got = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"]).run(None, {"input": x})[0]
    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)


def test_stamping_writes_provenance_and_is_idempotent(tmp_path):
    p = tmp_path / "m.onnx"
    _external_model(p, "m.onnx.data")
    stamp_onnx(p)
    stamp_onnx(p)
    m = onnx.load(str(p), load_external_data=False)
    keys = [x.key for x in m.metadata_props]
    assert m.producer_name, "producer_name was not stamped"
    assert len(keys) == len(set(keys)), f"re-stamping duplicated metadata: {keys}"


def test_plain_model_without_external_data_is_unaffected(tmp_path):
    p = tmp_path / "plain.onnx"
    w = np.ones((4, 4), dtype=np.float32)
    g = helper.make_graph(
        [helper.make_node("MatMul", ["input", "W"], ["output"])], "g",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor("W", TensorProto.FLOAT, [4, 4], w.tobytes(), raw=True)])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 10
    onnx.save(m, str(p))
    stamp_onnx(p)
    assert not list(tmp_path.glob("*.data")), "stamping invented a sidecar for an inline model"
    s = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
    assert s.run(None, {"input": np.ones((1, 4), dtype=np.float32)})[0].shape == (1, 4)
