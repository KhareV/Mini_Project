import numpy as np

from inference.stream_replay import ECGStreamReplayer


class FakeModel:
    def predict(self, signal, source_fs):
        from inference.ecg_model_v1 import ECGPrediction
        return ECGPrediction("ABNORMAL" if signal.mean() > 0 else "NORMAL",
                              float(signal.mean()), 0.5, "TEST", "1.1.0")


def test_chunk_boundaries_do_not_change_stream_windows():
    signal = np.arange(6250, dtype=np.float32)
    expected = ECGStreamReplayer(FakeModel(), 500).ingest(signal)
    chunked = ECGStreamReplayer(FakeModel(), 500)
    actual = []
    for start in range(0, len(signal), 137):
        actual.extend(chunked.ingest(signal[start:start + 137]))
    assert [p.probability_abnormal for p in actual] == [p.probability_abnormal for p in expected]
    assert len(actual) == 1
    assert chunked.buffered_samples == 3750


def test_stream_emits_overlapping_windows_and_rejects_nan():
    stream = ECGStreamReplayer(FakeModel(), 10, window_seconds=1, stride_seconds=0.5)
    assert len(stream.ingest(np.ones(15, dtype=np.float32))) == 2
    assert len(stream.ingest(np.ones(10, dtype=np.float32))) == 2
    assert stream.windows_emitted == 4
    try:
        stream.ingest(np.array([np.nan]))
    except ValueError:
        pass
    else:
        raise AssertionError("non-finite samples must be rejected")
