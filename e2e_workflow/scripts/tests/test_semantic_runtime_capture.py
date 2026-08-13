import os
import sys
import types
import unittest
from unittest import mock


SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCRIPTS)
import semantic_runtime_capture as capture


class _Logger(object):
    def __init__(self):
        self.calls = []

    def begin_callable(self, target):
        self.calls.append(("begin", target))
        return target

    def end_callable(self, entry, args, kwargs, output):
        self.calls.append(("end", entry, args, kwargs, output))


class SemanticRuntimeCaptureTest(unittest.TestCase):
    def test_prefill_phase_filter_accepts_extend_runtime_mode(self):
        logger = capture.SemanticRuntimeLogger.__new__(
            capture.SemanticRuntimeLogger)
        logger.layers = {3}
        logger.phases = {"PREFILL"}
        logger.require_profiler = False
        logger._profile_seen = False
        logger.max_forwards = 1
        logger._bucket_forwards = {}
        logger._context = {
            "phase": "EXTEND", "batch_size": 1, "input_tokens": 8}
        with mock.patch.object(logger, "active", return_value=True):
            self.assertTrue(logger._allowed(3))

    def test_warmup_does_not_consume_bucket_before_profiler_starts(self):
        logger = capture.SemanticRuntimeLogger.__new__(
            capture.SemanticRuntimeLogger)
        logger.layers = {3}
        logger.phases = {"DECODE"}
        logger.require_profiler = True
        logger._profile_seen = False
        logger.max_forwards = 1
        logger._bucket_forwards = {}
        logger._context = {
            "phase": "DECODE", "batch_size": 4, "input_tokens": 4}
        with mock.patch.object(logger, "active", return_value=True), \
                mock.patch.object(
                    capture, "_profiler_active", return_value=False):
            self.assertFalse(logger._allowed(3))
            self.assertEqual(logger._bucket_forwards, {})
        with mock.patch.object(logger, "active", return_value=True), \
                mock.patch.object(
                    capture, "_profiler_active", return_value=True):
            self.assertTrue(logger._allowed(3))
            self.assertTrue(logger._profile_seen)
        with mock.patch.object(logger, "active", return_value=True), \
                mock.patch.object(
                    capture, "_profiler_active", return_value=False):
            self.assertTrue(logger._allowed(3))

    def test_pre_profiler_forward_does_not_spend_the_only_bucket(self):
        """A collapsed bucket space must survive the pre-profiler forwards.

        install_on_model calls mark_forward() after EVERY forward, including
        the warmup and benchmark forwards that ran before the profiler window
        opened and were therefore never recorded.  With a varied-length
        workload that only wasted some buckets -- a later one was always still
        unspent.  With a uniform-length workload there is exactly one prefill
        bucket and one steady decode batch size, so spending it there left the
        capture with no shape metadata at all.
        """
        logger = capture.SemanticRuntimeLogger.__new__(
            capture.SemanticRuntimeLogger)
        logger.layers = {3}
        logger.phases = {"PREFILL"}
        logger.require_profiler = True
        logger._profile_seen = False
        logger.max_forwards = 1
        logger._bucket_forwards = {}
        logger._context = {
            "phase": "EXTEND", "batch_size": 1, "input_tokens": 8192}

        # Warmup / pre-profiler forward: dropped, and must not be counted.
        with mock.patch.object(logger, "active", return_value=True), \
                mock.patch.object(
                    capture, "_profiler_active", return_value=False):
            self.assertFalse(logger._allowed(3))
        logger.mark_forward()
        self.assertEqual(logger._bucket_forwards, {})

        # The profiler window opens on a forward of the SAME bucket -- the only
        # bucket this workload produces.  Its budget must still be available.
        with mock.patch.object(logger, "active", return_value=True), \
                mock.patch.object(
                    capture, "_profiler_active", return_value=True):
            self.assertTrue(logger._allowed(3))
        logger.mark_forward()
        self.assertEqual(
            logger._bucket_forwards, {("EXTEND", 1, 8192): 1})

        # Budget is spent now, so a second recorded forward is refused.
        with mock.patch.object(logger, "active", return_value=True), \
                mock.patch.object(
                    capture, "_profiler_active", return_value=True):
            self.assertFalse(logger._allowed(3))

    def test_targeted_callable_is_monkeypatched_and_logged(self):
        module = types.SimpleNamespace(launcher=lambda value: value + 1)
        logger = _Logger()
        capture._PATCHED_CALLABLES.clear()
        with mock.patch.dict(
                os.environ,
                {"GEAK_SEMANTICS_CALLABLE_TARGETS": "pkg.mod:launcher"}), \
                mock.patch.object(
                    capture.importlib, "import_module", return_value=module), \
                mock.patch.object(capture, "get_logger", return_value=logger):
            capture._install_callable_probes()
            self.assertEqual(module.launcher(4), 5)
        self.assertEqual(logger.calls[0], ("begin", "pkg.mod:launcher"))
        self.assertEqual(logger.calls[1][-1], 5)


if __name__ == "__main__":
    unittest.main()
