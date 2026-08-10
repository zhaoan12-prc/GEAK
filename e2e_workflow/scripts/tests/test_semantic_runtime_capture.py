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
