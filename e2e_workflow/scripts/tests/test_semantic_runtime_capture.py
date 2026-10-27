import contextlib
import io
import os
import sys
import tempfile
import types
import unittest
from unittest import mock


SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCRIPTS)
import semantic_runtime_capture as capture


class _Logger(object):
    def __init__(self):
        self.calls = []

    def begin_callable(self, target, args, kwargs, callable_obj):
        self.calls.append((
            "begin", target, tuple(args), dict(kwargs), callable_obj))
        return target

    def end_callable(self, entry, args, kwargs, output):
        self.calls.append(("end", entry, args, kwargs, output))


class SemanticRuntimeCaptureTest(unittest.TestCase):
    def test_all_layer_scope_ignores_representative_shape_filter(self):
        logger = capture.SemanticRuntimeLogger.__new__(
            capture.SemanticRuntimeLogger)
        logger.layer_scopes = True
        logger.layers = {3}
        logger.phases = {"DECODE"}
        logger.max_forwards = 1
        logger._bucket_forwards = {}
        logger._context = {
            "phase": "DECODE", "batch_size": 4, "input_tokens": 4}
        with mock.patch.object(logger, "active", return_value=True), \
                mock.patch.object(capture, "_stream_capturing",
                                  return_value=False):
            self.assertEqual(logger._layer_scope_source(17), "eager")
        self.assertTrue(capture._MAIN_LAYER_RE.search("model.layers.17"))
        self.assertFalse(capture._MAIN_LAYER_RE.search("model.layers.17.mlp"))

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
        self.assertEqual(logger.calls[0][0:3], (
            "begin", "pkg.mod:launcher", (4,)))
        self.assertEqual(logger.calls[1][-1], 5)

    def test_dispatcher_schema_record_preserves_argument_names_and_mutation(self):
        class Alias(object):
            is_write = True
            before_set = {"a"}
            after_set = {"a"}

            def __str__(self):
                return "Tensor(a!)"

        class Argument(object):
            def __init__(self, name, value_type, default=False, alias=None):
                self.name = name
                self.type = value_type
                self.kwarg_only = False
                self.default_value = None
                self.alias_info = alias
                self._default = default

            def has_default_value(self):
                return self._default

        class Schema(object):
            name = "custom::quant"
            overload_name = ""
            arguments = [
                Argument("out", "Tensor", alias=Alias()),
                Argument("input", "Tensor"),
                Argument("scale", "Tensor"),
                Argument("shuffle", "bool", default=True),
            ]
            returns = []

            def __str__(self):
                return "custom::quant(Tensor(a!) out, Tensor input, Tensor scale, bool shuffle=False) -> ()"

        record = capture._schema_record(Schema())
        self.assertEqual(record["name"], "custom::quant")
        self.assertEqual(
            [argument["name"] for argument in record["arguments"]],
            ["out", "input", "scale", "shuffle"])
        self.assertTrue(
            record["arguments"][0]["alias_info"]["is_write"])
        self.assertTrue(record["arguments"][3]["has_default"])



class TestBucketAccountingUnderAWarmupDriver(unittest.TestCase):
    """The bucket counter must not be spent by forwards the recorder refused.

    `_allowed` blocks recording until torch.profiler is live, so a warmup forward can
    leave no trace marker to match. `mark_forward` used to increment anyway, so a driver
    that warms up BEFORE opening the profile window -- bench_e2e.sh, by design -- arrived
    at the capture window with every bucket already full and wrote an EMPTY shape log.
    Empty is indistinguishable downstream from "this phase has no shapes", so this failed
    silently; observed on the first vLLM eager probe.
    """

    def _logger(self, tmp, require_profiler=True):
        env = {
            "GEAK_SEMANTICS_CAPTURE": "1",
            "GEAK_SEMANTICS_SHAPE_LOG": os.path.join(tmp, "shape.jsonl"),
            "GEAK_SEMANTICS_FORWARDS_PER_BUCKET": "1",
            "GEAK_SEMANTICS_REQUIRE_PROFILER": "1" if require_profiler else "0",
        }
        with mock.patch.dict(os.environ, env, clear=False):
            return capture.SemanticRuntimeLogger()

    def test_warmup_forwards_do_not_consume_the_bucket(self):
        with tempfile.TemporaryDirectory() as tmp:
            logger = self._logger(tmp)
            logger.set_context("DECODE", 8, 8)
            for _ in range(50):                 # a whole warmup + timed-repeat phase
                logger.mark_forward()
            self.assertEqual(logger._bucket_forwards, {},
                             "no bucket may be spent before the profiler is live")

            # Profiler comes up; the very next forward must still be admitted.
            logger._profile_seen = True
            logger.mark_forward()
            self.assertEqual(logger._bucket_forwards[("DECODE", 8, 8)], 1)

    def test_the_limit_still_applies_once_the_profiler_is_live(self):
        with tempfile.TemporaryDirectory() as tmp:
            logger = self._logger(tmp)
            logger._profile_seen = True
            logger.set_context("DECODE", 8, 8)
            logger.mark_forward()
            logger.mark_forward()
            self.assertEqual(logger._bucket_forwards[("DECODE", 8, 8)], 2)

    def test_counting_is_unconditional_when_no_profiler_is_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            logger = self._logger(tmp, require_profiler=False)
            logger.set_context("EXTEND", 2, 4096)
            logger.mark_forward()
            self.assertEqual(logger._bucket_forwards[("EXTEND", 2, 4096)], 1)


class _ScopeLogger(object):
    def __init__(self):
        self.scopes = []
        self.active = []

    def layer_scope(self, layer_id, op_path):
        self.scopes.append((layer_id, op_path))
        return contextlib.nullcontext()

    def layer_active(self, layer_id, op_path):
        self.active.append((layer_id, op_path))
        return contextlib.nullcontext()


class _Layer(object):
    def forward_entry(self, value):
        return value + 1


class _Model(object):
    def __init__(self):
        self.layer0, self.layer1, self.mlp = _Layer(), _Layer(), object()

    def named_modules(self, remove_duplicate=True):
        return [("", self), ("model.layers.0", self.layer0),
                ("model.layers.1", self.layer1),
                ("model.layers.0.mlp", self.mlp)]


class LayerEntryCallablesTest(unittest.TestCase):
    def _wrap(self, env, module=None):
        model, logger = _Model(), _ScopeLogger()
        clean = {key: value for key, value in os.environ.items()
                 if not key.startswith("GEAK_SEMANTICS_LAYER_ENTRY_")}
        clean.update(env)
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.dict(os.environ, clean, clear=True))
            if module is not None:
                stack.enter_context(mock.patch.object(
                    capture.importlib, "import_module", return_value=module))
            wrapped = capture._wrap_layer_entry_functions(model, logger)
        return model, logger, wrapped

    def test_nothing_beyond_forward_is_wrapped_by_default(self):
        model, logger, wrapped = self._wrap({})
        self.assertEqual(wrapped, 0)
        self.assertEqual(model.layer0.forward_entry(1), 2)
        self.assertEqual(logger.scopes, [])

    def test_configured_method_marks_each_main_layer(self):
        model, logger, wrapped = self._wrap(
            {"GEAK_SEMANTICS_LAYER_ENTRY_METHODS": "forward_entry"})
        self.assertEqual(wrapped, 2)
        self.assertEqual(model.layer1.forward_entry(1), 2)
        self.assertEqual(logger.scopes, [(1, "model.layers.1")])
        self.assertEqual(logger.active, [(1, "model.layers.1")])

    def test_configured_function_is_replaced_in_its_calling_module(self):
        module = types.SimpleNamespace(run_layer=lambda layer, value: value * 2)
        model, logger, _ = self._wrap(
            {"GEAK_SEMANTICS_LAYER_ENTRY_FUNCTIONS": "pkg.model:run_layer"},
            module)
        self.assertEqual(module.run_layer(model.layer0, 3), 6)
        self.assertEqual(module.run_layer(object(), 3), 6)
        self.assertEqual(logger.scopes, [(0, "model.layers.0")])

    def test_misconfigured_entries_fail_loudly(self):
        with self.assertRaises(RuntimeError):
            self._wrap({"GEAK_SEMANTICS_LAYER_ENTRY_METHODS": "no_such_method"})
        with self.assertRaises(RuntimeError):
            self._wrap({"GEAK_SEMANTICS_LAYER_ENTRY_FUNCTIONS": "no_colon"})
        with self.assertRaises(RuntimeError):
            self._wrap({"GEAK_SEMANTICS_LAYER_ENTRY_FUNCTIONS": "pkg.model:missing"},
                       types.SimpleNamespace())

    def _backbone(self, target):
        """A monokernel-style ``Class.method`` entry: the layer is the
        ``.module`` of the record passed after ``self``."""
        class Backbone(object):
            def _layer(self, record, value):
                return value * 2

        model, logger, _ = self._wrap(
            {"GEAK_SEMANTICS_LAYER_ENTRY_FUNCTIONS": target},
            types.SimpleNamespace(Backbone=Backbone))
        self.addCleanup(capture._LAYER_ENTRY_STATS.clear)
        return model, logger, Backbone()

    def test_class_method_locates_the_layer_through_an_argument_attribute(self):
        model, logger, backbone = self._backbone("pkg.mono:Backbone._layer@1.module")
        record = types.SimpleNamespace(module=model.layer1)
        self.assertEqual(backbone._layer(record, 3), 6)
        self.assertEqual(logger.scopes, [(1, "model.layers.1")])

    def test_unresolved_calls_are_counted_and_warned_once_per_reason(self):
        target = "pkg.mono:Backbone._layer@1.module"
        model, logger, backbone = self._backbone(target)
        with mock.patch.object(sys, "stderr", io.StringIO()) as err:
            for _ in range(4):          # 2 forwards x 2 main layers
                backbone._layer(model.layer0, 3)
            backbone._layer(types.SimpleNamespace(module=object()), 3)
        stats = capture._LAYER_ENTRY_STATS[target]
        self.assertEqual((stats["calls"], stats["hits"]), (5, 0))
        self.assertEqual(stats["misses"],
                         {"missing_attr": 4, "not_main_layer": 1})
        self.assertEqual(err.getvalue().count("has no .module"), 1)
        self.assertIn("ran 4 times without resolving a main layer",
                      err.getvalue())
        self.assertEqual(logger.scopes, [])

    def test_impossible_locators_fail_at_wrap_time(self):
        for target in ("pkg.mono:Backbone._layer@3",   # self, record, value
                       "pkg.mono:Backbone._layer@x.module"):
            with self.assertRaises(RuntimeError):
                self._backbone(target)

    def test_nested_scope_for_the_same_layer_emits_one_marker(self):
        markers = []

        class _Record(object):
            def __init__(self, name):
                self.name = name

            def __enter__(self):
                markers.append(self.name)

            def __exit__(self, *exc):
                return False

        fake_torch = types.SimpleNamespace(
            profiler=types.SimpleNamespace(record_function=_Record))
        logger = capture.SemanticRuntimeLogger.__new__(
            capture.SemanticRuntimeLogger)
        logger._scope_state = capture.threading.local()
        logger._context = {"phase": "DECODE", "batch_size": 4, "input_tokens": 4}
        with mock.patch.dict(sys.modules, {"torch": fake_torch}), \
                mock.patch.object(logger, "_layer_scope_source",
                                  return_value="capture"):
            with logger.layer_scope(3, "model.layers.3"):
                with logger.layer_scope(3, "model.layers.3"):
                    pass
                with logger.layer_scope(4, "model.layers.4"):
                    pass
            with logger.layer_scope(3, "model.layers.3"):
                pass
        self.assertEqual([m.split("|layer=")[1].split("|")[0] for m in markers],
                         ["3", "4", "3"])


if __name__ == "__main__":
    unittest.main()


class _BackRefModel(object):
    """A model whose layer i keeps a registered link to layer i+1 (DSv4.1's
    `_next_layer`): named_modules() reaches layer 1 through layer 0 first."""

    def __init__(self):
        self.layer0, self.layer1 = _Layer(), _Layer()

    def named_modules(self, remove_duplicate=True):
        rows = [("", self), ("model.layers.0", self.layer0),
                ("model.layers.0._next_layer", self.layer1),
                ("model.layers.1", self.layer1)]
        if not remove_duplicate:
            return rows
        seen, out = set(), []
        for path, module in rows:
            if id(module) not in seen:
                seen.add(id(module))
                out.append((path, module))
        return out


class CanonicalModulePathTest(unittest.TestCase):
    def test_a_back_reference_does_not_hide_the_main_layer_path(self):
        paths = [path for path, _ in
                 capture._canonical_named_modules(_BackRefModel())]
        self.assertEqual(paths, ["", "model.layers.0", "model.layers.1"])

    def test_entry_wrapping_marks_the_layer_under_its_own_index(self):
        model, logger = _BackRefModel(), _ScopeLogger()
        env = {key: value for key, value in os.environ.items()
               if not key.startswith("GEAK_SEMANTICS_LAYER_ENTRY_")}
        env["GEAK_SEMANTICS_LAYER_ENTRY_METHODS"] = "forward_entry"
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertEqual(
                capture._wrap_layer_entry_functions(model, logger), 2)
        model.layer1.forward_entry(1)
        self.assertEqual(logger.scopes, [(1, "model.layers.1")])


class LayerActiveTest(unittest.TestCase):
    def _logger(self):
        logger = capture.SemanticRuntimeLogger.__new__(
            capture.SemanticRuntimeLogger)
        logger._stacks = capture.threading.local()
        return logger

    def test_layer_body_is_the_launcher_parent_while_it_runs(self):
        logger = self._logger()
        with logger.layer_active(7, "model.layers.7"):
            parent = logger._active_modules()[-1]
            self.assertEqual((parent["layer_id"], parent["op_path"]),
                             (7, "model.layers.7"))
        self.assertEqual(logger._active_modules(), [])

    def test_out_of_order_exit_removes_only_its_own_entry(self):
        logger = self._logger()
        inner = {"layer_id": 7, "op_path": "model.layers.7.mlp"}
        with logger.layer_active(7, "model.layers.7"):
            logger._active_modules().append(inner)
        self.assertEqual(logger._active_modules(), [inner])


class DraftGraphRunnerTest(unittest.TestCase):
    """DSPARK draft workers capture with the base decode graph runner class, so
    the runner instance -- not the class -- decides whether it exports."""

    def _install(self, tmp):
        class _Runner(object):
            def __init__(self, draft):
                self.model_runner = types.SimpleNamespace(is_draft_worker=draft)

        class _Prof(object):
            def __init__(self, **kwargs):
                self.exported = []

            def export_chrome_trace(self, path):
                self.exported.append(path)

        activity = types.SimpleNamespace(CPU="cpu", CUDA="cuda")
        fake_torch = types.ModuleType("torch")
        fake_torch.profiler = types.SimpleNamespace(
            ProfilerActivity=activity, profile=_Prof)
        runner_mod = types.SimpleNamespace(CudaGraphRunner=_Runner)
        modules = {
            "torch": fake_torch, "torch.profiler": fake_torch.profiler,
            "sglang": types.ModuleType("sglang"),
            "sglang.srt": types.ModuleType("sglang.srt"),
            "sglang.srt.model_executor": types.ModuleType(
                "sglang.srt.model_executor"),
            "sglang.srt.model_executor.cuda_graph_runner": runner_mod,
        }
        env = {"GEAK_SEMANTICS_GRAPH_CAPTURE_TRACE": os.path.join(tmp, "g.json"),
               "GEAK_SEMANTICS_RANK": "0"}
        with mock.patch.dict(sys.modules, modules), \
                mock.patch.dict(os.environ, env), \
                mock.patch.object(capture, "_GRAPH_CAPTURE_PROFILER_INSTALLED",
                                  False), \
                mock.patch.object(capture, "_install_draft_graph_profiler_noop"):
            capture.install_graph_capture_profiler()
        return _Runner, _Prof

    def test_only_the_target_runner_exports_its_capture_trace(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner_cls, prof_cls = self._install(tmp)
            target, draft = runner_cls(False), runner_cls(True)
            self.assertIsInstance(
                runner_cls._init_profile_context_and_memory_record(target),
                prof_cls)
            self.assertIsNone(
                runner_cls._init_profile_context_and_memory_record(
                    draft).__enter__())
            prof = prof_cls()
            runner_cls._post_process_after_profile(draft, prof)
            self.assertEqual(prof.exported, [])
            with mock.patch.object(capture, "_graph_capture_trace_path",
                                   return_value=os.path.join(tmp, "g.json")):
                runner_cls._post_process_after_profile(target, prof)
            self.assertEqual(prof.exported, [os.path.join(tmp, "g.json")])

    def test_a_runner_without_a_model_runner_is_the_target(self):
        self.assertFalse(capture._is_draft_graph_runner(object()))



class CallableProbeClassMemberTest(unittest.TestCase):
    """A callable target may name a class member (`module:Class.method`). DeepSeek-V4's
    `MQALayer.forward` target crashed every scheduler at load with AttributeError when the
    probe looked the dotted name up on the module."""

    def test_class_members_are_wrapped_where_they_live(self):
        import types
        import semantic_runtime_capture as cap
        mod = types.ModuleType("geak_fake_probe_target")

        class Layer:
            def forward(self, x):
                return x + 1

            @staticmethod
            def s(x):
                return x * 2

            @classmethod
            def c(cls, x):
                return (cls.__name__, x)

        mod.Layer = Layer
        sys.modules[mod.__name__] = mod
        self.addCleanup(sys.modules.pop, mod.__name__, None)
        seen = []

        class Logger:
            def begin_callable(self, target, *_):
                seen.append(target)

            def end_callable(self, *_):
                pass

        saved_logger, saved_env = cap._LOGGER, os.environ.get("GEAK_SEMANTICS_CALLABLE_TARGETS")
        self.addCleanup(setattr, cap, "_LOGGER", saved_logger)
        cap._LOGGER = Logger()
        os.environ["GEAK_SEMANTICS_CALLABLE_TARGETS"] = ",".join(
            mod.__name__ + ":" + a for a in ("Layer.forward", "Layer.s", "Layer.c"))
        try:
            cap._install_callable_probes()
        finally:
            if saved_env is None:
                os.environ.pop("GEAK_SEMANTICS_CALLABLE_TARGETS", None)
            else:
                os.environ["GEAK_SEMANTICS_CALLABLE_TARGETS"] = saved_env
        self.assertEqual(Layer().forward(1), 2)
        self.assertEqual(Layer.s(3), 6)
        self.assertEqual(Layer.c(4), ("Layer", 4))
        self.assertEqual(len(seen), 3)


class ContainerHookTest(unittest.TestCase):
    """Inside the selected layers every container gets a marker, whatever its class name
    (DeepSeek-V4's attention is `MQALayer`); GEAK_SEMANTICS_HOOK_CONTAINERS=0 restores the
    class-name rule."""

    def _hooked(self, env):
        try:
            import torch
        except ImportError:
            self.skipTest("torch unavailable")
        import semantic_runtime_capture as cap

        class MQALayer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.wq = torch.nn.Linear(2, 2)

        class Block(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.self_attn = MQALayer()

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.ModuleList([Block(), Block()])

        model = Model()
        logger = types.SimpleNamespace(
            active=lambda: True, layers={1}, layer_scope=None, layer_active=None)
        with mock.patch.object(cap, "get_logger", return_value=logger), \
                mock.patch.object(cap, "_wrap_layer_entry_functions", return_value=0), \
                mock.patch.dict(os.environ, env):
            cap._register_hooks(model)
        return {name for name, mod in model.named_modules() if mod._forward_pre_hooks}

    def test_a_container_without_a_known_class_name_is_hooked_in_selected_layers(self):
        hooked = self._hooked({"GEAK_SEMANTICS_HOOK_CONTAINERS": "1"})
        self.assertIn("layers.1.self_attn", hooked)
        self.assertIn("layers.1.self_attn.wq", hooked)
        self.assertNotIn("layers.0.self_attn", hooked)      # not a selected layer
        self.assertNotIn("layers", hooked)                  # a ModuleList has no forward

    def test_the_class_name_rule_is_restorable(self):
        hooked = self._hooked({"GEAK_SEMANTICS_HOOK_CONTAINERS": "0"})
        self.assertNotIn("layers.1.self_attn", hooked)
        self.assertIn("layers.1.self_attn.wq", hooked)


class ForwardBatchLookupTest(unittest.TestCase):
    def test_forward_batch_is_found_anywhere_and_its_absence_warned_once(self):
        batch = types.SimpleNamespace(forward_mode="DECODE")
        lookup = capture._forward_batch_of
        with mock.patch.object(capture, "_WARNED_FORWARDS", set()), \
                mock.patch.object(sys, "stderr", io.StringIO()) as err:
            self.assertIs(lookup((1, 2), {"forward_batch": batch}, "f"), batch)
            self.assertIs(lookup((1, 2, batch), {}, "f"), batch)
            self.assertIs(lookup((batch, 1, 2), {}, "f"), batch)
            self.assertIsNone(lookup((1, 2, 3), {}, "f"))
            self.assertIsNone(lookup((1, 2, 3), {}, "f"))
        self.assertEqual(err.getvalue().count("f got no forward_batch"), 1)

    def test_a_runtime_installed_instance_forward_sets_the_phase(self):
        class Model(object):
            def forward(self, *args):
                return "class"

        model, logger = Model(), mock.Mock()
        model.forward = lambda input_ids, positions, forward_batch: "mono"
        with mock.patch.multiple(
                capture, get_logger=mock.Mock(return_value=logger),
                install_graph_capture_profiler=mock.DEFAULT,
                dump_operator_schema_manifest=mock.DEFAULT,
                _install_callable_probes=mock.DEFAULT,
                _register_hooks=mock.DEFAULT, _INSTALLED_CLASSES=set()), \
                mock.patch.object(sys, "stderr", io.StringIO()):
            capture.install_on_model(model)
            batch = types.SimpleNamespace(forward_mode="DECODE")
            self.assertEqual(model.forward(1, 2, batch), "mono")
        logger.set_context.assert_called_once()
        logger.mark_forward.assert_called_once()
