import os
import sys
import types
import unittest


SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCRIPTS)
import semantic_runtime_capture as capture


class ResolveTargetTest(unittest.TestCase):
    """A launcher is not always a plain module-level function.

    Three of the shapes GEAK has to probe are unreachable with a single
    getattr on an imported module: a method (``CustomAllreduce.fused_ar_rms``),
    an autograd Function entry point (``ChunkGatedDeltaRuleFunction.apply``),
    and a native custom op (``aiter::moe_sorting_fwd``), whose torch.ops
    namespace is not a module at all.
    """

    def setUp(self):
        module = types.ModuleType("geak_fake_launchers")

        def launcher(x, weight):
            return x

        class Holder(object):
            def method(self, x):
                return x

            @staticmethod
            def entry(x):
                return x

        module.launcher = launcher
        module.Holder = Holder
        module.not_callable = 7
        sys.modules["geak_fake_launchers"] = module
        self.module = module
        self.addCleanup(sys.modules.pop, "geak_fake_launchers", None)

    def test_plain_module_attribute(self):
        holder, name, original = capture._resolve_target(
            "geak_fake_launchers:launcher")
        self.assertIs(holder, self.module)
        self.assertEqual(name, "launcher")
        self.assertIs(original, self.module.launcher)

    def test_dotted_attribute_resolves_to_the_class(self):
        holder, name, original = capture._resolve_target(
            "geak_fake_launchers:Holder.method")
        self.assertIs(holder, self.module.Holder)
        self.assertEqual(name, "method")
        self.assertTrue(callable(original))

    def test_dotted_attribute_is_patchable_on_the_class(self):
        holder, name, original = capture._resolve_target(
            "geak_fake_launchers:Holder.entry")
        sentinel = lambda x: "patched"
        setattr(holder, name, sentinel)
        self.addCleanup(setattr, holder, name, original)
        self.assertEqual(self.module.Holder.entry(1), "patched")

    def test_missing_separator_is_rejected(self):
        with self.assertRaises(RuntimeError):
            capture._resolve_target("geak_fake_launchers.launcher")

    def test_empty_attribute_is_rejected(self):
        with self.assertRaises(RuntimeError):
            capture._resolve_target("geak_fake_launchers:")

    def test_torch_ops_namespace_is_not_imported_as_a_module(self):
        """torch.ops.<ns> is a lazily-populated object, so importlib would
        raise; resolution must walk the attribute chain instead."""
        torch = types.ModuleType("torch")
        ops = types.SimpleNamespace()
        namespace = types.SimpleNamespace()
        namespace.some_op = lambda *a, **k: None
        ops.fake_ns = namespace
        torch.ops = ops
        sys.modules["torch"] = torch
        self.addCleanup(sys.modules.pop, "torch", None)

        holder, name, original = capture._resolve_target(
            "torch.ops.fake_ns:some_op")
        self.assertIs(holder, namespace)
        self.assertEqual(name, "some_op")
        self.assertTrue(callable(original))


class PositionalNameTest(unittest.TestCase):
    def test_names_come_from_the_signature(self):
        def launcher(x, weight, x_scale=None):
            return x

        capture._CALLABLE_SIGNATURES["t"] = __import__(
            "inspect").signature(launcher)
        self.addCleanup(capture._CALLABLE_SIGNATURES.pop, "t", None)
        self.assertEqual(
            capture._positional_names("t", (1, 2)), ["x", "weight"])

    def test_no_signature_yields_no_names(self):
        """A native op has no introspectable signature; shapes are still
        recorded, just with positional labels."""
        capture._CALLABLE_SIGNATURES["native"] = None
        self.addCleanup(capture._CALLABLE_SIGNATURES.pop, "native", None)
        self.assertEqual(capture._positional_names("native", (1, 2)), [])


if __name__ == "__main__":
    unittest.main()
