import os
import sys
import unittest


SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCRIPTS)
import semantic_kernel_mapping as mapping


def _frame(name, ts, dur, tid=7):
    return {"cat": "python_function", "name": name, "ts": ts, "dur": dur,
            "tid": tid}


def _launch(corr, ts, tid=7):
    return {"cat": "cuda_runtime", "name": "hipModuleLaunchKernel",
            "ts": ts, "tid": tid, "args": {"correlation": corr}}


def _kernel(name, corr, ts=100, dur=10, ext=None):
    args = {"correlation": corr, "stream": 1}
    if ext is not None:
        args["External id"] = ext
    return {"cat": "kernel", "name": name, "ts": ts, "dur": dur, "args": args}


# A Triton launch as the profiler records it: model code, then the launcher
# trampolines, then the built-in. No cpu_op anywhere on the path.
TRITON_STACK = [
    _frame("fp8.py(731): apply", 0, 100),
    _frame("fp8_utils.py(760): aiter_w8a8_block_fp8_linear", 5, 90),
    _frame("gemm_a8w8_blockscale.py(19): gemm_a8w8_blockscale", 10, 80),
    _frame("triton/runtime/jit.py(574): run", 20, 60),
    _frame("<built-in function launch>", 30, 40),
]


class PythonLaunchSiteTest(unittest.TestCase):
    def test_innermost_caller_frame_is_the_launch_site(self):
        events = TRITON_STACK + [_launch(42, 50)]
        sites = mapping._python_launch_sites(events)
        site = sites[42]
        self.assertEqual(
            site["name"], "gemm_a8w8_blockscale.py(19): gemm_a8w8_blockscale")

    def test_jit_and_dispatcher_frames_are_never_the_answer(self):
        # aiter/jit/core.py is the same wrapper for every aiter kernel, so
        # reporting it would name the trampoline rather than the caller.
        events = [
            _frame("forward_mla.py(394): forward_absorb_core", 0, 100),
            _frame("aiter/jit/core.py(1396): wrapper", 10, 80),
            _frame("torch/_ops.py(839): __call__", 20, 60),
            _launch(7, 50),
        ]
        sites = mapping._python_launch_sites(events)
        self.assertEqual(
            sites[7]["name"], "forward_mla.py(394): forward_absorb_core")

    def test_aiter_guard_and_custom_wrapper_chain_is_seen_through(self):
        # Real stack shape for a fused aiter collective: three trampolines sit
        # between the launch and the call worth reporting.
        events = [
            _frame("parallel_state.py(652): fused_allreduce_rmsnorm", 0, 100),
            _frame("custom_all_reduce.py(675): fused_ar_rms", 5, 90),
            _frame("torch_guard.py(202): wrapper", 10, 80),
            _frame("aiter/jit/core.py(1637): custom_wrapper", 15, 70),
            _frame("<built-in method fused_allreduce_rmsnorm>", 20, 60),
            _launch(11, 50),
        ]
        self.assertEqual(
            mapping._python_launch_sites(events)[11]["name"],
            "custom_all_reduce.py(675): fused_ar_rms")

    def test_graph_replay_yields_nothing_rather_than_the_replay_call(self):
        # Under graphs no Python runs at replay, so the only enclosing frame is
        # the replay itself. Returning it would block two-trace mapping, which
        # is the route that does work here.
        events = [
            _frame("model_runner.py(3373): _forward_raw", 0, 100),
            _frame("cuda_graph_runner.py(1317): replay", 10, 80),
            _launch(9, 50),
        ]
        self.assertNotIn(9, mapping._python_launch_sites(events))

    def test_unknown_correlation_and_missing_frames_resolve_to_none(self):
        sites = mapping._python_launch_sites(TRITON_STACK + [_launch(1, 50)])
        self.assertNotIn(999, sites)
        self.assertNotIn(1, mapping._python_launch_sites([_launch(1, 50)]))

    def test_frame_that_ended_before_the_launch_does_not_enclose(self):
        events = [
            _frame("earlier.py(1): done_before", 0, 10),
            _frame("real.py(2): caller", 20, 100),
            _launch(3, 50),
        ]
        self.assertEqual(
            mapping._python_launch_sites(events)[3]["name"],
            "real.py(2): caller")

    def test_frames_on_another_thread_are_not_borrowed(self):
        events = [
            _frame("other_thread.py(1): caller", 0, 100, tid=99),
            _launch(4, 50, tid=7),
        ]
        self.assertNotIn(4, mapping._python_launch_sites(events))


class PythonStackRowTest(unittest.TestCase):
    def test_triton_row_names_its_launcher_instead_of_going_unresolved(self):
        events = TRITON_STACK + [_launch(42, 50), _kernel("gemm_kernel", 42)]
        rows, _, _, _, _ = mapping._event_rows(events, {})
        parent = rows[0]["parent_operator"]
        self.assertEqual(parent["mapping_level"], "python_stack")
        self.assertEqual(
            parent["canonical_op"],
            "gemm_a8w8_blockscale.py(19): gemm_a8w8_blockscale")
        self.assertEqual(
            parent["python_launch_site"], parent["canonical_op"])

    def test_dispatched_kernel_keeps_its_cpu_op_parent(self):
        # The fallback is additive: a kernel that reaches a cpu_op through
        # External id must be unaffected by anything on the Python stack.
        events = TRITON_STACK + [
            _launch(42, 50),
            {"cat": "cpu_op", "name": "aiter::add_rmsnorm", "ts": 40,
             "dur": 30, "args": {"External id": 5, "Input Dims": [[2, 4]],
                                 "Input type": ["Half"]}},
            _kernel("add_rmsnorm_kernel", 42, ts=45, ext=5),
        ]
        rows, _, _, _, _ = mapping._event_rows(events, {})
        parent = rows[0]["parent_operator"]
        self.assertEqual(parent["mapping_level"], "external_id")
        self.assertEqual(parent["canonical_op"], "aiter::add_rmsnorm")
        self.assertIsNone(parent["python_launch_site"])

    def test_graph_replayed_kernel_stays_unresolved_for_two_trace(self):
        events = [
            _frame("cuda_graph_runner.py(1317): replay", 10, 80),
            _launch(9, 50),
            _kernel("moe_gemm", 9, ts=55),
        ]
        rows, _, _, _, _ = mapping._event_rows(events, {})
        self.assertEqual(
            rows[0]["parent_operator"]["mapping_level"], "unresolved")


if __name__ == "__main__":
    unittest.main()
