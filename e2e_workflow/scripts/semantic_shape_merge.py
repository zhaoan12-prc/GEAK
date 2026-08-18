#!/usr/bin/env python3
"""Merge representative-layer Shape evidence without changing Clean Trace rows."""
import argparse
import json
import os
import re


LEGACY_LOG_RE = re.compile(
    r"phase=(?P<phase>\S+).*?rank=(?P<rank>\d+).*?bs=(?P<bs>-?\d+)"
    r".*?toks=(?P<toks>-?\d+).*?layer=(?P<layer>-?\d+)"
    r".*?op_instance_id=(?P<op_instance_id>\S+)"
    r".*?op_name=(?P<op_name>\S+).*?op_type=(?P<op_type>\S+)"
    r".*?op_path=(?P<op_path>\S+).*?io=(?P<io>\S+)"
    r".*?tensor_path=(?P<tensor_path>\S+).*?arg_name=(?P<arg_name>\S+)"
    r".*?tensor_role=(?P<tensor_role>\S+).*?shape=\[(?P<shape>[^\]]*)\]"
    r".*?dtype=(?P<dtype>\S+).*?device=(?P<device>\S+)"
    r".*?stride=\[(?P<stride>[^\]]*)\]")


# Routes that establish which call launched a kernel, as opposed to which
# module encloses it. The shape-logger probe knows only the latter, so when one
# of these already holds the op, the probe contributes its shape and nothing
# else.
OP_LEVEL_SOURCES = {
    "external_id": "clean_trace_external_id",
    "python_stack": "python_stack_launch_site",
    "two_trace_external_id": "two_trace_graph_off_external_id",
    "two_trace_python_stack": "two_trace_graph_off_python_stack",
}
OP_LEVEL_ORIGINS = {
    "external_id": "clean_trace",
    "python_stack": "python_stack",
    "two_trace_external_id": "two_trace_mapping",
    "two_trace_python_stack": "two_trace_mapping",
}
_STACK_BASIS = ("innermost enclosing python_function frame of the launching "
                "thread, via correlation")
OP_LEVEL_BASIS = {
    "external_id": "cpu_op reached by External id in the Clean Trace",
    "python_stack": _STACK_BASIS,
    "two_trace_external_id": (
        "cpu_op of the positionally bound row in the graph-off mapping trace"),
    "two_trace_python_stack": (
        "%s, in the graph-off mapping trace" % _STACK_BASIS),
}
KERNEL_SCOPE_OP_LEVELS = tuple(OP_LEVEL_SOURCES)


def _load(path):
    with open(path) as fh:
        return json.load(fh)


def _dims(value):
    if isinstance(value, list):
        return [int(item) for item in value]
    return [
        int(item) for item in re.split(r"[x,]", str(value or ""))
        if item.strip()]


def _shape_records(path):
    records = []
    if not path or not os.path.exists(path):
        return records
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except ValueError:
                match = LEGACY_LOG_RE.search(line)
                if not match:
                    continue
                record = match.groupdict()
                for key in ("rank", "bs", "toks", "layer"):
                    record[key] = int(record[key])
                record["shape"] = _dims(record.get("shape"))
                record["stride"] = _dims(record.get("stride"))
            records.append(record)
    return records


def _metadata_tensors(meta, io, path=""):
    if not isinstance(meta, dict):
        return []
    if meta.get("kind") == "tensor":
        return [{
            "io": io,
            "tensor_path": path or io,
            "arg_name": path or io,
            "tensor_role": io,
            "shape": _dims(meta.get("shape")),
            "dtype": meta.get("dtype"),
            "device": meta.get("device"),
            "stride": _dims(meta.get("stride")),
            "contiguous": meta.get("contiguous"),
            "alias_id": meta.get("alias_id"),
        }]
    values = meta.get("items")
    result = []
    if isinstance(values, list):
        for index, item in enumerate(values):
            result.extend(_metadata_tensors(
                item, io, "%s[%d]" % (path, index) if path else "[%d]" % index))
    elif isinstance(values, dict):
        for key, item in values.items():
            result.extend(_metadata_tensors(
                item, io, "%s.%s" % (path, key) if path else str(key)))
    return result


def _bucket_fields(record):
    phase = str(record.get("phase", "")).lower()
    batch_size = record.get("batch_size", record.get("bs", -1))
    input_tokens = record.get("input_tokens", record.get("toks", -1))
    parts = str(record.get("bucket") or "").split(":")
    if parts and parts[0]:
        phase = phase or parts[0].lower()
    phase = {"extend": "prefill", "prompt": "prefill",
             "generation": "decode"}.get(phase, phase)
    if len(parts) >= 2 and parts[1].lstrip("-").isdigit():
        batch_size = int(parts[1])
    if len(parts) >= 3 and parts[2].lstrip("-").isdigit():
        input_tokens = int(parts[2])
    return phase, int(batch_size or -1), int(input_tokens or -1)


def _record_layer_id(record):
    # NOTE: use `is not None`, not `or`, so layer_id 0 (the first decoder layer)
    # is kept as 0 and not coerced to the -1 "global op" sentinel.
    for key in ("layer_id", "layer"):
        value = record.get(key)
        if value is not None:
            return int(value)
    return -1


def _groups(records):
    grouped = {}
    for record in records:
        oid = record.get("op_instance_id") or record.get("oid")
        if not oid:
            continue
        phase, batch_size, input_tokens = _bucket_fields(record)
        group = grouped.setdefault(oid, {
            "op_instance_id": oid,
            "phase": phase,
            "rank": int(record.get("rank", 0) or 0),
            "layer_id": _record_layer_id(record),
            "batch_size": batch_size,
            "input_tokens": input_tokens,
            "op_name": record.get(
                "op_name", record.get("target_op", "")),
            "op_type": record.get(
                "op_type", record.get("target_op", "")),
            "op_path": record.get("op_path", ""),
            "tensors": [],
            "mapping_cardinality": record.get("mapping_cardinality"),
            "capture_window": record.get("capture_window"),
            "static_context": record.get("static_context"),
        })
        if str(record.get("schema", "")).startswith("geak.semantics_"):
            group["tensors"].extend(
                _metadata_tensors(record.get("inputs"), "input", "args"))
            group["tensors"].extend(
                _metadata_tensors(record.get("kwargs"), "input", "kwargs"))
            group["tensors"].extend(
                _metadata_tensors(
                    record.get("parameters"), "weight", "parameters"))
            group["tensors"].extend(
                _metadata_tensors(record.get("output"), "output", "output"))
            _name_launcher_arguments(group, record)
        else:
            group["tensors"].append({
                "io": record.get("io", "raw_arg"),
                "tensor_path": record.get(
                    "tensor_path", record.get("tpath", "")),
                "arg_name": record.get("arg_name", record.get("arg", "")),
                "tensor_role": record.get(
                    "tensor_role", record.get("role", "raw_arg")),
                "shape": _dims(record.get("shape")),
                "dtype": record.get("dtype"),
                "device": record.get("device"),
                "stride": _dims(record.get("stride")),
            })
    return list(grouped.values())


def _normalize(value):
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


LAUNCHER_MARKER = "::launcher:"

# Calling conventions for wrapped launchers, keyed by a token in the callable
# name. ``names`` is only a fallback for shape logs captured before the probe
# recorded signature names; a recorded name always wins. ``io`` states which
# arguments are weights, which a positional list cannot express and which
# _tensor_schema needs before it can report A[M,K] x W[N,K] -> O[M,N].
_LAUNCHER_ARGUMENT_CONVENTIONS = (
    ("gemm_a8w8_blockscale", {
        "names": ("x", "weight", "x_scale", "w_scale"),
        "io": {"weight": "weight"},
    }),
    ("gemm_a8w8", {
        "names": ("x", "weight", "x_scale", "w_scale"),
        "io": {"weight": "weight"},
    }),
)


def _launcher_convention(symbol):
    lowered = str(symbol or "").lower()
    for token, convention in _LAUNCHER_ARGUMENT_CONVENTIONS:
        if token in lowered:
            return convention
    return None


def _name_launcher_arguments(group, record):
    """Replace positional ``args[i]`` labels with real parameter names.

    A launcher probe records a positional argument list, so without this the
    ledger reports args[0]/args[1]/args[2] and nothing states which operand is
    the activation, which is the weight, and which are the block scales. The
    names come from the wrapped callable's signature (recorded by the probe as
    ``arg_names``), falling back to a declared convention for shape logs
    captured before that existed.
    """
    symbol = _launcher_symbol(group)
    if not symbol:
        return
    recorded = record.get("arg_names") or []
    convention = _launcher_convention(symbol) or {}
    fallback = convention.get("names") or ()
    io_roles = convention.get("io") or {}
    for tensor in group["tensors"]:
        match = re.match(r"^args\[(\d+)\]$", str(tensor.get("tensor_path")))
        if not match:
            continue
        index = int(match.group(1))
        name = ""
        if index < len(recorded):
            name = str(recorded[index])
        elif index < len(fallback):
            name = str(fallback[index])
        if not name or name.startswith("args["):
            continue
        tensor["arg_name"] = name
        tensor["tensor_path"] = name
        tensor["parameter_name"] = name
        if name in io_roles:
            # Direction is unchanged -- a weight is still an input. This only
            # records that the operand is a parameter rather than activation.
            tensor["io"] = io_roles[name]
            tensor["tensor_role"] = io_roles[name]


def _launcher_symbol(group):
    """Terminal callable name of a targeted-launcher probe group, else "".

    ``semantic_runtime_capture.begin_callable`` records a launcher probe with
    ``op_path = "<enclosing module path>::launcher:<module>:<attr>"``.  Module
    forward hooks carry no marker, so this also distinguishes the two kinds of
    record.
    """
    path = str(group.get("op_path") or "")
    marker = path.find(LAUNCHER_MARKER)
    if marker < 0:
        return ""
    return path[marker + len(LAUNCHER_MARKER):].rpartition(":")[2].strip()


def _operator_symbol(text):
    """Terminal function name of a capture-plan operator string.

    Handles both python-stack frames such as
    ``aiter/ops/triton/gemm/basic/gemm_a8w8_blockscale.py(19):
    gemm_a8w8_blockscale`` and native op names such as
    ``aiter::moe_sorting_fwd``.
    """
    return str(text or "").strip().rpartition(":")[2].strip()


def _symbols_match(plan_symbol, launcher_symbol):
    """Compare a plan operator symbol with a probe's launcher symbol.

    Substring either way, because the binding a caller invokes is frequently an
    alias of the defining name (``gemm_a8w8_blockscale`` imported into
    ``fp8_utils`` as ``triton_gemm_a8w8_blockscale``).
    """
    left = _normalize(plan_symbol)
    right = _normalize(launcher_symbol)
    # Substring matching on a very short name would pair unrelated operators,
    # and several of the candidate fields hold module paths rather than
    # function names, so require a distinctive symbol on both sides.
    if len(left) < 6 or len(right) < 6:
        return False
    return left in right or right in left


def _plan_operator(target, row):
    """The operator string the capture plan (or the row) attributes this to."""
    explicit = target.get("candidate_op_path") or target.get(
        "candidate_wrapper")
    if not explicit and target.get("parent_operator") != "unresolved":
        explicit = target.get("parent_operator")
    if not explicit:
        parent = row.get("parent_operator")
        explicit = (
            parent.get("canonical_op") if isinstance(parent, dict) else parent)
    return explicit or ""


def _callable_operators(target, row):
    """Every string that might name the callable this row launched.

    Matching a launcher probe needs a *function* name.  Several of these
    fields hold a module path instead -- ``semantic_runtime_marker_mapping``
    fills ``candidate_wrapper`` with the enclosing ``model.layers.N`` marker --
    so all plausible sources are offered and the symbol comparison decides.
    """
    parent = row.get("parent_operator")
    if not isinstance(parent, dict):
        parent = {"canonical_op": parent}
    ordered = [
        target.get("candidate_terminal_launcher"),
        parent.get("python_launch_site"),
        parent.get("canonical_op"),
        target.get("parent_operator"),
        target.get("candidate_op_path"),
        target.get("candidate_wrapper"),
    ]
    return [
        str(value) for value in ordered
        if value and str(value) != "unresolved"]


def _launcher_bindings(rows, groups, target_by_row, table):
    """Bind rows to targeted-launcher probe records, 1:1 and in captured order.

    A launcher probe wrapped the real kernel launcher and recorded that call's
    own operands, so it is strictly stronger evidence than the enclosing
    module's tensors -- for a blockscale GEMM it carries the true FP8 A/B and
    hence M/N/K, where the module hook only sees the layer's bf16 hidden state.

    Binding is deliberately conservative: within one
    ``(phase, representative layer, callable symbol)`` bucket the number of
    candidate rows must equal the number of probe records, and they are then
    paired in captured order.  Any other count is left unbound so the existing
    ordered wrapper alignment still applies unchanged.  This keeps the mapping
    deterministic and auditable rather than guessing which launch is which.
    """
    phase = str(table["phase"]).lower()
    layer_id = int(table["representative_layer_id"])
    launchers = [
        group for group in groups
        if int(group.get("rank", 0) or 0) == 0
        and group.get("phase") == phase
        and group.get("layer_id") == layer_id
        and _launcher_symbol(group)]
    if not launchers:
        return {}

    # The replay captures several forward passes, so the same launcher fires
    # once per pass from the same enclosing module. The representative-layer
    # table holds one row per launch, not per pass, so collapse to the first
    # record of each enclosing module and keep capture order.
    by_symbol = {}
    for group in launchers:
        symbol = _launcher_symbol(group)
        enclosing = str(group.get("op_path") or "").split(LAUNCHER_MARKER)[0]
        seen = by_symbol.setdefault(symbol, {})
        seen.setdefault(enclosing, group)
    by_symbol = {
        symbol: list(seen.values()) for symbol, seen in by_symbol.items()}

    bound = {}
    for symbol, symbol_groups in by_symbol.items():
        matched = [
            row for row in rows
            # A kernel_exact row is already resolved from the Clean Trace and
            # never consults probe groups, so counting it here would only
            # skew the 1:1 check and block an otherwise valid binding.
            if row.get("shape", {}).get("source") != "kernel_exact"
            and any(
                _symbols_match(_operator_symbol(operator), symbol)
                for operator in _callable_operators(
                    target_by_row.get(row["row_id"], {}), row))]
        if matched and len(matched) == len(symbol_groups):
            for row, group in zip(matched, symbol_groups):
                bound[row["row_id"]] = group
    return bound


def _candidate_groups(row, target, groups, table, two_trace_entry=None):
    phase = str(table["phase"]).lower()
    layer_id = int(table["representative_layer_id"])
    candidates = [
        group for group in groups
        if group["rank"] == 0 and group["phase"] == phase
        and group["layer_id"] == layer_id]
    op_instance_id = target.get("candidate_op_instance_id")
    if op_instance_id:
        return [
            group for group in candidates
            if group.get("op_instance_id") == op_instance_id]
    explicit = target.get("candidate_op_path") or target.get(
        "candidate_wrapper")
    if not explicit and target.get("parent_operator") != "unresolved":
        explicit = target.get("parent_operator")
    if not explicit and two_trace_entry:
        # Position-bound identity beats a bare kernel name: the two-trace map
        # already resolved WHICH launch this row is, so use its operator as the
        # wrapper hint instead of falling back to name-only text matching.
        recovered = (two_trace_entry.get("op") or {})
        if recovered.get("recovered"):
            explicit = recovered.get("canonical_op")
    if not explicit:
        return []
    needle = _normalize(explicit)
    return [
        group for group in candidates
        if needle and any(
            needle in _normalize(group.get(key))
            or _normalize(group.get(key)) in needle
            for key in ("op_path", "op_name", "op_type")
            if group.get(key))]


def _op_stage_match(group, row):
    text = "%s %s %s" % (
        group.get("op_name", ""), group.get("op_type", ""),
        group.get("op_path", ""))
    text = text.lower()
    stage = row.get("stage")
    rules = {
        "norm": r"norm",
        "rope": r"rope|rotary",
        "attn": r"attention|attn|mha|mla",
        "linear_attn": r"linear.?attention|gated.?delta|conv1d|recurrent",
        "gemm": r"linear|gemm|matmul|projection|proj",
        "topk": r"top.?k|router|gate",
        "moe": r"moe|expert",
        "activation": r"activation|silu|gelu|swiglu",
        "quant": r"quant|fp8|scale",
        "communication": r"all.?reduce|communicat|collective",
    }
    return bool(re.search(rules.get(stage, r"(?!x)x"), text))


def _leaf_groups(groups):
    paths = [str(group.get("op_path") or "") for group in groups]
    leaves = []
    for group in groups:
        path = str(group.get("op_path") or "")
        if not any(
                other.startswith(path + ".")
                for other in paths if path and other != path):
            leaves.append(group)
    return leaves


def _align_rows_to_groups(rows, groups):
    """Ordered wrapper alignment; never upgrades wrapper context to Kernel exact."""
    if not groups:
        return [None] * len(rows)
    aligned = []
    index = 0
    anchor_seen = [False] * len(groups)
    for row in rows:
        while (index + 1 < len(groups)
               and _op_stage_match(groups[index + 1], row)
               and (anchor_seen[index]
                    or not _op_stage_match(groups[index], row))):
            index += 1
        if _op_stage_match(groups[index], row):
            anchor_seen[index] = True
        aligned.append(groups[index] if anchor_seen[index] else None)
    return aligned


def _axis(value, role, source):
    return {"axis_role": role, "value": int(value), "source": source}


def _tensor_schema(group, row, table, exact_bucket):
    tensors = []
    trace_dims = row.get("shape", {}).get("input_dims") or []
    clean_bucket = table.get("selected_bucket") or {}
    clean_dynamic = (
        clean_bucket.get("input_tokens")
        or clean_bucket.get("batch_size"))
    logger_dynamic = {
        group.get("input_tokens"), group.get("batch_size")}
    for tensor in group["tensors"]:
        item = dict(tensor)
        item["axes"] = [
            _axis(value, "unresolved", "shape_logger")
            for value in tensor["shape"]]
        if (item["io"] in ("input", "output") and item["axes"]
                and item["shape"][0] in logger_dynamic and clean_dynamic):
            item["axes"][0] = _axis(
                clean_dynamic, "token_or_batch",
                "shape_logger" if exact_bucket else "clean_trace_step")
        item["effective_shape"] = [
            axis["value"] for axis in item["axes"]]
        tensors.append(item)

    inputs = [item for item in tensors if item["io"] == "input"]
    weights = [item for item in tensors if item["io"] == "weight"]
    outputs = [item for item in tensors if item["io"] == "output"]
    linear = None
    if inputs and weights and len(inputs[0]["shape"]) == 2 and len(
            weights[0]["shape"]) == 2:
        m_log, k_log = inputs[0]["shape"]
        n_log, wk_log = weights[0]["shape"]
        if k_log == wk_log:
            m_value = inputs[0]["axes"][0]["value"]
            m_source = inputs[0]["axes"][0]["source"]
            if (not exact_bucket and trace_dims and
                    isinstance(trace_dims[0], list) and
                    len(trace_dims[0]) == 2 and trace_dims[0][1] == k_log):
                m_value = trace_dims[0][0]
                m_source = "clean_trace"
            linear = {
                "interface": "A[M,K] x W[N,K] -> O[M,N]",
                "M": _axis(m_value, "token_or_batch", m_source),
                "K": _axis(k_log, "reduction", "weight_metadata"),
                "N": _axis(n_log, "output", "weight_metadata"),
                "validated": (
                    not outputs or len(outputs[0]["shape"]) != 2
                    or outputs[0]["shape"][1] == n_log),
            }
    return {"tensors": tensors, "linear_interface": linear}


def _bucket_status(table, group):
    clean = table.get("selected_bucket") or {}
    same_bs = clean.get("batch_size") in (None, group["batch_size"])
    same_tokens = clean.get("input_tokens") in (None, 0, group["input_tokens"])
    if same_bs and same_tokens:
        return "exact"
    return "compatible" if same_bs else "mismatch"


def _layer_tensor(tensor, table, group):
    if tensor is None:
        return None
    value = dict(tensor)
    value["logger_shape"] = list(tensor.get("shape") or [])
    value["effective_shape"] = list(value["logger_shape"])
    clean = table.get("selected_bucket") or {}
    dynamic = clean.get("input_tokens") or clean.get("batch_size")
    if (value["effective_shape"] and dynamic
            and value["effective_shape"][0] in {
                group.get("input_tokens"), group.get("batch_size")}):
        value["effective_shape"][0] = dynamic
        value["axis_0_source"] = (
            "shape_logger" if _bucket_status(table, group) == "exact"
            else "clean_trace_step")
    return value


def _dtype_label(value):
    text = str(value or "?").replace("c10::", "").replace("torch.", "")
    lowered = text.lower()
    if "float8" in lowered or lowered in ("fp8", "e4m3", "e4m3fnuz"):
        return "FP8"
    if "bfloat16" in lowered or lowered == "bf16":
        return "BF16"
    if lowered in ("half", "float16", "fp16"):
        return "FP16"
    if lowered in ("float", "float32", "fp32"):
        return "FP32"
    if lowered in ("double", "float64", "fp64"):
        return "FP64"
    if lowered in ("long", "long int", "int64", "int64_t"):
        return "INT64"
    if lowered in ("int", "int32", "int32_t"):
        return "INT32"
    if lowered in ("short", "int16", "int16_t"):
        return "INT16"
    if lowered in ("char", "int8", "int8_t"):
        return "INT8"
    if lowered in ("bool", "boolean"):
        return "BOOL"
    return text.upper()


# Per-operator argument roles. These encode the real calling convention,
# including the AITER/HIP habit of passing destination buffers as leading
# positional arguments -- which is exactly what a positional guess gets wrong.
# A token match here is the only evidence that an argument's input/output role
# is actually *known* rather than assumed.
_OPERATOR_ROLE_MAPS = (
    ("dynamic_per_token_scaled_quant", (
        "y", "x", "scale")),
    ("add_rmsnorm", (
        "y", "residual", "x", "residual_out", "weight")),
    ("::rmsnorm", ("y", "x", "weight")),
    ("::copy_", ("dst", "src")),
    ("rope_cached_positions", (
        "q", "k", "q_out", "k_out", "q_cache", "k_cache",
        "positions")),
    ("fmha", ("q", "k", "v", "cu_seqlens_q", "cu_seqlens_k")),
    ("mha_batch_prefill", (
        "q", "k", "v", "cu_seqlens_q", "cu_seqlens_k",
        "block_table")),
    ("store_cache", (
        "k", "v", "k_cache_out", "v_cache_out", "slot_mapping")),
    ("qr_all_reduce", ("workspace", "x", "y")),
    ("silu_and_mul", ("y", "x")),
    ("::sigmoid", ("x",)),
    ("::mul", ("x", "other")),
    ("::gt", ("x", "other")),
    ("::fill_", ("dst",)),
    ("::mm", ("x", "weight")),
    ("::bmm", ("x", "other")),
    ("::neg", ("x",)),
    ("grouped_topk", (
        "logits", "bias", "topk_weights", "topk_ids")),
    ("fmoe", (
        "x", "y", "w13", "w2", "sorted_ids", "sorted_weights",
        "sorted_expert_ids", "num_valid_ids", "workspace",
        "x_scale", "w13_scale", "w2_scale")),
    ("_index_put_impl_", ("cache", "indices", "x")),
    ("::arange", ("start", "end", "step", "y")),
    # AITER MoE ops take their destination buffers as *leading* positional
    # arguments, so the direction cannot be read off the position and a role
    # name alone would put them in the wrong column. Each entry below was
    # checked argument by argument against the recorded dtypes and extents in
    # this run's traces, and carries the destination indices explicitly.
    # Both shape paths index by *declared parameter position*: `Input Dims`
    # keeps an empty slot for a scalar argument and the probe records
    # `args[<n>]`, so these tuples list every parameter, scalars included, in
    # signature order. Naming a scalar slot costs nothing -- it is skipped when
    # rendered -- and keeps the tensor positions honest.
    # aiter/ops/moe_op.py::topk_softmax
    ("topk_softmax", (
        "topk_weights", "topk_ids", "token_expert_indices", "logits",
        "need_renorm", "num_shared_experts", "shared_expert_scoring_func"),
     (0, 1, 2)),
    # aiter/ops/moe_sorting.py::moe_sorting_fwd
    ("moe_sorting_fwd", (
        "topk_ids", "topk_weights", "sorted_token_ids", "sorted_weights",
        "sorted_expert_ids", "num_valid_ids", "moe_buf", "num_experts",
        "unit_size", "local_expert_mask", "num_local_tokens",
        "dispatch_policy"),
     (2, 3, 4, 5, 6)),
    # aiter/ops/moe_op.py::ck_moe_stage1
    ("ck_moe_stage1", (
        "x", "w13", "w2", "sorted_token_ids", "sorted_expert_ids",
        "num_valid_ids", "y", "topk", "kernelName", "w13_scale", "x_scale",
        "block_m", "sorted_weights", "quant_type", "activation", "splitk",
        "use_non_temporal_load", "dst_type"),
     (6,)),
    # aiter/ops/moe_op.py::ck_moe_stage2
    ("ck_moe_stage2", (
        "x", "w13", "w2", "sorted_token_ids", "sorted_expert_ids",
        "num_valid_ids", "y", "topk", "kernelName", "w2_scale", "x_scale",
        "block_m", "sorted_weights", "quant_type", "activation", "splitk",
        "use_non_temporal_load", "dst_type", "is_shuffled"),
     (6,)),
)


def _operator_roles(row):
    """Roles for this row's operator, or None when the operator is unknown.

    None means "no calling-convention knowledge". Callers must not invent an
    input/output split in that case.
    """
    roles, _ = _operator_convention(row)
    return roles


def _operator_convention(row):
    """(roles, destination indices) for this row's operator, or (None, ()).

    An entry may declare which positional arguments the operator *writes*.
    That is calling-convention evidence, not a guess from the role name, and it
    is what lets an out-param passed as argument 0 be reported as an output
    without any role name having to imply direction.
    """
    op = str(
        (row.get("parent_operator") or {}).get("canonical_op", "")).lower()
    if not op:
        return None, ()
    for entry in _OPERATOR_ROLE_MAPS:
        token, roles = entry[0], entry[1]
        if token in op:
            return roles, (entry[2] if len(entry) > 2 else ())
    return None, ()


def _trace_role(row, index):
    stage = str(row.get("stage") or "").lower()
    roles = _operator_roles(row)
    if roles is not None and index < len(roles):
        return roles[index]
    defaults = {
        "gemm": ("x", "weight", "bias", "x_scale", "weight_scale"),
        "quant": ("y", "x", "scale"),
        "norm": ("y", "x", "weight", "bias"),
        "attn": ("q", "k", "v", "cu_seqlens_q", "cu_seqlens_k"),
        "linear_attn": ("q", "k", "v", "state"),
        "activation": ("y", "x"),
        "communication": ("x", "y"),
        "topk": ("logits", "bias", "topk_weights", "topk_ids"),
        "moe": ("x", "y", "w13", "w2"),
        "memory": ("dst", "src"),
    }
    roles = defaults.get(stage, ())
    return roles[index] if index < len(roles) else "tensor_%d" % index


def _path_name(tensor):
    value = str(
        tensor.get("arg_name") or tensor.get("tensor_path") or "")
    value = value.replace("parameters.", "").replace("param.", "")
    match = re.search(r"(?:kwargs|args)\.([A-Za-z_]\w*)$", value)
    if match:
        return match.group(1)
    if re.match(r"^(?:args|input|output)(?:\[\d+\])?$", value):
        return ""
    return value.rsplit(".", 1)[-1].replace("[", "_").replace("]", "")


def _probe_role(row, tensor, index, output_index):
    io = str(tensor.get("io") or "input").lower()
    path_name = _path_name(tensor)
    semantic_names = {
        "input": "x", "hidden_states": "x", "x": "x", "X": "x",
        "XQ": "x", "WQ": "weight", "w": "weight", "weight": "weight",
        "x_scale": "x_scale", "input_scale": "x_scale",
        "w_scale": "weight_scale", "weight_scale": "weight_scale",
        "q": "q", "query": "q", "k": "k", "key": "k",
        "v": "v", "value": "v", "mixed_qkv": "mixed_qkv",
        "initial_state": "state", "out": "y",
    }
    if path_name in semantic_names:
        return semantic_names[path_name]
    if io == "output":
        if str(row.get("stage") or "").lower() == "quant":
            return "y" if output_index == 0 else "scale"
        return "y" if output_index == 0 else "y_%d" % output_index
    if io in ("weight", "parameter"):
        return path_name or ("weight" if index == 0 else "weight_%d" % index)
    stage = str(row.get("stage") or "").lower()
    defaults = {
        "gemm": ("x", "weight", "x_scale", "weight_scale", "bias"),
        "quant": ("x", "scale"),
        "norm": ("x", "residual", "weight", "bias"),
        "attn": ("q", "k", "v", "scale", "state"),
        "linear_attn": ("q", "k", "v", "a", "b", "state"),
        "activation": ("x",),
        "communication": ("x",),
        "topk": ("logits", "bias"),
        "moe": ("x", "topk_weights", "topk_ids"),
        "memory": ("src", "dst"),
        "elementwise": ("x", "other"),
    }
    roles = defaults.get(stage, ("x",))
    return path_name or (
        roles[index] if index < len(roles) else "input_%d" % index)


def _layer_wrapper_role(tensor, index, output_index):
    io = str(tensor.get("io") or "input").lower()
    path_name = _path_name(tensor)
    if path_name:
        return path_name
    if io == "output":
        return "wrapper_output_%d" % output_index
    if io in ("weight", "parameter"):
        return "wrapper_weight_%d" % index
    return "wrapper_input_%d" % index


def _is_output_role(role):
    return (
        role == "y" or role.startswith("y_") or role == "dst"
        or role.startswith("wrapper_output_")
        or role.endswith("_out") or role in (
            "topk_weights", "topk_ids", "scale"))


def _schema_declares_io(tensors):
    """True when the probe actually labelled an output.

    Shape rows recovered from a cpu_op's ``Input Dims`` carry every positional
    argument as ``io="input"`` -- the profiler cannot say which one is an
    out-param. Such a schema carries no io evidence, so the probe role table
    (which assumes argument 0 is the input) must not be applied to it.
    """
    return any(
        str(tensor.get("io") or "").lower() == "output"
        for tensor in tensors)


def _row_operator_symbol(row):
    """The source symbol this row's operator is named after, if any.

    Trace operator names are `aiter::moe_sorting_fwd`, a bare
    `ChunkGatedDeltaRuleFunction`, or a `path.py(123): symbol` launch site --
    all of which end in the symbol defined in the runtime source, which is what
    `_operator_symbol` already extracts.
    """
    op = str((row.get("parent_operator") or {}).get("canonical_op") or "")
    if not op or op == "unresolved":
        return ""
    return _operator_symbol(op)


def _attach_source_parameter_names(row, parameter_symbols):
    """Record this row's operator parameter names read from runtime source.

    Names only. The source says what argument 3 is *called*, not whether the
    operator writes to it, so this never sets a direction -- an entry in
    `_OPERATOR_ROLE_MAPS` remains the only thing that can.
    """
    if not parameter_symbols:
        return
    entry = parameter_symbols.get(_row_operator_symbol(row))
    if not entry or not entry.get("parameters"):
        return
    # Recorded on the row, not inside `shape`: the merge replaces a row's whole
    # shape dict when it rebinds one (two-trace dims, probe dims), which would
    # drop this with it.
    row["source_parameter_names"] = {
        "parameters": list(entry["parameters"]),
        "source": entry.get("source"),
        "line": entry.get("line"),
        "evidence": "runtime_source_signature",
    }


def _source_parameter_role(row, index):
    """Parameter name at this argument position, from the runtime source."""
    names = (row.get("source_parameter_names") or {}).get("parameters") or []
    return names[index] if index < len(names) else ""


def _positional_role(tensor, index, row=None):
    """Name an argument without claiming to know its direction."""
    if row is not None:
        source_name = _source_parameter_role(row, index)
        if source_name:
            return source_name
    return _path_name(tensor) or "arg%d" % index


def _argument_index(tensor, fallback):
    """The argument position this tensor actually occupies.

    A probe's tensor list can be a subset of the call's arguments, so its list
    position is not the argument position. ``arg_name`` records the real one
    (``args[6]``); an operator role table is indexed by that, never by where
    the tensor happened to land in the list.
    """
    match = re.search(
        r"args\[(\d+)\]$", str(tensor.get("arg_name") or ""))
    return int(match.group(1)) if match else fallback


def _semantic_shape_text(prefix, tensors):
    """Render tensors, splitting inputs from outputs.

    Items are ``(role, text)`` -- direction inferred from the role name -- or
    ``(role, text, is_output)`` when the direction is actually recorded. A
    recorded direction always wins: a probe that labelled an argument an input
    must not be overridden because a positional role table happened to name it
    `scale` or `topk_weights`.
    """
    def is_output(item):
        return item[2] if len(item) > 2 else _is_output_role(item[0])

    inputs = [item[1] for item in tensors if not is_output(item)]
    outputs = [item[1] for item in tensors if is_output(item)]
    values = inputs + outputs
    if not values:
        return prefix + ": scalar/no tensor shape"
    separator = "<br>"
    if inputs and outputs:
        return (
            prefix + ": " + separator.join(inputs)
            + "<br><br>" + separator.join(outputs))
    return prefix + ": " + separator.join(values)


def _shape_text(row):
    shape = row.get("shape", {})
    evidence = row.get("semantic_evidence", {})
    level = evidence.get("level", "U")
    if level == "K":
        dims = shape.get("input_dims") or []
        types = shape.get("input_types") or []
        # `Input Dims` is a positional argument list, not an input list. Only a
        # known calling convention can say which entries are destinations.
        roles, destinations = _operator_convention(row)
        known = roles is not None
        tensors = []
        for index, dim in enumerate(dims):
            if not isinstance(dim, list) or not dim:
                continue
            role = (_trace_role(row, index) if known
                    else _source_parameter_role(row, index) or "arg%d" % index)
            dtype = types[index] if index < len(types) else "Tensor"
            text = "%s=%s[%s]" % (
                role, _dtype_label(dtype),
                "×".join(str(value) for value in dim))
            if destinations:
                tensors.append((role, text, index in destinations))
            elif known:
                tensors.append((role, text))
            else:
                # Named from source, or not named at all. Either way no
                # direction is known, so state none: without this an argument
                # the source happens to call `scale` would be moved into the
                # output column by the role-name heuristic alone.
                tensors.append((role, text, False))
        return _semantic_shape_text("K", tensors)
    schema = shape.get("logger_schema") or {}
    tensors = schema.get("tensors") or []
    if level == "P" and tensors:
        values = []
        output_index = 0
        layer_wrapper = (
            evidence.get("wrapper_scope") == "phase_layer_wrapper")
        # Pick the role resolver by what the evidence actually supports:
        #   layer wrapper  -> wrapper I/O, io labels are real
        #   schema declares an output -> a probe labelled io, trust _probe_role
        #   operator known -> positional args, but the calling convention is
        #                     known, so the K-path operator roles apply
        #   otherwise      -> name positionally and claim no direction
        schema_io = _schema_declares_io(tensors)
        operator_roles, operator_destinations = _operator_convention(row)
        operator_known = operator_roles is not None
        for index, tensor in enumerate(tensors[:12]):
            dims = tensor.get("effective_shape") or tensor.get("shape") or []
            if layer_wrapper:
                role = _layer_wrapper_role(tensor, index, output_index)
            elif schema_io:
                role = _probe_role(row, tensor, index, output_index)
            elif operator_known:
                role = _trace_role(row, _argument_index(tensor, index))
            else:
                role = _positional_role(
                    tensor, _argument_index(tensor, index), row)
            tensor_is_output = (
                str(tensor.get("io") or "").lower() == "output")
            if tensor_is_output:
                output_index += 1
            entry = (role, "%s=%s[%s]" % (
                role, _dtype_label(tensor.get("dtype")),
                "×".join(str(value) for value in dims)))
            # layer_wrapper and schema_io rows have a recorded direction. An
            # operator whose entry declares its destinations has the direction
            # from its calling convention, which outranks a positional guess
            # for exactly the ops that pass an out-param first. Anything else
            # falls back to the role name.
            if layer_wrapper or schema_io:
                entry += (tensor_is_output,)
            elif operator_known and operator_destinations:
                entry += (
                    _argument_index(tensor, index) in operator_destinations,)
            elif not operator_known:
                # Source-derived or positional name: no direction evidence.
                entry += (False,)
            values.append(entry)
        if len(tensors) > 12:
            values.append((
                "metadata", "metadata=+%d tensors" % (len(tensors) - 12)))
        # This column is the shape, so it is labelled with the shape's scope.
        # A kernel-scope op whose dims came from an enclosing wrapper reads
        # P(wrapper) here and still shows its launch site in the operator
        # column; probe_scope is the op's and stays in the evidence record.
        scope = evidence.get("shape_scope") or evidence.get(
            "probe_scope", "wrapper")
        return _semantic_shape_text("P(%s)" % scope, values)
    if level == "P":
        # Attribution without shape: the Python stack names the launching call
        # but, unlike a cpu_op, carries no Input Dims.
        return "P(%s): op=%s, shape unavailable" % (
            evidence.get("probe_scope", "wrapper"),
            evidence.get("contained_by") or "unresolved")
    reason_code = evidence.get("reason_code", "unavailable")
    reason = evidence.get("reason", "shape unavailable")
    return "U(%s): %s" % (reason_code, reason)


_UNBOUND_CAUSE_REASONS = {
    "absent_from_mapping_replay": (
        "kernel_absent_from_mapping_replay",
        "the graph-off mapping replay never launched this kernel, so no probe "
        "can observe it: the graph-on and graph-off runs selected different "
        "implementations for this operation"),
    "launched_outside_every_marker": (
        "kernel_launched_outside_probe_scope",
        "the mapping replay launched this kernel outside every GEAK wrapper "
        "marker, so the launching module was not instrumented"),
    "launch_count_differs": (
        "probe_launch_count_differs",
        "the mapping replay launched this kernel a different number of times "
        "than the clean trace inside the selected forward"),
}


def _unavailable_reason(row, target, candidate_count):
    kernel = str(row.get("short_name") or row.get("raw_name") or "")
    # A cause recorded by the marker mapper is measured, not inferred, so it
    # outranks the name-shape guesses below -- except for rows that carry no
    # model tensor at all, which are handled first.
    cause = (target.get("runtime_marker_unbound_cause") or {}).get("code")
    if "__amd_rocclr_fillBufferAligned" in kernel:
        return (
            "runtime_internal_buffer_operation",
            "runtime buffer operation has no source-confirmed model tensor wrapper")
    if row.get("event_type") == "gpu_memcpy" or kernel == "Memcpy":
        return (
            "runtime_copy_without_unique_tensor",
            "runtime copy has no unique source-confirmed model tensor attribution")
    if not kernel:
        return (
            "unnamed_runtime_kernel",
            "trace event has no stable kernel identity for probe correlation")
    if candidate_count > 1:
        return (
            "multiple_wrapper_candidates",
            "multiple matching wrapper instances prevent unique shape attribution")
    if cause in _UNBOUND_CAUSE_REASONS:
        return _UNBOUND_CAUSE_REASONS[cause]
    status = target.get("runtime_marker_mapping_status")
    if status == "not_found":
        return (
            "kernel_not_observed_in_probe",
            "target kernel was not observed inside a matching runtime probe marker")
    if status == "ambiguous_count":
        return (
            "ambiguous_probe_cardinality",
            "probe launch cardinality does not uniquely match the clean trace row")
    if kernel == "kentry":
        return (
            "non_unique_native_kernel_name",
            "generic native kernel name cannot be uniquely assigned to a wrapper")
    return (
        "no_source_confirmed_wrapper",
        "no source-confirmed op_path or wrapper candidate was available")


def _is_runtime_internal(row):
    kernel = str(row.get("short_name") or row.get("raw_name") or "")
    return "__amd_rocclr_fillBufferAligned" in kernel


def _context_value(fields, name):
    value = (fields or {}).get(name)
    return value.get("value") if isinstance(value, dict) else value


def _structural_summary(table):
    context = table.get("structural_context") or {}
    static = context.get("static_model_context") or {}
    runtime = context.get("runtime_context") or {}
    scope = context.get("pattern_scope") or {}
    attention_type = str(scope.get("attention_type", "")).lower()
    ffn_type = str(scope.get("ffn_type", "")).lower()
    attention_categories = (
        {"mla"} if "mla" in attention_type else
        {"linear_attention"} if "linear" in attention_type else
        {"full_attention"})
    ffn_categories = (
        {"moe"} if ffn_type == "moe" else {"dense_ffn"})
    enabled_categories = (
        {"common", "quantization"} | attention_categories | ffn_categories)
    values = []
    for category, fields in (
            ("common", ("hidden_size", "model_dtype", "norm_type")),
            ("full_attention", (
                "num_attention_heads", "num_key_value_heads", "head_dim")),
            ("mla", (
                "q_lora_rank", "kv_lora_rank", "qk_nope_head_dim",
                "qk_rope_head_dim", "v_head_dim")),
            ("linear_attention", (
                "key_heads", "key_head_dim", "value_heads",
                "value_head_dim", "conv_kernel_dim")),
            ("dense_ffn", ("intermediate_size", "activation")),
            ("moe", (
                "num_experts", "experts_per_token", "num_shared_experts",
                "shared_expert_intermediate_size", "moe_intermediate_size")),
            ("quantization", (
                "quant_method", "weight_block_size", "activation_scheme"))):
        if category not in enabled_categories:
            continue
        category_fields = static.get(category) or {}
        for name in fields:
            value = _context_value(category_fields, name)
            if value is not None:
                values.append("%s.%s=%s" % (category, name, value))
    for name in ("tensor_parallel_size", "expert_parallel_size"):
        value = _context_value(runtime, name)
        if value is not None:
            values.append("runtime.%s=%s" % (name, value))
    return ", ".join(values) if values else "unavailable"


def _layer_io_summary(table):
    layer_io = table.get("layer_io") or {}
    parts = ["source=%s" % layer_io.get("source", "unavailable")]
    for name in ("input", "output"):
        tensor = layer_io.get(name)
        if not tensor:
            continue
        dims = tensor.get("effective_shape") or tensor.get("shape") or []
        parts.append("%s=%s[%s]" % (
            name, _dtype_label(tensor.get("dtype")),
            "×".join(str(value) for value in dims)))
    if layer_io.get("bucket_match"):
        parts.append("bucket=%s" % layer_io["bucket_match"])
    return ", ".join(parts)


def _markdown(table_doc):
    lines = ["# Ordered Unique Layer Kernel Tables — Semantics 1.2", ""]
    for table in table_doc.get("tables", []):
        lines.extend([
            "## %s — %s" % (
                str(table["phase"]).upper(),
                table.get("pattern_display_name", table["pattern_id"])),
            "",
            "- pattern layers (%d): `%s`" % (
                int(table.get(
                    "pattern_layer_count",
                    len(table.get("pattern_layer_ids", [])))),
                json.dumps(table.get("pattern_layer_ids", []))),
            "- representative layer: `L%s`" % table["representative_layer_id"],
            "- selected bucket: `%s`" % json.dumps(
                table.get("selected_bucket", {}), sort_keys=True),
            "- complete-layer device event count: `%s`" % table.get(
                "event_count", len(table.get("rows", []))),
            "- raw one-layer device event total us: `%.3f`" % float(
                table.get("layer_total_us", 0)),
            "- structural context: `%s`" % _structural_summary(table),
            "- representative layer I/O: `%s`" % _layer_io_summary(table),
            "",
            "| pos | stage | kernel | parent operator | shape type | duration us | layer total % |",
            "|---:|---|---|---|---|---:|---:|",
        ])
        for row in table.get("rows", []):
            lines.append("| %d | %s | `%s` | %s | %s | %.3f | %.3f |" % (
                row["pos"], row.get("stage", "unknown"),
                row.get("short_name", row.get("raw_name", "?")),
                str(row.get("parent_operator", {}).get(
                    "canonical_op", "unresolved")).replace("|", "\\|"),
                _shape_text(row).replace("|", "\\|"),
                float(row.get("duration_us", 0)),
                float(row.get("layer_total_pct", 0))))
        lines.append("")
    return "\n".join(lines) + "\n"


def _is_trace_shape(value):
    """A trace shape is a list of plain ints -- possibly empty, for a 0-d tensor."""
    return isinstance(value, list) and all(
        isinstance(item, int) and not isinstance(item, bool)
        for item in value)


def _two_trace_arg_shapes(index, dim):
    """One trace Input Dims argument -> (arg_name, shape) per tensor it carries.

    Most arguments are one tensor, so ``[4, 7168]`` is one shape.  A TensorList
    argument nests one shape per element instead -- ``aten::cat`` reaches here
    as ``[[4, 16, 512], [4, 16, 64]]`` -- and those element shapes are exactly
    what a cat row needs, so the list is expanded rather than dropped.

    Only these two forms are read.  A scalar argument's ``[]``, a mixed list, or
    a nesting deeper than one level carries nothing this can name unambiguously,
    so it yields no tensor rather than a guess.
    """
    if not isinstance(dim, list) or not dim:
        return []
    if _is_trace_shape(dim):
        return [("args[%d]" % index, list(dim))]
    if all(_is_trace_shape(inner) for inner in dim):
        return [("args[%d][%d]" % (index, position), list(inner))
                for position, inner in enumerate(dim) if inner]
    return []


# A transplanted shape keeps the scope it had in the mapping trace. Only a
# kernel_exact mapping row -- one cpu_op, one device kernel -- describes the
# bound kernel's own operands; a 1:N parent hands the same dims to every kernel
# it launched, so reporting those as kernel scope would assert N kernels share
# one operand list. semantic_two_trace_mapping._shape_scope makes the call.
TWO_TRACE_SHAPE_SOURCES = {
    "kernel": "two_trace_graph_off_external_id",
    "wrapper": "two_trace_graph_off_parent_context",
}
TWO_TRACE_ROW_SHAPE_SOURCES = {
    "kernel": "two_trace_kernel_dims",
    "wrapper": "two_trace_wrapper_dims",
}


def _two_trace_shape_scope(entry):
    scope = (entry.get("shape") or {}).get("scope")
    if scope in TWO_TRACE_SHAPE_SOURCES:
        return scope
    # Maps written before the scope field existed recorded the mapping row's
    # own shape source, which is what the scope is derived from.
    return ("kernel"
            if (entry.get("shape") or {}).get("source") == "kernel_exact"
            else "wrapper")


def _two_trace_schema(entry):
    """Trace-native Input Dims of the bound kernel, as an axis-free tensor list."""
    tensors = []
    shape_source = TWO_TRACE_SHAPE_SOURCES[_two_trace_shape_scope(entry)]
    dims = (entry.get("shape") or {}).get("input_dims") or []
    types = (entry.get("shape") or {}).get("input_types") or []
    for index, dim in enumerate(dims):
        dtype = types[index] if index < len(types) else None
        for arg_name, shape in _two_trace_arg_shapes(index, dim):
            tensors.append({
                "arg_name": arg_name,
                "io": "input",
                "shape": shape,
                "dtype": dtype,
                "source": shape_source,
            })
    return {"tensors": tensors, "linear_interface": None}


def _two_trace_evidence(entry, table):
    """P evidence recovered from the workload-identical graph-off run.

    The op is kernel scope whenever this is reached -- the binding resolved
    which launch the row is.  The shape is only kernel scope when the mapping
    row's dims were the kernel's own; ``shape_scope`` keeps the two readable
    apart rather than letting the op's precision speak for the dims.
    """
    match = entry.get("match") or {}
    shape_scope = _two_trace_shape_scope(entry)
    return {
        "level": "P",
        "probe_scope": "kernel",
        "status": "matched",
        "source": TWO_TRACE_SHAPE_SOURCES[shape_scope],
        "evidence_origin": "two_trace_mapping",
        "contained_by": (entry.get("op") or {}).get("canonical_op"),
        "op_instance_id": (entry.get("op") or {}).get("op_instance_id"),
        "confidence": match.get("confidence", "medium"),
        "mapping_basis": (
            "kernel name + ordered position inside the layer instance "
            "(order-preserving LCS; name match %s, position delta %s)" % (
                match.get("kernel_name_match"),
                match.get("position_delta"))),
        "binding": match.get("binding"),
        "position_delta": match.get("position_delta"),
        "kernel_name_match": match.get("kernel_name_match"),
        "representative_layer_match": match.get("representative_layer_match"),
        "shape_scope": shape_scope,
        "shape_source": TWO_TRACE_SHAPE_SOURCES[shape_scope],
        "mapping_shape_cardinality": (
            "1:1" if shape_scope == "kernel" else "1:N"),
        "bucket_match": (
            "exact" if match.get("representative_layer_match") == "same_layer"
            else "compatible"),
        "wrapper_scope": None,
        "source_evidence": [],
        "schema": _two_trace_schema(entry),
    }


def merge(table_path, capture_plan_path, shape_log_path, out_dir,
          two_trace_map_path=""):
    table_doc = _load(table_path)
    capture_plan = _load(capture_plan_path)
    parameter_symbols = (
        (capture_plan.get("operator_parameter_index") or {}).get("symbols")
        or {})
    two_trace_entries = {}
    if two_trace_map_path:
        two_trace_entries = (
            _load(two_trace_map_path).get("entries") or {})
    groups = _groups(_shape_records(shape_log_path))
    target_by_row = {
        target["row_id"]: target
        for target in capture_plan.get("capture_targets", [])}
    audits = []
    for table in table_doc.get("tables", []):
        table_groups = [
            group for group in groups
            if group["rank"] == 0
            and group["phase"] == str(table["phase"]).lower()
            and group["layer_id"] == int(table["representative_layer_id"])]
        clean_bucket = table.get("selected_bucket") or {}
        exact_groups = [
            group for group in table_groups
            if group["batch_size"] == clean_bucket.get("batch_size")
            and group["input_tokens"] == clean_bucket.get("input_tokens")]
        selected_groups = exact_groups
        if not selected_groups and table_groups:
            clean_bs = clean_bucket.get("batch_size")
            clean_tokens = (
                clean_bucket.get("input_tokens") or clean_bs or -1)
            buckets = {}
            for group in table_groups:
                key = (group["batch_size"], group["input_tokens"])
                buckets.setdefault(key, []).append(group)
            matching_bs = [
                (key, values) for key, values in buckets.items()
                if key[0] == clean_bs]
            choices = matching_bs or list(buckets.items())
            _, selected_groups = min(
                choices, key=lambda item: (
                    abs(item[0][1] - clean_tokens), item[0]))
        if selected_groups:
            first_inputs = [
                tensor for tensor in selected_groups[0]["tensors"]
                if tensor["io"] == "input"]
            last_outputs = [
                tensor for tensor in selected_groups[-1]["tensors"]
                if tensor["io"] == "output"]
            table["layer_io"] = {
                "source": "shape_logger",
                "bucket_match": "exact" if exact_groups else "compatible",
                "logger_context": {
                    "batch_size": selected_groups[0]["batch_size"],
                    "input_tokens": selected_groups[0]["input_tokens"],
                },
                "input": _layer_tensor(
                    first_inputs[0] if first_inputs else None,
                    table, selected_groups[0]),
                "output": _layer_tensor(
                    last_outputs[-1] if last_outputs else None,
                    table, selected_groups[-1]),
                "note": (
                    "Layer boundary I/O from first/last captured representative "
                    "wrapper; not copied to internal Kernel exact shapes."),
            }
        else:
            table["layer_io"] = {
                "source": "unavailable", "bucket_match": "unavailable",
                "input": None, "output": None,
            }
        # Targeted-launcher probes are resolved once per table: the binding is
        # a cross-row decision (counts must line up within a symbol bucket),
        # so it cannot be made from inside the per-row branch below.
        launcher_bindings = _launcher_bindings(
            table.get("rows", []), groups, target_by_row, table)
        for row in table.get("rows", []):
            original = json.loads(json.dumps(row))
            two_trace_entry = two_trace_entries.get(row["row_id"])
            # Operator attribution is transplanted whenever the graph-off run
            # resolved it and this graph-on row could not.  That is independent
            # of which shape evidence wins below: a replayed graph loses the
            # parent link and the Input Dims for the same reason, but a row may
            # legitimately take its shape from the runtime probe while still
            # wanting the recovered parent.
            # python_stack is overridable, but only by a binding that actually
            # adds something. Two-trace reaches across runs to a different layer
            # instance, so when its op came from a python stack too and it
            # carries no dims, it restates what this trace already said from
            # closer up. An external-id parent is never overridden.
            two_trace_op = (two_trace_entry or {}).get("op") or {}
            two_trace_adds = (
                two_trace_op.get("mapping_level") != "python_stack"
                or bool((two_trace_entry or {}).get("shape", {}).get(
                    "recovered")))
            own_level = row.get("parent_operator", {}).get("mapping_level")
            if (two_trace_entry and two_trace_op.get("recovered")
                    and (own_level in (None, "unresolved")
                         or (own_level == "python_stack" and two_trace_adds))):
                recovered_op = two_trace_entry["op"]
                row["parent_operator"] = {
                    **row.get("parent_operator", {}),
                    "canonical_op": recovered_op.get("canonical_op"),
                    "op_instance_id": recovered_op.get("op_instance_id"),
                    "mapping_level": (
                        "two_trace_python_stack"
                        if recovered_op.get("mapping_level") == "python_stack"
                        else "two_trace_external_id"),
                    "mapping_cardinality": recovered_op.get(
                        "mapping_cardinality", "unresolved"),
                    "device_launch_count": recovered_op.get(
                        "device_launch_count"),
                    "confidence": (
                        two_trace_entry.get("match") or {}).get(
                            "confidence", "medium"),
                    "recovered_by": "two_trace_mapping",
                }
            if row.get("shape", {}).get("source") == "kernel_exact":
                evidence = {
                    "level": "K", "status": "preserved",
                    "source": "clean_trace_external_id",
                }
            elif (two_trace_entry
                    and (two_trace_entry.get("shape") or {}).get("recovered")):
                evidence = _two_trace_evidence(two_trace_entry, table)
                recovered_shape = two_trace_entry["shape"]
                row["shape"] = {
                    **row.get("shape", {}),
                    "source": TWO_TRACE_ROW_SHAPE_SOURCES[
                        evidence["shape_scope"]],
                    "input_dims": recovered_shape.get("input_dims"),
                    "input_types": recovered_shape.get("input_types"),
                    "logger_schema": evidence["schema"],
                }
            else:
                target = target_by_row.get(row["row_id"], {})
                # The shape-logger probe below knows only the enclosing module,
                # so it must not overwrite an op that a kernel-scope route
                # already established -- the logger's shape is worth taking,
                # its "model.layers.N" is not.
                op_level = row.get("parent_operator", {}).get("mapping_level")
                kernel_scope_op = op_level in KERNEL_SCOPE_OP_LEVELS
                runtime_internal = _is_runtime_internal(row)
                candidates = (
                    [] if runtime_internal
                    else _candidate_groups(
                        row, target, groups, table, two_trace_entry))
                # A launcher probe observed this very launch, so it outranks
                # any wrapper candidate matched by name/order below.
                launcher_group = (
                    None if runtime_internal
                    else launcher_bindings.get(row["row_id"]))
                if launcher_group is not None:
                    candidates = [launcher_group]
                alignment_source = False
                if len(candidates) == 1:
                    group = candidates[0]
                    bucket_status = _bucket_status(table, group)
                    cardinality = target.get(
                        "mapping_cardinality", "unresolved")
                    # A direct launcher probe IS the kernel-scope route: it
                    # wrapped the launcher and logged its operands. Requiring
                    # candidate_terminal_launcher (which semantic_source_mapping
                    # leaves unset when it reports not_found) or cardinality
                    # 1:1 (which a legitimately multi-launch wrapper never
                    # reaches) would discard that evidence and fall back to the
                    # enclosing module's shapes.
                    kernel_exact = (
                        not alignment_source
                        and (launcher_group is not None
                             or (cardinality == "1:1"
                                 and bool(target.get(
                                     "candidate_terminal_launcher")))))
                    level = "P"
                    probe_scope = "kernel" if kernel_exact else "wrapper"
                    evidence = {
                        "level": level,
                        "probe_scope": probe_scope,
                        "status": "matched",
                        "source": (
                            "shape_logger_terminal_launcher"
                            if kernel_exact else
                            "ordered_parent_wrapper_alignment"
                            if alignment_source else
                            "shape_logger_parent_wrapper"),
                        "contained_by": group["op_path"],
                        "op_instance_id": group["op_instance_id"],
                        "confidence": "medium" if alignment_source else "high",
                        "mapping_basis": (
                            "ordered leaf-wrapper sequence advanced only by "
                            "a matching semantic anchor"
                            if alignment_source else
                            "unique source/runtime candidate"),
                        "wrapper_scope": (
                            target.get("shape_log_layer_evidence", {})
                            .get("scope")),
                        "source_evidence": target.get("source_evidence", []),
                        "bucket_match": bucket_status,
                        "schema": _tensor_schema(
                            group, row, table, bucket_status == "exact"),
                    }
                    if kernel_scope_op and probe_scope == "wrapper":
                        # Keep the logger's shape, but not its op. Only the op
                        # is kernel-scope here, which shape_scope records so the
                        # two are not read as one claim.
                        evidence.update({
                            "probe_scope": "kernel",
                            "source": OP_LEVEL_SOURCES[op_level],
                            "contained_by": row["parent_operator"].get(
                                "canonical_op"),
                            "evidence_origin": OP_LEVEL_ORIGINS[op_level],
                            "shape_scope": "wrapper",
                            "shape_source": "shape_logger_parent_wrapper",
                            "wrapper_op_path": group["op_path"],
                            "confidence": "medium",
                            "mapping_basis": OP_LEVEL_BASIS[op_level],
                        })
                    else:
                        row["parent_operator"] = {
                            **row.get("parent_operator", {}),
                            "canonical_op": group["op_path"],
                            "mapping_level": (
                                "logger_one_to_one" if probe_scope == "kernel"
                                else "parent_wrapper_context"),
                            "confidence": (
                                "high" if probe_scope == "kernel"
                                else "medium"),
                        }
                    row["shape"] = {
                        **row.get("shape", {}),
                        "source": (
                            "runtime_probe_kernel"
                            if probe_scope == "kernel"
                            else "runtime_probe_wrapper"),
                        "logger_schema": evidence["schema"],
                    }
                elif kernel_scope_op:
                    # No usable shape probe, but the row is not unattributed:
                    # the trace says which call launched it.
                    evidence = {
                        "level": "P",
                        "probe_scope": "kernel",
                        "status": "matched",
                        "source": OP_LEVEL_SOURCES[op_level],
                        "evidence_origin": OP_LEVEL_ORIGINS[op_level],
                        "contained_by": row["parent_operator"].get(
                            "canonical_op"),
                        "shape_scope": "none",
                        "confidence": "medium",
                        "mapping_basis": OP_LEVEL_BASIS[op_level],
                        "candidate_count": len(candidates),
                        "wrapper_scope": None,
                        "source_evidence": [],
                    }
                else:
                    reason_code, reason = _unavailable_reason(
                        row, target, len(candidates))
                    evidence = {
                        "level": "U",
                        "status": "unavailable",
                        "source": "no_unique_parent_wrapper",
                        "candidate_count": len(candidates),
                        "reason_code": reason_code,
                        "reason": reason,
                    }
            row["semantic_evidence"] = evidence
            # After attribution, not before: two-trace mapping transplants the
            # operator for exactly the rows a graph-replayed stage could not
            # resolve, so the symbol to look up does not exist yet at the top
            # of this loop.
            _attach_source_parameter_names(row, parameter_symbols)
            audits.append({
                "phase": table["phase"],
                "pattern_id": table["pattern_id"],
                "representative_layer_id": table["representative_layer_id"],
                "pos": row["pos"],
                "row_id": row["row_id"],
                "kernel": row["short_name"],
                "evidence": evidence,
                "parent_recovered_by": row.get(
                    "parent_operator", {}).get("recovered_by"),
                "clean_trace_identity_unchanged": all(
                    row.get(key) == original.get(key)
                    for key in ("row_id", "raw_event_index", "device_seq_index",
                                "raw_name", "short_name", "duration_us")),
            })
    os.makedirs(out_dir, exist_ok=True)
    table_out = os.path.join(out_dir, "pattern_layer_kernel_table.json")
    markdown_out = os.path.join(out_dir, "ORDERED_UNIQUE_LAYER_TABLES.md")
    audit_out = os.path.join(out_dir, "KERNEL_SEMANTIC_EVIDENCE.jsonl")
    verify_out = os.path.join(out_dir, "SHAPE_TYPE_VERIFICATION.json")
    coverage_out = os.path.join(out_dir, "OP_COVERAGE_MANIFEST.json")
    with open(table_out, "w") as fh:
        json.dump(table_doc, fh, indent=2)
    with open(markdown_out, "w") as fh:
        fh.write(_markdown(table_doc))
    with open(audit_out, "w") as fh:
        for audit in audits:
            fh.write(json.dumps(audit, sort_keys=True) + "\n")
    unchanged = all(
        audit["clean_trace_identity_unchanged"] for audit in audits)
    table_checks = []
    for table in table_doc.get("tables", []):
        rows = table.get("rows", [])
        duration_sum = round(sum(
            float(row.get("duration_us", 0) or 0) for row in rows), 6)
        check = {
            "phase": table["phase"],
            "pattern_id": table["pattern_id"],
            "representative_layer_id": table["representative_layer_id"],
            "row_count": len(rows),
            "declared_event_count": table.get("event_count"),
            "ordered_positions": [
                row.get("pos") for row in rows] == list(range(len(rows))),
            "duration_sum_us": duration_sum,
            "declared_layer_total_us": table.get("layer_total_us"),
        }
        check["status"] = "pass" if (
            check["row_count"] == check["declared_event_count"]
            and check["ordered_positions"]
            and abs(duration_sum - float(
                check["declared_layer_total_us"] or 0)) <= 1e-6
        ) else "fail"
        table_checks.append(check)
    unchanged = unchanged and all(
        check["status"] == "pass" for check in table_checks)
    counts = {}
    for audit in audits:
        level = audit["evidence"]["level"]
        counts[level] = counts.get(level, 0) + 1
    two_trace_stats = {
        "enabled": bool(two_trace_map_path),
        "map_path": (
            os.path.abspath(two_trace_map_path)
            if two_trace_map_path else ""),
        "shape_rows": sum(
            1 for audit in audits
            if audit["evidence"].get("source")
            in TWO_TRACE_SHAPE_SOURCES.values()),
        "kernel_scope_shape_rows": sum(
            1 for audit in audits
            if audit["evidence"].get("source")
            == TWO_TRACE_SHAPE_SOURCES["kernel"]),
        "wrapper_scope_shape_rows": sum(
            1 for audit in audits
            if audit["evidence"].get("source")
            == TWO_TRACE_SHAPE_SOURCES["wrapper"]),
        "operator_rows": sum(
            1 for audit in audits
            if audit.get("parent_recovered_by") == "two_trace_mapping"),
    }
    # Same definition the evidence ledger uses: every row is classified and
    # every U row carries a machine-readable reason_code.
    unexplained = [
        audit for audit in audits
        if audit["evidence"]["level"] == "U"
        and not audit["evidence"].get("reason_code")]
    classification_complete = (
        sum(counts.values()) == len(audits) and not unexplained)
    verification = {
        "schema_version": 1,
        "status": "pass" if unchanged else "fail",
        "clean_trace_identity_unchanged": unchanged,
        "classification_complete": classification_complete,
        "unexplained_u_count": len(unexplained),
        "evidence_counts": counts,
        "row_count": len(audits),
        "shape_log_group_count": len(groups),
        "two_trace_mapping": two_trace_stats,
        "representative_table_checks": table_checks,
    }
    with open(verify_out, "w") as fh:
        json.dump(verification, fh, indent=2)
    unavailable = []
    for audit in audits:
        if audit["evidence"]["level"] != "U":
            continue
        item = {
            key: audit[key] for key in (
                "phase", "pattern_id", "representative_layer_id",
                "pos", "row_id", "kernel")}
        # Both fields are required: `reason` is for a human reading the
        # table, `reason_code` is what validate_kpu_model_pair gates on.
        # Emitting only the prose made every U row fail that gate.
        item["reason_code"] = audit["evidence"].get("reason_code")
        item["reason"] = audit["evidence"].get("reason")
        unavailable.append(item)
    unavailable_reason_counts = {}
    for item in unavailable:
        code = item.get("reason_code") or "unspecified"
        unavailable_reason_counts[code] = (
            unavailable_reason_counts.get(code, 0) + 1)
    coverage = {
        "schema_version": 1,
        "scope": "representative_layers_only",
        "status": verification["status"],
        "classification_complete": verification.get(
            "classification_complete", False),
        "row_count": len(audits),
        "evidence_counts": counts,
        "covered_count": sum(
            count for level, count in counts.items() if level != "U"),
        "unavailable": unavailable,
        "unavailable_reason_counts": unavailable_reason_counts,
    }
    with open(coverage_out, "w") as fh:
        json.dump(coverage, fh, indent=2)
    return {
        "status": verification["status"],
        "semantic_table_json": table_out,
        "semantic_table_md": markdown_out,
        "kernel_semantic_evidence_jsonl": audit_out,
        "shape_type_verification_json": verify_out,
        "op_coverage_manifest": coverage_out,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", required=True)
    parser.add_argument("--capture-plan", required=True)
    parser.add_argument("--shape-log", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--two-trace-map", default="")
    parser.add_argument("--result-json", default="")
    args = parser.parse_args()
    result = merge(
        args.table, args.capture_plan, args.shape_log, args.out_dir,
        args.two_trace_map)
    if args.result_json:
        with open(args.result_json, "w") as fh:
            json.dump(result, fh, indent=2)
    print(json.dumps(result))
    return 0 if result["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
