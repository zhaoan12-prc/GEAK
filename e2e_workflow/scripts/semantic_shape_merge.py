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


def _candidate_groups(
        row, target, groups, table, allow_verified_cross_layer=False):
    phase = str(table["phase"]).lower()
    layer_id = int(table["representative_layer_id"])
    phase_candidates = [
        group for group in groups
        if group["rank"] == 0 and group["phase"] == phase]
    op_instance_id = target.get("candidate_op_instance_id")
    if op_instance_id:
        matched = [
            group for group in phase_candidates
            if group.get("op_instance_id") == op_instance_id]
        if allow_verified_cross_layer:
            return matched
        return [group for group in matched if group["layer_id"] == layer_id]
    candidates = [
        group for group in phase_candidates if group["layer_id"] == layer_id]
    explicit = target.get("candidate_op_path") or target.get(
        "candidate_wrapper")
    if not explicit and target.get("parent_operator") != "unresolved":
        explicit = target.get("parent_operator")
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


def _axis_0_alignment(tensor, group, table, exact_bucket):
    """Return an auditable axis-0 alignment decision.

    A graph-capture bucket may differ from the clean replay batch. Only a
    floating, rank-2-or-higher activation whose leading dimension equals the
    capture bucket cardinality is eligible for substitution. This excludes
    weights, integer indices/offsets, block tables, and ambiguous 1-D buffers.
    """
    shape = list(tensor.get("shape") or [])
    clean = table.get("selected_bucket") or {}
    target = clean.get("input_tokens") or clean.get("batch_size")
    observed = {
        value for value in (
            group.get("input_tokens"), group.get("batch_size"))
        if isinstance(value, int) and value > 0}
    dtype = str(tensor.get("dtype") or "").lower()
    tensor_path = str(tensor.get("tensor_path") or "").lower()
    io = str(tensor.get("io") or "").lower()
    excluded_name = re.search(
        r"weight|expert|cache|block|index|indices|offset|slot|position",
        tensor_path)
    floating = any(token in dtype for token in (
        "float", "bfloat", "half", "fp8", "bf16", "fp16", "fp32"))
    eligible = bool(
        target and len(shape) >= 2 and shape[0] in observed
        and io in ("input", "output") and floating and not excluded_name)
    effective = list(shape)
    if eligible:
        effective[0] = int(target)
    changed = effective != shape
    return effective, {
        "status": (
            "axis_0_rewritten" if changed else
            "exact_bucket_unchanged" if eligible and exact_bucket else
            "eligible_unchanged" if eligible else "not_eligible"),
        "axis": 0 if eligible else None,
        "axis_role": "token_or_batch" if eligible else "unresolved",
        "logger_value": shape[0] if shape else None,
        "effective_value": effective[0] if effective else None,
        "source": (
            "shape_logger" if eligible and exact_bucket else
            "clean_trace_step" if eligible else "shape_logger"),
        "rule": (
            "floating activation/token-scale axis 0 equals the capture "
            "bucket cardinality" if eligible else
            "axis 0 was not proven to be a batch/token activation axis"),
    }


def _tensor_schema(group, row, table, exact_bucket):
    tensors = []
    trace_dims = row.get("shape", {}).get("input_dims") or []
    for tensor in group["tensors"]:
        item = dict(tensor)
        item["logger_shape"] = list(tensor.get("shape") or [])
        item["axes"] = [
            _axis(value, "unresolved", "shape_logger")
            for value in item["logger_shape"]]
        effective, alignment = _axis_0_alignment(
            item, group, table, exact_bucket)
        item["axis_0_alignment"] = alignment
        if alignment["axis_role"] == "token_or_batch" and item["axes"]:
            item["axes"][0] = _axis(
                effective[0], "token_or_batch", alignment["source"])
        item["effective_shape"] = effective
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
    exact_bucket = _bucket_status(table, group) == "exact"
    value["effective_shape"], value["axis_0_alignment"] = (
        _axis_0_alignment(value, group, table, exact_bucket))
    if value["axis_0_alignment"]["axis"] == 0:
        value["axis_0_source"] = value["axis_0_alignment"]["source"]
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


def _trace_role(row, index):
    op = str(
        row.get("parent_operator", {}).get("canonical_op", "")).lower()
    stage = str(row.get("stage") or "").lower()
    role_maps = (
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
        ("grouped_topk", (
            "logits", "bias", "topk_weights", "topk_ids")),
        ("fmoe", (
            "x", "y", "w13", "w2", "sorted_ids", "sorted_weights",
            "sorted_expert_ids", "num_valid_ids", "workspace",
            "x_scale", "w13_scale", "w2_scale")),
        ("_index_put_impl_", ("cache", "indices", "x")),
        ("::arange", ("start", "end", "step", "y")),
    )
    for token, roles in role_maps:
        if token in op and index < len(roles):
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


def _semantic_shape_text(prefix, tensors):
    inputs = [text for role, text in tensors if not _is_output_role(role)]
    outputs = [text for role, text in tensors if _is_output_role(role)]
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
        tensors = []
        for index, dim in enumerate(dims):
            if not isinstance(dim, list) or not dim:
                continue
            role = _trace_role(row, index)
            dtype = types[index] if index < len(types) else "Tensor"
            tensors.append((role, "%s=%s[%s]" % (
                role, _dtype_label(dtype),
                "×".join(str(value) for value in dim))))
        return _semantic_shape_text("K", tensors)
    schema = shape.get("logger_schema") or {}
    tensors = schema.get("tensors") or []
    if level == "P" and tensors:
        values = []
        output_index = 0
        layer_wrapper = (
            evidence.get("wrapper_scope") == "phase_layer_wrapper")
        for index, tensor in enumerate(tensors[:12]):
            dims = tensor.get("effective_shape") or tensor.get("shape") or []
            role = (
                _layer_wrapper_role(tensor, index, output_index)
                if layer_wrapper else
                _probe_role(row, tensor, index, output_index))
            if str(tensor.get("io") or "").lower() == "output":
                output_index += 1
            values.append((role, "%s=%s[%s]" % (
                role, _dtype_label(tensor.get("dtype")),
                "×".join(str(value) for value in dims))))
        if len(tensors) > 12:
            values.append((
                "metadata", "metadata=+%d tensors" % (len(tensors) - 12)))
        scope = evidence.get("probe_scope", "wrapper")
        return _semantic_shape_text("P(%s)" % scope, values)
    reason_code = evidence.get("reason_code", "unavailable")
    reason = evidence.get("reason", "shape unavailable")
    return "U(%s): %s" % (reason_code, reason)


def _unavailable_reason(row, target, candidate_count):
    kernel = str(row.get("short_name") or row.get("raw_name") or "")
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


def _phase_resolution(audits, groups):
    """B8: a phase the shape log covers must resolve at least one row.

    CUDA-graph replay does not walk the Python module tree, so its clean rows
    may have unresolved parent operators. Graph-construction metadata is bound
    through the separately audited capture plan. If the shape log covers a
    phase but no row resolves, the mapping or sequence audit failed and the
    merge must fail loudly.
    """
    log_phases = set()
    for group in groups:
        if group.get("rank") == 0 and group.get("phase"):
            log_phases.add(str(group["phase"]).lower())
    per_phase = {}
    for audit in audits:
        phase = str(audit.get("phase", "")).lower()
        entry = per_phase.setdefault(phase, {"rows": 0, "resolved": 0})
        entry["rows"] += 1
        if audit["evidence"]["level"] != "U":
            entry["resolved"] += 1
    phases = []
    failed = False
    for phase in sorted(set(per_phase) | log_phases):
        entry = per_phase.get(phase, {"rows": 0, "resolved": 0})
        covered = phase in log_phases
        bad = covered and entry["rows"] > 0 and entry["resolved"] == 0
        failed = failed or bad
        phases.append({
            "phase": phase,
            "rows": entry["rows"],
            "resolved": entry["resolved"],
            "resolved_fraction": round(
                entry["resolved"] / entry["rows"], 4) if entry["rows"] else 0.0,
            "shape_log_covers_phase": covered,
            "status": "fail" if bad else "pass",
            "note": (
                "the shape log carries records for this phase but not one "
                "table row resolved -- CUDA-graph construction markers were "
                "not successfully bound to the clean Kernel rows"
            ) if bad else None,
        })
    return {"status": "fail" if failed else "pass",
            "gating": True,
            "shape_log_phases": sorted(log_phases),
            "phases": phases}


def merge(table_path, capture_plan_path, shape_log_path, out_dir):
    table_doc = _load(table_path)
    capture_plan = _load(capture_plan_path)
    groups = _groups(_shape_records(shape_log_path))
    target_by_row = {
        target["row_id"]: target
        for target in capture_plan.get("capture_targets", [])
        if target.get("row_id")}
    clean_sequence_verified = (
        (capture_plan.get("runtime_marker_mapping") or {})
        .get("clean_table_sequence_audit", {}).get("status") == "pass")
    target_by_position = {}
    if clean_sequence_verified:
        target_by_position = {
            (str(target.get("phase", "")).lower(),
             target.get("pattern_id"), int(target.get("pos", -1))): target
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
        for row in table.get("rows", []):
            original = json.loads(json.dumps(row))
            if row.get("shape", {}).get("source") == "kernel_exact":
                evidence = {
                    "level": "K", "status": "preserved",
                    "source": "clean_trace_external_id",
                }
            else:
                target = target_by_row.get(row["row_id"])
                clean_row_binding = "row_id"
                if target is None and clean_sequence_verified:
                    target = target_by_position.get((
                        str(table.get("phase", "")).lower(),
                        table.get("pattern_id"), int(row.get("pos", -1))))
                    clean_row_binding = "verified_pattern_position"
                target = target or {}
                runtime_internal = _is_runtime_internal(row)
                candidates = (
                    [] if runtime_internal
                    else _candidate_groups(
                        row, target, groups, table,
                        clean_row_binding == "verified_pattern_position"))
                alignment_source = False
                if len(candidates) == 1:
                    group = candidates[0]
                    bucket_status = _bucket_status(table, group)
                    cardinality = target.get(
                        "mapping_cardinality", "unresolved")
                    kernel_exact = (
                        cardinality == "1:1"
                        and bool(target.get("candidate_terminal_launcher"))
                        and not alignment_source)
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
                        "clean_row_binding": clean_row_binding,
                        "wrapper_scope": (
                            target.get("shape_log_layer_evidence", {})
                            .get("scope")),
                        "source_evidence": target.get("source_evidence", []),
                        "bucket_match": bucket_status,
                        "schema": _tensor_schema(
                            group, row, table, bucket_status == "exact"),
                    }
                    wrapper_scope = evidence.get("wrapper_scope")
                    region_only = (
                        wrapper_scope == "phase_layer_semantic_region")
                    canonical_op = (
                        target.get("candidate_op_path")
                        if region_only else group["op_path"])
                    row["parent_operator"] = {
                        **row.get("parent_operator", {}),
                        "canonical_op": canonical_op,
                        "mapping_level": (
                            "logger_one_to_one" if probe_scope == "kernel"
                            else "parent_wrapper_context"),
                        "confidence": (
                            "high" if probe_scope == "kernel" else "medium"),
                    }
                    row["shape"] = {
                        **row.get("shape", {}),
                        "source": (
                            "runtime_probe_kernel"
                            if probe_scope == "kernel"
                            else "runtime_probe_region_context"
                            if region_only else "runtime_probe_wrapper"),
                        # Publish P-level probe shapes through the same canonical
                        # contract consumed by candidate grafting and unitside
                        # provenance. `logger_schema` remains the rich evidence.
                        "input_dims": [] if region_only else [
                            tensor.get("effective_shape") or tensor.get("shape")
                            for tensor in evidence["schema"].get("tensors", [])
                            if (tensor.get("effective_shape") or tensor.get("shape"))
                        ],
                        "input_types": [] if region_only else [
                            tensor.get("dtype") or "Tensor"
                            for tensor in evidence["schema"].get("tensors", [])
                            if (tensor.get("effective_shape") or tensor.get("shape"))
                        ],
                        "logger_schema": evidence["schema"],
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
            audits.append({
                "phase": table["phase"],
                "pattern_id": table["pattern_id"],
                "representative_layer_id": table["representative_layer_id"],
                "pos": row["pos"],
                "row_id": row["row_id"],
                "kernel": row["short_name"],
                "evidence": evidence,
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
    alignment_counts = {}
    alignment_rewrites = []
    for audit in audits:
        tensors = (audit.get("evidence", {}).get("schema", {})
                   .get("tensors", []))
        for tensor in tensors:
            alignment = tensor.get("axis_0_alignment") or {}
            status = alignment.get("status", "not_recorded")
            alignment_counts[status] = alignment_counts.get(status, 0) + 1
            if status == "axis_0_rewritten":
                alignment_rewrites.append({
                    "row_id": audit["row_id"],
                    "tensor_path": tensor.get("tensor_path"),
                    "io": tensor.get("io"),
                    "dtype": tensor.get("dtype"),
                    "logger_shape": tensor.get("logger_shape"),
                    "effective_shape": tensor.get("effective_shape"),
                    "source": alignment.get("source"),
                    "rule": alignment.get("rule"),
                })
    phase_resolution = _phase_resolution(audits, groups)
    verification = {
        "schema_version": 1,
        "status": "pass" if (
            unchanged and phase_resolution["status"] == "pass") else "fail",
        "clean_trace_identity_unchanged": unchanged,
        "evidence_counts": counts,
        "shape_alignment_audit": {
            "policy": (
                "only axis 0 of proven floating activation/token-scale "
                "tensors may follow the clean replay batch/token dimension"),
            "status_counts": alignment_counts,
            "rewritten_tensor_count": len(alignment_rewrites),
            "rewrites": alignment_rewrites,
        },
        "row_count": len(audits),
        "shape_log_group_count": len(groups),
        "representative_table_checks": table_checks,
        "phase_resolution": phase_resolution,
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
        item["reason"] = audit["evidence"].get("reason")
        unavailable.append(item)
    coverage = {
        "schema_version": 1,
        "scope": "representative_layers_only",
        "row_count": len(audits),
        "evidence_counts": counts,
        "covered_count": sum(
            count for level, count in counts.items() if level != "U"),
        "unavailable": unavailable,
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
    parser.add_argument("--result-json", default="")
    args = parser.parse_args()
    result = merge(
        args.table, args.capture_plan, args.shape_log, args.out_dir)
    if args.result_json:
        with open(args.result_json, "w") as fh:
            json.dump(result, fh, indent=2)
    print(json.dumps(result))
    return 0 if result["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
