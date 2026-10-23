"""Throwaway online Routing graph probe: 20 agents, one forward/variant, no updates.

Original builders and online assembly are extracted verbatim; only the approved
two per-head contractions change. No compile, target graph, packet environment.
"""
import ast
from datetime import datetime, timezone
import difflib
import importlib.util
import json
import os
from pathlib import Path
import struct
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("per_head_repair", ROOT / "routing_repair_probe.py")
repair = importlib.util.module_from_spec(spec)
spec.loader.exec_module(repair)
probe = repair.probe


def assignment(node, name):
    return isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)


def extract_online(source):
    tree, lines = ast.parse(source), source.splitlines(keepends=True)
    names = ("MLP", "MultiHeadsAttModel", "Q_Net")
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    starts = [node for node in tree.body if assignment(node, "encoder")]
    ends = [node for node in tree.body if assignment(node, "model")]
    if len(functions) != 3 or set(node.name for node in functions) != set(names) or len(starts) != 1 or len(ends) != 1 or starts[0].lineno >= ends[0].lineno:
        raise ValueError("Missing/ambiguous upstream online graph boundaries")
    imports = [node for node in tree.body if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("keras")]
    definitions = "".join("".join(lines[node.lineno-1:node.end_lineno]) for node in imports + functions)
    graph = "".join(lines[starts[0].lineno-1:ends[0].end_lineno])
    for node in ast.walk(ast.parse(graph)):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in ("compile", "fit", "train_on_batch", "apply_gradients", "minimize"):
            raise ValueError("Optimizer/training call in online graph extraction")
    return definitions, graph


def neighbor_indices(agent):
    return [(agent + offset) % 20 for offset in range(4)]


def native_run(variant, directory):
    output = Path(directory)
    summary = {"variant": variant, "implementation": "PER_HEAD_REPAIRED_ONLINE_GRAPH",
               "status": "STARTED", "stage": "RUNTIME_GUARD", "python": sys.version,
               "agents": 20, "features": 35, "neighbors": 4, "heads": 8, "batch_size": 1,
               "device": "cpu", "fixture_seed": 0, "forward_batches": 0, "optimizer_steps": 0,
               "environment": "NOT_RUN", "training": "NOT_RUN", "target_graph": "NOT_RUN"}

    def save():
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    try:
        save()
        if sys.version_info[:2] != (3, 7) or struct.calcsize("P") != 8 or not __debug__:
            raise RuntimeError("Python3.7x64 required with assertions enabled")
        if sys.prefix == sys.base_prefix or not sys.flags.isolated:
            raise RuntimeError("Isolated venv required")
        if "include-system-site-packages = false" not in (Path(sys.prefix) / "pyvenv.cfg").read_text(encoding="utf-8").lower():
            raise RuntimeError("Venv must exclude system site packages")
        summary.update(isolated=True, venv_prefix=sys.prefix, base_prefix=sys.base_prefix, stage="PACKAGE_METADATA")
        save()
        packages = probe.installed_packages(sys.path)
        summary["installed_packages"] = packages
        versions = {row["name"].lower(): row["version"] for row in packages}
        for name, expected in dict(probe.PINS, h5py="2.10.0", protobuf="3.20.3").items():
            if versions.get(name) != expected:
                raise RuntimeError("Package pin mismatch: " + name)
        summary["stage"] = "IMPORTS"
        save()
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        os.environ["KERAS_BACKEND"] = "tensorflow"
        import numpy as np
        import tensorflow as tf
        namespace = {"np": np, "tf": tf, "n_data": 20, "neighbors": 4, "len_feature": 35, "action_space": 4}
        definitions = output / "model_definitions.py"
        exec(compile(definitions.read_text(encoding="utf-8"), str(definitions), "exec"), namespace)
        if tf.config.get_visible_devices("GPU"):
            raise RuntimeError("GPU visible despite CPU-only request")
        summary["runtime_versions"] = {"numpy": np.__version__, "tensorflow": tf.__version__, "keras": versions["keras"]}
        np.random.seed(0)
        tf.random.set_seed(0)
        summary["stage"] = "BUILD"
        save()
        with tf.device("/CPU:0"):
            assembly = output / "online_graph.py"
            exec(compile(assembly.read_text(encoding="utf-8"), str(assembly), "exec"), namespace)
            model = namespace["model"]
            summary.update(input_count=len(model.inputs), output_count=len(model.outputs), output_shapes=model.output_shape)
            if len(model.inputs) != 41 or len(model.outputs) != 20:
                raise ValueError("Expected 41 inputs and 20 Q outputs")
            observations = np.linspace(-1, 1, 700, dtype="float32").reshape(1, 20, 35)
            inputs, masks = [], []
            for agent in range(20):
                mask = np.zeros((1, 4, 20), dtype="float32")
                for slot, neighbor in enumerate(neighbor_indices(agent)):
                    mask[0, slot, neighbor] = 1
                masks.append(mask)
                inputs.extend([observations[:, agent, :], mask])
            selector = namespace["vec"].astype("float32")[None, :, :]
            inputs.append(selector)
            np.savez(output / "fixture.npz", observations=observations, gather_masks=np.stack(masks, axis=1), selector=selector)
            summary["stage"] = "FORWARD"
            save()
            prediction = model.predict(inputs, batch_size=1, verbose=0)
            summary["forward_batches"] = 1
            if not isinstance(prediction, list) or len(prediction) != 20:
                raise ValueError("Expected 20 Q-value arrays")
            q = np.stack(prediction, axis=1)
            (output / "q_values.json").write_text(json.dumps(q.tolist()) + "\n", encoding="utf-8")
            if q.shape != (1, 20, 4) or not np.isfinite(q).all():
                raise ValueError("Nonfinite/unexpected Q values: " + str(q.shape))
            summary.update(status="COMPLETED_PENDING_REVIEW", stage="FINISHED", q_shape=list(q.shape),
                           finite=True, q_min=float(q.min()), q_max=float(q.max()))
    except Exception:
        summary.update(status="FAILED", error=traceback.format_exc())
        raise
    finally:
        save()


def main(argv=None):
    args = probe.parse_args(argv)
    if sys.version_info[:2] != (3, 12) or not __debug__:
        raise RuntimeError("Supervisor requires Python3.12 with assertions enabled")
    control = probe.controls()
    if control.sha(repair.ORIGINAL_PROBE) != "06d2aed1b91c7af3ac16ad5229f1251cea9989f3d652f52565b13c087d194332":
        raise ValueError("Original probe identity drift")
    runtime = args.runtime.resolve(strict=True)
    audit = control.check_identity()
    output = control.new_output(args.output, ROOT / "routing_full_model_checks")
    inputs = [Path(__file__).resolve(), ROOT / "check_routing_full_model_probe.py", ROOT / "routing_repair_probe.py",
              repair.ORIGINAL_PROBE, ROOT / "routing_constraints.txt", ROOT / "compat/torch_surviving_cpu_v1/main.py",
              probe.REFERENCE / "check_reference.py", probe.REFERENCE / "manifest.json", runtime]
    hashes = {str(path): control.sha(path) for path in inputs}
    receipt = {"status": "STARTED", "scope": "PER_HEAD_REPAIRED_ONLINE_GRAPH_SYNTHETIC_FORWARD_ONLY",
               "execution_owner": "ASSISTANT_USER_AUTHORIZED", "authorization": "yes approve, full online graph probe, 3 October 2026",
               "output": str(output), "runtime": str(runtime), "hashes_before": hashes,
               "max_seconds_total": args.max_seconds, "started_at_utc": datetime.now(timezone.utc).isoformat(),
               "agents": 20, "features": 35, "neighbors": 4, "heads": 8, "batch_size": 1,
               "optimizer_steps": 0, "device": "cpu", "fixture_seed": 0,
               "regularized_aggregation": "UPSTREAM_RAW_SCORES_PRESERVED", "environment": "NOT_RUN",
               "training": "NOT_RUN", "ns3": "NOT_RUN", "scientific_gates": "CLOSED", "variants": []}
    control.write_json(output / "source_before.json", audit)
    control.write_json(output / "run_receipt.json", receipt)
    deadline, exit_code = time.monotonic() + args.max_seconds, 1
    try:
        for variant, filename in probe.VARIANTS:
            directory = control.new_output(output / variant, output)
            upstream = probe.routing_source(filename)
            original, graph = extract_online(upstream.read_text(encoding="utf-8-sig"))
            repaired = repair.repair_extract(original)
            for name, text in (("original_model_definitions.py", original), ("model_definitions.py", repaired), ("online_graph.py", graph)):
                (directory / name).write_text(text, encoding="utf-8")
            (directory / "repair.patch").write_text("".join(difflib.unified_diff(original.splitlines(keepends=True),
                repaired.splitlines(keepends=True), fromfile="original_model_definitions.py", tofile="model_definitions.py")), encoding="utf-8")
            child = ("import importlib.util; s=importlib.util.spec_from_file_location('full', "
                     + repr(str(Path(__file__).resolve()))
                     + "); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); m.native_run("
                     + repr(variant) + ", " + repr(str(directory)) + ")")
            command = [str(runtime), "-I", "-B", "-u", "-c", child]
            remaining = deadline - time.monotonic()
            status, code = control.run_child(command, directory, remaining) if remaining > 0 else ("NOT_STARTED_DEADLINE", 124)
            row = {"variant": variant, "process_status": status, "exit_code": code, "command": command,
                   "upstream_sha256": control.sha(upstream),
                   "extract_sha256": {name: control.sha(directory / name) for name in ("original_model_definitions.py", "model_definitions.py", "online_graph.py", "repair.patch")}}
            if (directory / "summary.json").exists():
                row["summary"] = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
            if code == 0:
                s = row.get("summary", {})
                if s.get("status") != "COMPLETED_PENDING_REVIEW" or s.get("q_shape") != [1, 20, 4] or s.get("forward_batches") != 1 or s.get("optimizer_steps") != 0 or not s.get("finite"):
                    raise ValueError("Full online child lacks approved evidence")
            receipt["variants"].append(row)
            control.write_json(output / "run_receipt.json", receipt)
        codes = [row["exit_code"] for row in receipt["variants"]]
        exit_code = 124 if 124 in codes else (0 if codes == [0, 0] else 1)
        receipt["status"] = "COMPLETED_PENDING_REVIEW" if exit_code == 0 else ("TIMEOUT" if exit_code == 124 else "FAILED_PENDING_REVIEW")
    except KeyboardInterrupt:
        receipt["status"], exit_code = "INTERRUPTED", 130
    except Exception:
        receipt["status"], exit_code = "SUPERVISOR_FAILED", 1
        receipt["supervisor_error"] = traceback.format_exc()
    finally:
        try:
            control.write_json(output / "source_after.json", control.check_identity())
            receipt["hashes_after"] = {str(path): control.sha(path) for path in inputs}
            if hashes != receipt["hashes_after"]:
                raise ValueError("Probe/runtime identity changed during execution")
        except Exception:
            receipt["identity_error"] = traceback.format_exc()
            receipt["status"], exit_code = "IDENTITY_FAILED", 1
        receipt.update(exit_code=exit_code, finished_at_utc=datetime.now(timezone.utc).isoformat())
        control.write_json(output / "run_receipt.json", receipt)
    print("{}: {}".format(receipt["status"], output))
    return exit_code


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    sys.exit(main())
