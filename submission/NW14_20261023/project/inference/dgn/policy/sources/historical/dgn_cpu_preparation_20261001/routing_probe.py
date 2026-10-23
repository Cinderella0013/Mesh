"""Throwaway, user-run TF Routing attention probe; not a routing baseline.

Supervisor: Python 3.12 stdlib. Native child: isolated Python 3.7 x64.
No environment, optimizer, reward/attention repair, or upstream script import.
"""
import argparse
import ast
from datetime import datetime, timezone
from email.parser import Parser
import importlib.util
import json
import os
from pathlib import Path
import struct
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
REFERENCE = ROOT.parent / "marl_reference_20261001"
VARIANTS = (("baseline", "routers.py"), ("regularized", "routers_regularization.py"))
PINS = {"tensorflow": "2.1.0", "keras": "2.3.1", "numpy": "1.17.4", "matplotlib": "3.1.1"}


def controls():
    path = ROOT / "compat/torch_surviving_cpu_v1/main.py"
    spec = importlib.util.spec_from_file_location("cpu_controls", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # Stdlib definitions only.
    return module


def routing_source(filename):
    return REFERENCE / "upstream/dgn_tf/Routing" / filename


def extract(source):
    """Copy exact Keras import lines and one function, never module side effects."""
    tree, lines = ast.parse(source), source.splitlines(keepends=True)
    selected = [node for node in tree.body if
                (isinstance(node, ast.ImportFrom) and node.module.startswith("keras")) or
                (isinstance(node, ast.FunctionDef) and node.name == "MultiHeadsAttModel")]
    if sum(isinstance(node, ast.FunctionDef) for node in selected) != 1:
        raise ValueError("Expected exactly one upstream attention definition")
    return "".join("".join(lines[node.lineno - 1:node.end_lineno]) for node in selected)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-seconds", type=int, default=300)
    args = parser.parse_args(argv)
    if not 0 < args.max_seconds <= 300:
        parser.error("--max-seconds must be 1..300 total for both variants")
    return args


def installed_packages(search_paths):
    """Read wheel metadata on the child's actual import path, without subprocesses."""
    packages = {}
    for directory in dict.fromkeys(search_paths):
        path = Path(directory)
        if not path.is_dir():
            continue
        for metadata in sorted(path.glob("*.dist-info/METADATA")):
            message = Parser().parsestr(metadata.read_text(encoding="utf-8"))
            name, version = message["Name"], message["Version"]
            if not name or not version or name.lower() in packages:
                raise ValueError("Missing/ambiguous package metadata: " + str(metadata))
            packages[name.lower()] = {"name": name, "version": version}
    return [packages[name] for name in sorted(packages)]


def native_run(variant, directory):
    # This function is reachable only in a separate user-invoked child.
    output = Path(directory)
    summary = {"variant": variant, "status": "STARTED", "stage": "RUNTIME_GUARD",
               "python": sys.version, "executable": sys.executable,
               "native_imports_started": False, "device": "cpu", "fixture_seed": 0,
               "heads": 8, "neighbors": 4, "batch_size": 1, "forward_batches": 0,
               "optimizer_steps": 0, "environment": "NOT_RUN", "training": "NOT_RUN"}

    def save():
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    try:
        save()
        if sys.version_info[:2] != (3, 7) or struct.calcsize("P") != 8 or not __debug__:
            raise RuntimeError("Python 3.7 x64 required, with assertions enabled")
        if sys.prefix == sys.base_prefix or not sys.flags.isolated:
            raise RuntimeError("Isolated venv required; use the supervisor and setup runtime")
        configuration = (Path(sys.prefix) / "pyvenv.cfg").read_text(encoding="utf-8")
        if "include-system-site-packages = false" not in configuration.lower():
            raise RuntimeError("Venv must exclude system site packages")
        summary["venv_prefix"] = sys.prefix
        summary["base_prefix"] = sys.base_prefix
        summary["isolated"] = bool(sys.flags.isolated)
        summary["stage"] = "PACKAGE_METADATA"
        save()
        installed = installed_packages(sys.path)
        summary["installed_packages"] = installed
        versions = {row["name"].lower(): row["version"] for row in installed}
        for name, expected in dict(PINS, h5py="2.10.0", protobuf="3.20.3").items():
            if versions.get(name) != expected:
                raise RuntimeError("Declared package mismatch: " + name + " expected " + expected)
        summary["stage"], summary["native_imports_started"] = "IMPORTS", True
        save()
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        os.environ["KERAS_BACKEND"] = "tensorflow"
        import numpy as np
        import tensorflow as tf
        namespace = {"np": np, "tf": tf}
        # Contains imports/one original definition only, generated by the supervisor.
        source = (output / "attention_extract.py").read_text(encoding="utf-8")
        exec(compile(source, str(output / "attention_extract.py"), "exec"), namespace)
        if tf.config.get_visible_devices("GPU"):
            raise RuntimeError("GPU visible despite CPU-only request")
        summary["runtime_versions"] = {"numpy": np.__version__, "tensorflow": tf.__version__}
        np.random.seed(0)
        tf.random.set_seed(0)
        summary["stage"] = "BUILD"
        save()
        with tf.device("/CPU:0"):
            built = namespace["MultiHeadsAttModel"](l=4, d=128, dv=16, dout=128, nv=8)
            if variant == "regularized":
                model, attention = built
                model = namespace["Model"](inputs=model.inputs, outputs=[model.output, attention.output])
            else:
                model = built
            summary["output_shape"] = model.output_shape
            summary["stage"] = "FORWARD"
            save()
            values = np.linspace(-1, 1, 512, dtype="float32").reshape(1, 4, 128)
            selector = np.array([[[1, 0, 0, 0]]], dtype="float32")
            prediction = model.predict([values, values.copy(), values.copy(), selector], batch_size=1, verbose=0)
            summary["forward_batches"] = 1
            tensors = prediction if isinstance(prediction, list) else [prediction]
            shapes = [list(tensor.shape) for tensor in tensors]
            expected = [[1, 1, 128], [1, 8, 4, 4]] if variant == "regularized" else [[1, 1, 128]]
            if shapes != expected or not all(np.isfinite(tensor).all() for tensor in tensors):
                raise RuntimeError("Unexpected/nonfinite attention outputs: " + repr(shapes))
            if variant == "regularized":
                np.testing.assert_allclose(tensors[1].sum(axis=-1), 1, rtol=1e-5, atol=1e-5)
            summary.update(status="COMPLETED_PENDING_USER_REVIEW", stage="FINISHED",
                           result_shapes=shapes, finite=True)
    except Exception:
        summary.update(status="FAILED", error=traceback.format_exc())
        raise
    finally:
        save()


def main(argv=None):
    args = parse_args(argv)
    if sys.version_info[:2] != (3, 12) or not __debug__:
        raise RuntimeError("Supervisor requires Python 3.12 with assertions enabled")
    runtime = args.runtime.resolve(strict=True)
    control = controls()
    audit = control.check_identity()  # Existing immutable reference audit, no native imports.
    output = control.new_output(args.output, ROOT / "user_routing_checks")
    inputs = [Path(__file__), ROOT / "setup_routing_runtime.ps1", ROOT / "routing_constraints.txt",
              REFERENCE / "manifest.json", REFERENCE / "check_reference.py",
              ROOT / "compat/torch_surviving_cpu_v1/main.py", runtime]
    hashes_before = {str(path): control.sha(path) for path in inputs}
    receipt = {"status": "STARTED", "scope": "THROWAWAY_UPSTREAM_ATTENTION_RUNTIME_PROBE_ONLY",
               "execution_owner": "USER", "output": str(output), "runtime": str(runtime),
               "supervisor_python": sys.version, "max_seconds_total": args.max_seconds,
               "started_at_utc": datetime.now(timezone.utc).isoformat(), "hashes_before": hashes_before,
               "device": "cpu", "fixture_seed": 0, "optimizer_steps": 0,
               "environment": "NOT_RUN", "training": "NOT_RUN", "ns3": "NOT_RUN",
               "scientific_gates": "CLOSED", "paper_replication": "NOT_ESTABLISHED", "variants": []}
    control.write_json(output / "source_before.json", audit)
    control.write_json(output / "run_receipt.json", receipt)
    deadline = time.monotonic() + args.max_seconds
    exit_code = 1
    try:
        for variant, filename in VARIANTS:
            child_output = control.new_output(output / variant, output)
            original = routing_source(filename)
            text = extract(original.read_text(encoding="utf-8-sig"))
            (child_output / "attention_extract.py").write_text(text, encoding="utf-8")
            child = ("import importlib.util; s=importlib.util.spec_from_file_location('probe', "
                     + repr(str(Path(__file__).resolve()))
                     + "); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); m.native_run("
                     + repr(variant) + ", " + repr(str(child_output)) + ")")
            command = [str(runtime), "-I", "-B", "-u", "-c", child]
            remaining = deadline - time.monotonic()
            status, code = control.run_child(command, child_output, remaining) if remaining > 0 else ("NOT_STARTED_DEADLINE", 124)
            row = {"variant": variant, "process_status": status, "exit_code": code,
                   "command": command, "upstream_sha256": control.sha(original),
                   "extract_sha256": control.sha(child_output / "attention_extract.py")}
            if (child_output / "summary.json").exists():
                row["summary"] = json.loads((child_output / "summary.json").read_text(encoding="utf-8"))
            if code == 0:
                summary = row.get("summary", {})
                if (summary.get("status") != "COMPLETED_PENDING_USER_REVIEW" or
                        summary.get("forward_batches") != 1 or summary.get("optimizer_steps") != 0 or
                        summary.get("device") != "cpu"):
                    raise ValueError("Child success lacks approved probe evidence")
            receipt["variants"].append(row)
            control.write_json(output / "run_receipt.json", receipt)
        codes = [row["exit_code"] for row in receipt["variants"]]
        exit_code = 124 if 124 in codes else (0 if codes == [0, 0] else 1)
        receipt["status"] = "COMPLETED_PENDING_USER_REVIEW" if exit_code == 0 else ("TIMEOUT" if exit_code == 124 else "FAILED_PENDING_USER_REVIEW")
    except KeyboardInterrupt:
        receipt["status"], exit_code = "INTERRUPTED", 130
    except Exception:
        receipt["status"], exit_code = "SUPERVISOR_FAILED", 1
        receipt["supervisor_error"] = traceback.format_exc()
    finally:
        try:
            control.write_json(output / "source_after.json", control.check_identity())
            receipt["hashes_after"] = {str(path): control.sha(path) for path in inputs}
            if receipt["hashes_after"] != hashes_before:
                raise ValueError("Probe/helper/runtime identity changed during execution")
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
