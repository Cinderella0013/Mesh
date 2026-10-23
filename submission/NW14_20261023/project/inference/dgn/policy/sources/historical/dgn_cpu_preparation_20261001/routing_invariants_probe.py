"""Bounded repaired online model invariants/gradients; no optimizer updates."""
import hashlib
import math
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
spec = importlib.util.spec_from_file_location("full_online", ROOT / "routing_full_model_probe.py")
full = importlib.util.module_from_spec(spec)
spec.loader.exec_module(full)
repair, probe = full.repair, full.probe
PERMUTATION = [(i * 7 + 3) % 20 for i in range(20)]  # new index -> original index
ATOL, RTOL = 1e-6, 1e-5


def fixture_neighbors(order=None):
    order = list(range(20)) if order is None else list(order)
    if sorted(order) != list(range(20)):
        raise ValueError("Permutation must contain each agent exactly once")
    inverse = {old: new for new, old in enumerate(order)}
    return [[inverse[(old // 10) * 10 + (old % 10 + slot) % 10] for slot in range(4)]
            for old in order]


def check_q_values(base, permuted, perturbed):
    for q in (base, permuted, perturbed):
        if len(q) != 1 or len(q[0]) != 20 or any(len(row) != 4 for row in q[0]):
            raise ValueError("Expected Q shape [1,20,4]")
        if not all(math.isfinite(value) for row in q[0] for value in row):
            raise ValueError("Nonfinite Q value")
    pairs = [(permuted[0][new][a], base[0][old][a]) for new, old in enumerate(PERMUTATION) for a in range(4)]
    isolated = [(perturbed[0][i][a], base[0][i][a]) for i in range(10) for a in range(4)]
    close = lambda x, y: math.isclose(x, y, rel_tol=RTOL, abs_tol=ATOL)
    result = {"atol": ATOL, "rtol": RTOL,
              "permutation_pass": all(close(x, y) for x, y in pairs),
              "permutation_max_abs_error": max(abs(x-y) for x, y in pairs),
              "isolation_pass": all(close(x, y) for x, y in isolated),
              "isolation_max_abs_error": max(abs(x-y) for x, y in isolated),
              "affected_group_changed": any(not close(perturbed[0][i][a], base[0][i][a]) for i in range(10,20) for a in range(4))}
    if not all(result[name] for name in ("permutation_pass", "isolation_pass", "affected_group_changed")):
        raise ValueError("Q invariant failed: " + json.dumps(result))
    return result


def weight_digest(weights):
    digest = hashlib.sha256()
    for array in weights:
        digest.update(str((array.shape, str(array.dtype))).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def native_run(variant, directory):
    output = Path(directory)
    summary = {"variant": variant, "implementation": "PER_HEAD_REPAIRED_ONLINE_GRAPH",
               "status": "STARTED", "stage": "RUNTIME_GUARD", "python": sys.version,
               "agents": 20, "features": 35, "neighbors": 4, "heads": 8, "batch_size": 1,
               "device": "cpu", "fixture_seed": 0, "forward_batches": 0, "backward_evaluations": 0, "optimizer_steps": 0,
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
            weights_before = model.get_weights()
            np.savez(output / "weights_before.npz", *weights_before)
            summary["weights_sha256_before"] = weight_digest(weights_before)
            observations = np.linspace(-1, 1, 700, dtype="float32").reshape(1, 20, 35)
            selector = namespace["vec"].astype("float32")[None, :, :]
            predictions = {}
            for fixture in ("base", "permuted", "perturbed"):
                order = PERMUTATION if fixture == "permuted" else list(range(20))
                obs = observations[:, order, :].copy()
                if fixture == "perturbed":
                    obs[:, 10:, :] = -3 * obs[:, 10:, :] + 2
                inputs, masks = [], []
                for agent, neighbors in enumerate(fixture_neighbors(order)):
                    mask = np.zeros((1, 4, 20), dtype="float32")
                    for slot, neighbor in enumerate(neighbors):
                        mask[0, slot, neighbor] = 1
                    masks.append(mask)
                    inputs.extend([obs[:, agent, :], mask])
                inputs.append(selector)
                np.savez(output / (fixture + "_fixture.npz"), observations=obs,
                         gather_masks=np.stack(masks, axis=1), selector=selector,
                         order=np.array(order, dtype="int32"))
                if fixture == "base":
                    base_inputs = inputs
                summary["stage"] = "FORWARD_" + fixture.upper()
                summary["forward_batches"] += 1
                save()
                q = np.stack(model.predict(inputs, batch_size=1, verbose=0), axis=1)
                predictions[fixture] = q.tolist()
                (output / (fixture + "_q_values.json")).write_text(json.dumps(q.tolist()) + "\n", encoding="utf-8")
            summary["invariants"] = check_q_values(predictions["base"], predictions["permuted"], predictions["perturbed"])
            summary["stage"] = "GRADIENT_BUILD"
            summary["gradient_loss"] = "mean(square(concatenate(online Q outputs))) synthetic only"
            save()
            K = namespace["K"]
            weights = model.trainable_weights
            with model.outputs[0].graph.as_default():
                loss = K.mean(K.square(K.concatenate(model.outputs, axis=1)))
                gradients = K.gradients(loss, weights)
                summary["gradient_tensors"] = len(gradients)
                summary["connected_gradient_tensors"] = sum(grad is not None for grad in gradients)
                save()
                if not gradients or any(grad is None for grad in gradients):
                    raise ValueError("Missing/disconnected trainable gradient")
                evaluate = K.function(model.inputs, [loss] + gradients, updates=None)
            summary["stage"] = "BACKWARD"
            summary["forward_batches"] += 1  # gradient evaluation also computes a forward loss
            summary["backward_evaluations"] += 1
            save()
            values = evaluate(base_inputs)
            gradient_values = values[1:]
            summary["gradient_loss_value"] = float(values[0])
            if not np.isfinite(values[0]):
                raise ValueError("Nonfinite synthetic loss")
            np.savez(output / "gradients.npz", *gradient_values)
            gradient_rows = [{"index": i, "name": weight.name, "shape": list(value.shape),
                              "finite": bool(np.isfinite(value).all()), "nonzero_values": int(np.count_nonzero(value)),
                              "max_abs": float(np.max(np.abs(value))),
                              "l2": float(np.sqrt(np.sum(value.astype("float64") ** 2)))}
                             for i, (weight, value) in enumerate(zip(weights, gradient_values))]
            (output / "gradients.json").write_text(json.dumps(gradient_rows, indent=2) + "\n", encoding="utf-8")
            if len(gradient_values) != len(weights) or any(value.shape != tuple(K.int_shape(weight)) for weight, value in zip(weights, gradient_values)):
                raise ValueError("Gradient tensor shape/count mismatch")
            if not all(row["finite"] for row in gradient_rows):
                raise ValueError("Nonfinite gradient")
            summary["gradient_blocks"] = {}
            for component in ("encoder", "m1", "m2", "q_net"):
                indices = [i for i, weight in enumerate(weights) if any(weight is member for member in namespace[component].trainable_weights)]
                if not indices or not any(gradient_rows[i]["nonzero_values"] for i in indices):
                    raise ValueError("No nonzero gradient in " + component)
                summary["gradient_blocks"][component] = {"indices": indices, "finite": True, "nonzero": True}
            weights_after = model.get_weights()
            np.savez(output / "weights_after.npz", *weights_after)
            summary["weights_sha256_after"] = weight_digest(weights_after)
            if summary["weights_sha256_before"] != summary["weights_sha256_after"]:
                raise ValueError("Weights changed without optimizer authorization")
            if summary["forward_batches"] != 4 or summary["backward_evaluations"] != 1:
                raise ValueError("Execution budget mismatch")
            summary.update(status="COMPLETED_PENDING_REVIEW", stage="FINISHED", q_shape=[1,20,4],
                           finite=True, gradients_finite=True, weights_unchanged=True)
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
    output = control.new_output(args.output, ROOT / "routing_invariants_checks")
    inputs = [Path(__file__).resolve(), ROOT / "check_routing_invariants_probe.py", ROOT / "routing_full_model_probe.py", ROOT / "routing_repair_probe.py",
              repair.ORIGINAL_PROBE, ROOT / "routing_constraints.txt", ROOT / "compat/torch_surviving_cpu_v1/main.py",
              probe.REFERENCE / "check_reference.py", probe.REFERENCE / "manifest.json", runtime]
    hashes = {str(path): control.sha(path) for path in inputs}
    receipt = {"status": "STARTED", "scope": "PER_HEAD_REPAIRED_ONLINE_GRAPH_INVARIANTS_GRADIENTS_NO_UPDATES",
               "execution_owner": "ASSISTANT_USER_AUTHORIZED", "authorization": "yes approve, invariants/gradients 4 forward + 1 backward per variant, 3 October 2026",
               "output": str(output), "runtime": str(runtime), "hashes_before": hashes,
               "max_seconds_total": args.max_seconds, "started_at_utc": datetime.now(timezone.utc).isoformat(),
               "agents": 20, "features": 35, "neighbors": 4, "heads": 8, "batch_size": 1,
               "optimizer_steps": 0, "device": "cpu", "fixture_seed": 0,
               "forward_budget_per_variant": 4, "backward_budget_per_variant": 1, "target_graph": "NOT_RUN",
               "regularized_aggregation": "UPSTREAM_RAW_SCORES_PRESERVED", "environment": "NOT_RUN",
               "training": "NOT_RUN", "ns3": "NOT_RUN", "scientific_gates": "CLOSED", "variants": []}
    control.write_json(output / "source_before.json", audit)
    control.write_json(output / "run_receipt.json", receipt)
    deadline, exit_code = time.monotonic() + args.max_seconds, 1
    try:
        for variant, filename in probe.VARIANTS:
            directory = control.new_output(output / variant, output)
            upstream = probe.routing_source(filename)
            original, graph = full.extract_online(upstream.read_text(encoding="utf-8-sig"))
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
                if s.get("status") != "COMPLETED_PENDING_REVIEW" or s.get("q_shape") != [1, 20, 4] or s.get("forward_batches") != 4 or s.get("backward_evaluations") != 1 or s.get("optimizer_steps") != 0 or not s.get("finite") or not s.get("gradients_finite") or not s.get("weights_unchanged"):
                    raise ValueError("Invariants/gradient child lacks approved evidence")
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
