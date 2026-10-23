"""Bounded repaired online/target diagnostic: original soft update plus hard-copy control."""
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
import textwrap

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("invariants_helpers", ROOT / "routing_invariants_probe.py")
invariants = importlib.util.module_from_spec(spec)
spec.loader.exec_module(invariants)
full, repair, probe = invariants.full, invariants.repair, invariants.probe
COMPONENTS = ("encoder", "q_net", "m1", "m2")


def extract_target_and_sync(source):
    tree, lines = ast.parse(source), source.splitlines(keepends=True)
    starts = [n for n in tree.body if full.assignment(n, "encoder_t")]
    ends = [n for n in tree.body if full.assignment(n, "model_t")]
    taus = [n for n in tree.body if full.assignment(n, "TAU")]
    if len(starts) != 1 or len(ends) != 1 or starts[0].lineno >= ends[0].lineno or len(taus) != 1 or ast.literal_eval(taus[0].value) != 0.01:
        raise ValueError("Missing/ambiguous target boundaries or changed TAU")
    graph = "".join(lines[starts[0].lineno-1:ends[0].end_lineno])
    if any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in
           ("compile", "fit", "train_on_batch", "apply_gradients", "minimize", "set_weights") for n in ast.walk(ast.parse(graph))):
        raise ValueError("Training/update call inside target graph")
    expected = "".join('weights = {0}.get_weights()\ntarget_weights = {0}_t.get_weights()\n'
                       'for w in range(len(weights)):\n    target_weights[w] = TAU * weights[w] + (1 - TAU)* target_weights[w]\n'
                       '{0}_t.set_weights(target_weights)\n'.format(name) for name in COMPONENTS)
    expected_nodes = ast.parse(expected).body
    matches = []
    for parent in ast.walk(tree):
        body = getattr(parent, "body", None)
        if not isinstance(body, list):
            continue
        for i in range(len(body)-len(expected_nodes)+1):
            segment = body[i:i+len(expected_nodes)]
            if [ast.dump(n) for n in segment] == [ast.dump(n) for n in expected_nodes]:
                matches.append(segment)
    if len(matches) != 1:
        raise ValueError("Original four-component soft-update block missing/ambiguous/changed")
    segment = matches[0]
    update = textwrap.dedent("".join(lines[segment[0].lineno-1:segment[-1].end_lineno]))
    tau = "".join(lines[taus[0].lineno-1:taus[0].end_lineno])
    return graph, update, tau


def native_run(variant, directory):
    output = Path(directory)
    summary = {"variant": variant, "implementation": "PER_HEAD_REPAIRED_ONLINE_TARGET_GRAPHS",
               "status": "STARTED", "stage": "RUNTIME_GUARD", "python": sys.version,
               "agents": 20, "features": 35, "neighbors": 4, "heads": 8, "batch_size": 1,
               "device": "cpu", "fixture_seed": 0, "forward_batches": 0, "backward_evaluations": 0, "optimizer_steps": 0,
               "target_soft_updates": 0, "hard_copy_controls": 0,
               "environment": "NOT_RUN", "training": "NOT_RUN", "target_graph": "NOT_BUILT_YET"}

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
            target_assembly = output / "target_graph.py"
            exec(compile(target_assembly.read_text(encoding="utf-8"), str(target_assembly), "exec"), namespace)
            exec((output / "tau.py").read_text(encoding="utf-8"), namespace)
            model, target = namespace["model"], namespace["model_t"]
            summary.update(target_graph="BUILT", input_count=len(model.inputs), target_input_count=len(target.inputs),
                           output_count=len(model.outputs), target_output_count=len(target.outputs), tau=namespace["TAU"])
            if len(model.inputs) != 41 or len(target.inputs) != 41 or len(model.outputs) != 20 or len(target.outputs) != 20:
                raise ValueError("Expected 41 inputs and 20 outputs per graph")
            online_ids, target_ids = [id(w) for w in model.weights], [id(w) for w in target.weights]
            if len(set(online_ids)) != 22 or len(set(target_ids)) != 22 or set(online_ids) & set(target_ids):
                raise ValueError("Online/target variables shared or unexpected tensor count")
            summary["independent_variables"] = True
            pairs = []
            for component in COMPONENTS:
                a, b = namespace[component].weights, namespace[component+"_t"].weights
                if len(a) != len(b):
                    raise ValueError("Component tensor count differs")
                for ow, tw in zip(a, b):
                    if tuple(ow.shape) != tuple(tw.shape):
                        raise ValueError("Component tensor shape differs")
                    pairs.append({"component":component,"online_index":online_ids.index(id(ow)),
                                  "target_index":target_ids.index(id(tw)),"online_name":ow.name,"target_name":tw.name})
            if sorted(p["online_index"] for p in pairs) != list(range(22)) or sorted(p["target_index"] for p in pairs) != list(range(22)):
                raise ValueError("Incomplete/overlapping component weight coverage")
            (output / "weight_pairs.json").write_text(json.dumps(pairs,indent=2)+"\n",encoding="utf-8")
            online_before, target_before = model.get_weights(), target.get_weights()
            np.savez(output / "online_before.npz", *online_before)
            np.savez(output / "target_before.npz", *target_before)
            summary["online_sha256_before"] = invariants.weight_digest(online_before)
            if not any(not np.array_equal(online_before[p["online_index"]],target_before[p["target_index"]]) for p in pairs):
                raise ValueError("Initial weights identical; sync fixture is vacuous")
            summary["initial_weights_differ"] = True
            observations = np.linspace(-1, 1, 700, dtype="float32").reshape(1,20,35)
            inputs, masks = [], []
            for agent in range(20):
                mask = np.zeros((1,4,20),dtype="float32")
                for slot, neighbor in enumerate(full.neighbor_indices(agent)):
                    mask[0,slot,neighbor] = 1
                masks.append(mask)
                inputs.extend([observations[:,agent,:],mask])
            selector = namespace["vec"].astype("float32")[None,:,:]
            inputs.append(selector)
            np.savez(output / "fixture.npz", observations=observations,gather_masks=np.stack(masks,axis=1),selector=selector)

            def predict(label, graph):
                summary["stage"] = "FORWARD_" + label.upper()
                summary["forward_batches"] += 1
                save()
                q = np.stack(graph.predict(inputs,batch_size=1,verbose=0),axis=1)
                (output / (label+"_q_values.json")).write_text(json.dumps(q.tolist())+"\n",encoding="utf-8")
                if q.shape != (1,20,4) or not np.isfinite(q).all():
                    raise ValueError("Nonfinite/unexpected Q values: " + label)
                return q

            online_q = predict("online_before",model)
            predict("target_before",target)
            summary["stage"] = "SOFT_UPDATE"
            summary["target_soft_updates"] += 1
            save()
            update = output / "soft_update.py"
            exec(compile(update.read_text(encoding="utf-8"),str(update),"exec"),namespace)
            target_soft = target.get_weights()
            np.savez(output / "target_soft.npz", *target_soft)
            errors = []
            for pair in pairs:
                oi, ti = pair["online_index"],pair["target_index"]
                expected = namespace["TAU"]*online_before[oi]+(1-namespace["TAU"])*target_before[ti]
                errors.append(float(np.max(np.abs(expected-target_soft[ti]))))
            summary["soft_update_max_abs_error"] = max(errors)
            if max(errors) != 0 or not any(not np.array_equal(a,b) for a,b in zip(target_before,target_soft)):
                raise ValueError("Original soft update mismatched or did not change target")
            summary["soft_formula_match"] = True
            summary["online_sha256_after_soft"] = invariants.weight_digest(model.get_weights())
            if summary["online_sha256_after_soft"] != summary["online_sha256_before"]:
                raise ValueError("Soft update changed online weights")
            predict("target_soft",target)
            summary["stage"] = "HARD_COPY_CONTROL"
            summary["hard_copy_controls"] += 1
            save()
            for component in COMPONENTS:
                namespace[component+"_t"].set_weights(namespace[component].get_weights())
            target_hard = target.get_weights()
            np.savez(output / "target_hard.npz", *target_hard)
            if any(not np.array_equal(online_before[p["online_index"]],target_hard[p["target_index"]]) for p in pairs):
                raise ValueError("Hard-copy control tensor mismatch")
            hard_q = predict("target_hard",target)
            summary["hard_copy_q_max_abs_error"] = float(np.max(np.abs(online_q-hard_q)))
            if not np.allclose(online_q,hard_q,rtol=1e-5,atol=1e-6):
                raise ValueError("Hard-copy target Q differs from online")
            online_after = model.get_weights()
            np.savez(output / "online_after.npz", *online_after)
            summary["online_sha256_after"] = invariants.weight_digest(online_after)
            if summary["online_sha256_after"] != summary["online_sha256_before"]:
                raise ValueError("Sync control changed online weights")
            if summary["forward_batches"] != 4 or summary["target_soft_updates"] != 1 or summary["hard_copy_controls"] != 1:
                raise ValueError("Execution budget mismatch")
            summary.update(status="COMPLETED_PENDING_REVIEW",stage="FINISHED",q_shape=[1,20,4],
                           finite=True,online_unchanged=True,hard_copy_equal=True,target_graph="FORWARD_VERIFIED")
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
    output = control.new_output(args.output, ROOT / "routing_target_checks")
    inputs = [Path(__file__).resolve(), ROOT / "check_routing_target_probe.py", ROOT / "routing_invariants_probe.py", ROOT / "routing_full_model_probe.py", ROOT / "routing_repair_probe.py",
              repair.ORIGINAL_PROBE, ROOT / "routing_constraints.txt", ROOT / "compat/torch_surviving_cpu_v1/main.py",
              probe.REFERENCE / "check_reference.py", probe.REFERENCE / "manifest.json", runtime]
    hashes = {str(path): control.sha(path) for path in inputs}
    receipt = {"status": "STARTED", "scope": "PER_HEAD_REPAIRED_TARGET_GRAPH_SOFT_SYNC_HARD_COPY_CONTROL",
               "execution_owner": "ASSISTANT_USER_AUTHORIZED", "authorization": "yes approve, target/sync 4 forward, 0 backward/optimizer per variant, 3 October 2026",
               "output": str(output), "runtime": str(runtime), "hashes_before": hashes,
               "max_seconds_total": args.max_seconds, "started_at_utc": datetime.now(timezone.utc).isoformat(),
               "agents": 20, "features": 35, "neighbors": 4, "heads": 8, "batch_size": 1,
               "optimizer_steps": 0, "device": "cpu", "fixture_seed": 0,
               "forward_budget_per_variant":4, "backward_budget_per_variant":0, "soft_updates_per_variant":1,
               "hard_copy_controls_per_variant":1, "tau":0.01,
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
            target_graph, soft_update, tau = extract_target_and_sync(upstream.read_text(encoding="utf-8-sig"))
            repaired = repair.repair_extract(original)
            for name, text in (("original_model_definitions.py", original), ("model_definitions.py", repaired), ("online_graph.py", graph), ("target_graph.py",target_graph), ("soft_update.py",soft_update), ("tau.py",tau)):
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
                   "extract_sha256": {name: control.sha(directory / name) for name in ("original_model_definitions.py", "model_definitions.py", "online_graph.py", "target_graph.py", "soft_update.py", "tau.py", "repair.patch")}}
            if (directory / "summary.json").exists():
                row["summary"] = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
            if code == 0:
                s = row.get("summary", {})
                if s.get("status") != "COMPLETED_PENDING_REVIEW" or s.get("q_shape") != [1, 20, 4] or s.get("forward_batches") != 4 or s.get("backward_evaluations") != 0 or s.get("optimizer_steps") != 0 or not s.get("finite") or not s.get("online_unchanged") or not s.get("hard_copy_equal") or not s.get("soft_formula_match") or not s.get("independent_variables") or s.get("target_soft_updates") != 1 or s.get("hard_copy_controls") != 1:
                    raise ValueError("Target/sync child lacks approved evidence")
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
