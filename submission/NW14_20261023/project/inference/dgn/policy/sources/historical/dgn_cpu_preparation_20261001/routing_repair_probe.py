"""Separate throwaway per-head repair diagnostic, run with user authorization.

Reuses the immutable original native probe; changes exactly two contractions.
Keeps regularized raw-score aggregation. One forward batch/variant, no updates.
"""
from datetime import datetime, timezone
import difflib
import importlib.util
import json
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
ORIGINAL_PROBE = ROOT / "routing_probe.py"
spec = importlib.util.spec_from_file_location("original_routing_probe", ORIGINAL_PROBE)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)  # Stdlib definitions only.


def repair_extract(original):
    substitutions = (
        ("K.batch_dot(x[0],x[1] ,axes=[3,3])", "tf.matmul(x[0], x[1], transpose_b=True)"),
        ("K.batch_dot(x[0], x[1],axes=[3,2])", "tf.matmul(x[0], x[1])"),
    )
    for old, new in substitutions:
        if original.count(old) != 1:
            raise ValueError("Expected one immutable upstream contraction: " + old)
        original = original.replace(old, new)
    return original


def main(argv=None):
    args = probe.parse_args(argv)
    if sys.version_info[:2] != (3, 12) or not __debug__:
        raise RuntimeError("Supervisor requires Python3.12 with assertions enabled")
    control = probe.controls()
    if control.sha(ORIGINAL_PROBE) != "06d2aed1b91c7af3ac16ad5229f1251cea9989f3d652f52565b13c087d194332":
        raise ValueError("Original probe changed; preserve previous source/evidence")
    runtime = args.runtime.resolve(strict=True)
    audit = control.check_identity()
    output = control.new_output(args.output, ROOT / "routing_repair_checks")
    inputs = [Path(__file__).resolve(), ROOT / "check_routing_repair_probe.py", ORIGINAL_PROBE,
              ROOT / "routing_constraints.txt", ROOT / "compat/torch_surviving_cpu_v1/main.py",
              probe.REFERENCE / "check_reference.py", probe.REFERENCE / "manifest.json", runtime]
    hashes_before = {str(path): control.sha(path) for path in inputs}
    receipt = {
        "status": "STARTED", "scope": "PER_HEAD_CONTRACTION_REPAIR_DIAGNOSTIC_ONLY",
        "execution_owner": "ASSISTANT_USER_AUTHORIZED", "output": str(output), "runtime": str(runtime),
        "authorization": "User requested next task and explicitly authorized assistant execution in this chat",
        "supervisor_python": sys.version, "hashes_before": hashes_before,
        "started_at_utc": datetime.now(timezone.utc).isoformat(), "max_seconds_total": args.max_seconds,
        "heads": 8, "neighbors": 4, "batch_size": 1, "device": "cpu", "fixture_seed": 0,
        "optimizer_steps": 0, "regularized_aggregation": "UPSTREAM_RAW_SCORES_PRESERVED",
        "environment": "NOT_RUN", "training": "NOT_RUN", "ns3": "NOT_RUN",
        "scientific_gates": "CLOSED", "paper_replication": "NOT_ESTABLISHED", "variants": [],
    }
    control.write_json(output / "source_before.json", audit)
    control.write_json(output / "run_receipt.json", receipt)
    deadline, exit_code = time.monotonic() + args.max_seconds, 1
    try:
        for variant, filename in probe.VARIANTS:
            directory = control.new_output(output / variant, output)
            source = probe.routing_source(filename)
            original = probe.extract(source.read_text(encoding="utf-8-sig"))
            repaired = repair_extract(original)
            (directory / "original_attention_extract.py").write_text(original, encoding="utf-8")
            (directory / "attention_extract.py").write_text(repaired, encoding="utf-8")
            (directory / "repair.patch").write_text("".join(difflib.unified_diff(
                original.splitlines(keepends=True), repaired.splitlines(keepends=True),
                fromfile="original_attention_extract.py", tofile="attention_extract.py")), encoding="utf-8")
            child = ("import importlib.util; s=importlib.util.spec_from_file_location('probe', "
                     + repr(str(ORIGINAL_PROBE))
                     + "); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); m.native_run("
                     + repr(variant) + ", " + repr(str(directory)) + ")")
            command = [str(runtime), "-I", "-B", "-u", "-c", child]
            remaining = deadline - time.monotonic()
            status, code = control.run_child(command, directory, remaining) if remaining > 0 else ("NOT_STARTED_DEADLINE", 124)
            row = {"variant": variant, "implementation": "PER_HEAD_CONTRACTION_REPAIR",
                   "process_status": status, "exit_code": code, "command": command,
                   "upstream_sha256": control.sha(source),
                   "original_extract_sha256": control.sha(directory / "original_attention_extract.py"),
                   "repaired_extract_sha256": control.sha(directory / "attention_extract.py")}
            if (directory / "summary.json").exists():
                row["summary"] = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
            if code == 0:
                summary = row.get("summary", {})
                if (summary.get("status") != "COMPLETED_PENDING_USER_REVIEW" or
                        summary.get("forward_batches") != 1 or summary.get("optimizer_steps") != 0 or
                        summary.get("device") != "cpu" or not summary.get("finite")):
                    raise ValueError("Repair child lacks approved budget/finite evidence")
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
            if hashes_before != receipt["hashes_after"]:
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
