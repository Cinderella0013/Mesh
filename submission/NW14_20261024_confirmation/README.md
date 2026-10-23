# NW14 retained offline confirmation

This update does **not** add native simulations, training, builds or SAT results. Original scientific source snapshot: `f5921db0d72edc70e0c6a5163d4f669ed8eb43c5`. The adjacent `NW14_20261023` directory retains the six-method, 360-cell benchmark export. Normal remains separate from primary stress results. The six-method presentation was selected after results were seen; it is not a ranking of every originally tested policy.

## Reproduce on supplied observations

Use Python 3.9+ with NumPy. No PyTorch/model inference or simulator is invoked. From this directory, extract the supplied ZIP (standard library, no extra dependency):

```sh
python -m zipfile -e retained_inputs_120.zip .
python portable_replay/test_exact_planner.py
python portable_replay/replay_exact_retained.py
python q/test_q_margins.py
python q/q_margins.py --registry portable_replay/REGISTRY_120.csv
python gate/audit_gate.py --registry portable_replay/REGISTRY_120.csv --source-dir native_source
```

Expected: 18 planner tests; exact replay of Hybrid 60 + Q-neutral 60 runs (1,560 epochs/23,400 choices), no issues and zero pressure-update discrepancy. Four Q-analysis tests pass. Q analysis selects only the 60 Hybrid runs: primary local 138/9,750 versus whole frozen-epoch 220/9,750 changed choices. Normal local 27/1,950 and epoch 23/1,950 are separate. Gate audit reconstructs 11,700 requests and 1,301 held cases, previous REJECTED 1,251 + STALE 50. Read `REPORT_TH.md`, `gate/FINDINGS_TH.md` and JSON limitations before interpretation. Tested environment: Python 3.12.14 and NumPy 2.3.5.

`q` holds the actual reservation prefix fixed for its local comparison. The separate epoch comparison recomputes the prefix inside that single supplied observation. Neither produces a closed-loop counterfactual trajectory or SAT. Q tests include equal-pressure stay, alternative-channel influence, gate override and rejection of an inconsistent recorded action.

The original RAES SHA256 is `cfea06e17708e74422fc35606d1a2fd1c88c1078d3de04c7d03427bbeae1854a`. Only harness input/source paths were relocated for portable planner replay; expected outputs and planner arithmetic are unchanged. `native_source` contains pinned historical native source for readback only, not a complete clean-build project. `gate/source_hash_line_map.json` supplies source lines; raw CRLF versus LF hashes are documented.

The ZIP contains 1,080 original per-run input members (68,873,478 bytes) plus registry, exact planner source, harnesses and manifests. Copied input bytes were checked against historical originals. Receipt strings keep original machine paths as provenance; commands above use the relative exported layout. Outputs write beside each harness. Run without `python -O` because the audits deliberately use assertions to fail on inconsistent evidence.

This is author-run offline verification, not independently authenticated external replication. It does not establish hardware behavior, fault paths absent from the logs, a full native rebuild, new training, statistical superiority, novel GAT architecture or fresh generalization. Gate terminality refers to permanent policy requests, not scan retunes; removing gate has not been service-tested. All previous service numbers and negative timing evidence remain unchanged.

Drive: the user reported 10 files uploaded. Browser file inventory and Viewer permissions could not be independently checked. This public incremental package is separate from that earlier upload, and does not silently replace it.

`MANIFEST_SHA256.json` lists this curated snapshot, including the compressed retained inputs. The reviewed paper is under `paper/`; its six tables and four native equations are preserved. Adviser response TXT and private reviewer annotations are delivered locally, not published in this repository.
