# NW14 GAT+RAES — advisor source snapshot, 2026-10-23

This directory contains byte-preserved source/model snapshots bound to the six-method benchmark. Start with `../README_TH.md` and the result/evidence bundle linked from the repository handoff page.

## Map

- `source/native_build/wifi_threeway.cc` and its headers: the native AP–STA fixture used by the historical benchmark.
- `source/native_build/{receipt,dependencies,started}.json`: build identity, command and dependency hashes; historical paths are provenance, not portable installation paths.
- `source/library/`: recorded ns-3 library modifications and build records.
- `inference/<method>/policy/`: the method-specific capsule, config, model and exact source inventory. Keep each tree separate.
- `inference/<method>/sources/`: controller/supervisor snapshots. These can differ from the capsule's own source versions; do not flatten them.
- `inference/<method>/receipt.json`: model, source, environment and policy bindings.
- `training/selected_final_segment/`: selected resumed GAT training segment/config/checkpoint; **not a complete training campaign**.
- `SOURCE_BINDINGS.json`: SHA-256, original ZIP member and byte count for all 238 preserved files.

Method identities: `gat` = GAT+RAES raw; `actuation_hybrid` = GAT+RAES Hybrid; `actuation_qneutral` = Q-neutral ablation; `ganet` / `dgn` = project adaptations; `bfs_ca_inspired` = paper-inspired single-hop adaptation. The evaluated GAT/Hybrid/Q-neutral weight file is identical; their decision/actuation logic differs. This does not imply the learned GAT alone causes the full Hybrid improvement.

## Environment and reproduction boundary

The recorded GAT environment is Python 3.12.10, CPU, torch 2.12.0+cpu and NumPy 2.4.3. DGN has its own Python 3.7 x64 / TensorFlow 2.1.0 / Keras 2.3.1 environment recorded in its receipt. Use the respective receipts rather than combining them into one environment.

The native build depends on an external ns-3.47 tree/object build and additional recorded libraries. Path/hash guards reference the historical workspace. This export has **not** been verified as an independent clean native build. The selected training segment lacks earlier resume state/source versions; it must not be advertised as a complete from-scratch training recipe. Do not run a controller merely to inspect this snapshot.

Portable **evidence verification**, using Python standard library, is provided separately in `../evidence/verify_selection.py`. It recomputes recorded data, not a fresh simulator experiment. Two known seeds, selected conditions and post-result method selection do not establish statistical superiority or fresh held-out generalization.

No new training, simulator run, library rebuild or policy/metric change was performed for this handoff. There is no added license grant for third-party adapted sources; consult the original project/paper licenses before redistributing those components.
