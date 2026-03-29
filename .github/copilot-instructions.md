# Copilot instructions — Mesh WiFi GNN (small RL/GNN project)

This file gives focused, actionable guidance for AI coding agents working on this repository so you can be productive immediately.

Overview
- Purpose: decentralized channel-assignment RL prototype using a small GNN that predicts per-node Q-values. The goal is to minimize interference while keeping throughput high.
- Major components:
  - `train.py` — training loop / entrypoint. Contains hyperparameters (num_nodes, num_channels, episodes, epsilon) and the epsilon-greedy Q-learning update.
  - `marl_gnn.py` — model definition (`GNN_Q_Network`). Implements a simple message-passing step: x_gnn = adj @ x + x and outputs per-node Q-values.
  - `wifi_env.py` — environment (`WiFiMeshEnv`). Manages node positions, adjacency (distance threshold), one-hot state encoding, `step(actions)` and `reset()`.

Data shapes & flow (explicit)
- State: tensor of shape [num_nodes, num_channels] (one-hot per node) — returned by `WiFiMeshEnv._get_state()` and passed to `model(state, adj)`.
- Adjacency: `adj` is a square matrix [num_nodes, num_nodes] (torch.FloatTensor) built from distances; 1 = interference edge.
- Model I/O: `GNN_Q_Network.forward(state, adj)` -> `q_values` shaped [num_nodes, num_channels].
- Actions: Python list/array length `num_nodes` of ints (channel indices). `env.step(actions)` -> `(next_state, rewards (np.array), avg_interference)`.

Project-specific conventions and patterns
- Mixing NumPy and PyTorch: code uses NumPy for some RNG and arrays (channels, rewards) and PyTorch for tensors and model ops. Always convert explicitly (e.g., `torch.from_numpy()` / `.numpy()` as needed).
- One-hot state encoding is canonical — transforms channel indices to a [num_nodes, num_channels] tensor.
- Simple GNN: message passing is implemented by dense matrix multiply `adj @ features` plus a residual `+ x`. No graph library is used; keep edits small and local unless you add dependency.
- Bellman update is performed per-node in `train.py` by copying `q_values` and overwriting the taken-action index. Vectorize with care — indexing is per-node, per-action.

How to run (developer workflow)
- Install minimal deps (Windows PowerShell):
```powershell
python -m pip install torch numpy matplotlib
```
- Run training (repo root):
```powershell
python train.py
```
- Notes:
  - `train.py` currently performs CPU-only operations and uses blocking `plt.show()` to display plots. If running headless or in CI, replace with `plt.savefig("out.png")`.
  - GPU: code does not set device or move tensors to GPU. If adding GPU support, ensure `state`, `adj`, model, and any created tensors are moved consistently.

Common pitfalls and quick fixes
- Type mismatches: `adj` is a `torch.FloatTensor` but `channels` and `rewards` are NumPy arrays. Convert explicitly to avoid silent bugs.
- In-place target updates: `targets = q_values.clone()` is used; keep this pattern if you maintain PyTorch autograd correctness.
- Plotting blocks execution: replace `plt.show()` with `plt.savefig()` in automated runs.

Where to look for changes
- Hyperparameters and experiment changes: edit `train.py`.
- Model changes: edit `marl_gnn.py` (`GNN_Q_Network` constructor/forward).
- Environment logic and adjacency calculations: edit `wifi_env.py` (`_build_adj`, `step`, `_get_state`).

Code-search shortcuts for an agent
- Key symbols: `WiFiMeshEnv`, `GNN_Q_Network`, `train.py` loop, `adj`, `state`, `step(`, `reset()`.
- To find where shapes are produced/consumed, search for `_get_state`, `forward(self, state, adj)`, and `env.step(`.

Recommended low-risk improvements an AI agent can propose and implement
- Add device handling (CPU/GPU) and a small helper `to_device(obj, device)` converting nested structures.
- Vectorize Bellman update where possible for clarity and speed, but keep tests/prints to verify correctness.
- Add a minimal `requirements.txt` listing `torch`, `numpy`, `matplotlib` and a short `README.md` with quick run steps.

If anything here is unclear or you'd like additional examples (e.g., exact lines to change for GPU support or a sample vectorized Bellman update), tell me which part to expand and I'll update this file.
