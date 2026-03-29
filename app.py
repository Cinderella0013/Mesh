from flask import Flask, jsonify, render_template
import torch
import numpy as np

# นำเข้าคลาสของคุณ
from wifi_env import WiFiMeshEnv
from marl_gnn import GNN_Q_Network

app = Flask(__name__, template_folder='.') # ตั้งค่าให้โหลด HTML จากโฟลเดอร์ปัจจุบัน

# 1. โหลด Environment และ Model ของจริง
num_nodes = 15
num_channels = 4
env = WiFiMeshEnv(num_nodes=num_nodes, num_channels=num_channels, area_size=150, interference_range=30)
model = GNN_Q_Network(input_dim=num_channels + num_nodes, hidden_dim=64, output_dim=num_channels)

# โหลดน้ำหนักสมองกลที่เทรนเสร็จแล้ว
model.load_state_dict(torch.load("marl_gnn_model.pth"))
model.eval()

# สร้างตัวแปรเก็บ State ล่าสุด
current_state = None

@app.route('/')
def index():
    # เปิดหน้าเว็บ Dashboard ของคุณ
    return render_template('marl_gnn_simulator.html')

@app.route('/api/reset')
def reset():
    global current_state
    current_state = env.reset()
    
    # แก้ไข: ใช้ env.node_pos ให้ถูกต้อง
    actual_pos = env.node_pos
    
    # แก้ไข: แปลง Tensor → numpy ก่อน argmax
    state_np = current_state.numpy() if isinstance(current_state, torch.Tensor) else np.array(current_state)
    
    return jsonify({
        "positions": actual_pos.tolist(),
        "adj": env.adj.tolist(),
        "channels": np.argmax(state_np[:, :num_channels], axis=1).tolist()
    })
@app.route('/api/randomize')  # ✅ เพิ่มใหม่: สร้าง env ใหม่ตำแหน่งใหม่หมดเลย
def randomize():
    global current_state, env
    env = WiFiMeshEnv(num_nodes=num_nodes, num_channels=num_channels, area_size=150, interference_range=30)
    current_state = env.reset()
    state_np = current_state.numpy() if isinstance(current_state, torch.Tensor) else np.array(current_state)
    return jsonify({
        "positions": env.node_pos.tolist(),
        "adj": env.adj.tolist(),
        "channels": np.argmax(state_np[:, :num_channels], axis=1).tolist()
    })
    
@app.route('/api/shift_positions')
def shift_positions():
    global current_state
    # ขยับพิกัดแบบ random walk เล็กน้อย (จำลอง dynamic environment)
    noise = np.random.uniform(-10, 10, env.node_pos.shape)
    env.node_pos = np.clip(env.node_pos + noise, 0, 150)
    
    # สร้าง adjacency ใหม่จากพิกัดที่ขยับแล้ว
    env.adj = env._build_adj(30)
    
    # คำนวณ state ใหม่จาก channel ปัจจุบัน (ไม่ reset channel!)
    current_state = env._get_state()
    state_np = current_state.numpy()
    
    return jsonify({
        "positions": env.node_pos.tolist(),
        "adj": env.adj.tolist(),
        "channels": np.argmax(state_np[:, :num_channels], axis=1).tolist()
    })

@app.route('/api/step')
def step():
    global current_state
    if current_state is None:
        current_state = env.reset()

    state_tensor = current_state if isinstance(current_state, torch.Tensor) else torch.FloatTensor(current_state)
    adj_tensor = env.adj if isinstance(env.adj, torch.Tensor) else torch.FloatTensor(env.adj)

    with torch.no_grad():
        q_values = model(state_tensor, adj_tensor)

    # ✅ แก้: ถ้า Q-values เท่ากันหมด ให้สุ่มช่องที่ไม่ชนแทน
    actions = []
    for i in range(num_nodes):
        q = q_values[i]
        # หาช่องที่ดีที่สุดจาก Q-values
        best_action = torch.argmax(q).item()
        actions.append(best_action)

    next_state, rewards, avg_interference = env.step(actions)
    current_state = next_state

    ci_list = []
    for i in range(num_nodes):
        ci = sum(1 for j in range(num_nodes) if env.adj[i][j] == 1 and actions[i] == actions[j])
        ci_list.append(ci)

    for i in range(num_nodes):
        if ci_list[i] > 0:
            neighbor_channels = set()
            for j in range(num_nodes):
                if env.adj[i][j] == 1:
                    neighbor_channels.add(actions[j])
            free_channels = [c for c in range(num_channels) if c not in neighbor_channels]
            if free_channels:
                # เลือกช่องว่างที่ Q-value สูงสุด
                q = q_values[i]
                best_free = max(free_channels, key=lambda c: q[c].item())
                actions[i] = best_free
                ci_list[i] = 0

    # คำนวณ CI ใหม่หลังแก้ไข
    ci_list = []
    for i in range(num_nodes):
        ci = sum(1 for j in range(num_nodes) if env.adj[i][j] == 1 and actions[i] == actions[j])
        ci_list.append(ci)

    return jsonify({
        "actions": actions,
        "q_values": q_values.numpy().tolist(),
        "rewards": rewards.tolist() if isinstance(rewards, np.ndarray) else rewards,
        "ci_list": ci_list,
        "avg_ci": float(avg_interference)
    })

if __name__ == '__main__':
    print("🔥 เปิดเซิร์ฟเวอร์ AI แล้ว! เข้าไปที่ http://127.0.0.1:5000")
    app.run(debug=True, port=5000)