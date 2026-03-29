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
    # ดึงพิกัดที่ env ใช้สร้าง Adjacency Matrix จริงๆ
    actual_pos = env.positions if hasattr(env, 'positions') else env.pos
    return jsonify({
        "positions": actual_pos.tolist(),
        "adj": env.adj.tolist(), # เส้นประจะตรงกับ actual_pos เสมอ
        "channels": np.argmax(current_state[:, :4], axis=1).tolist()
    })

@app.route('/api/step')
def step():
    global current_state
    
    if current_state is None:
        current_state = env.reset()
    
    # 1. ให้ AI ของจริงทำนาย
    state_tensor = current_state if isinstance(current_state, torch.Tensor) else torch.FloatTensor(current_state)
    adj_tensor = env.adj if isinstance(env.adj, torch.Tensor) else torch.FloatTensor(env.adj)
    
    with torch.no_grad():
        q_values = model(state_tensor, adj_tensor)
        
    # เลือก Action ที่ดีที่สุด
    actions = [torch.argmax(q_values[i]).item() for i in range(num_nodes)]
    
    # 2. ส่ง Action ไปใน Environment จริง
    next_state, rewards, avg_interference = env.step(actions)
    current_state = next_state
    
    # คำนวณ CI แต่ละโหนด (จำลองเพื่อส่งให้ UI)
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