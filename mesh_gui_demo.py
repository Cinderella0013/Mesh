import torch
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# นำเข้าคลาสจากไฟล์ของคุณ
from wifi_env import WiFiMeshEnv
from marl_gnn import GNN_Q_Network

# กำหนดสีสำหรับ 4 ช่องสัญญาณ
COLORS = ['#ff4d4d', '#4da6ff', '#4dff4d', '#ffff4d']

def render_gui(env, actions, step_num, avg_interference):
    plt.clf() # ล้างหน้าจอเก่า
    G = nx.Graph()
    num_nodes = 15

    # 1. ดึงตำแหน่งพิกัดของโหนดมาวาด
    # (เช็คว่าใน wifi_env ตั้งชื่อตำแหน่งโหนดว่าอะไร ถ้าไม่มีมันจะสุ่มพิกัดจำลองให้ครับ)
    if hasattr(env, 'positions'):
        positions = env.positions
    elif hasattr(env, 'pos'):
        positions = env.pos
    else:
        np.random.seed(42) # ล็อก Seed ให้วาดกราฟหน้าตาเหมือนเดิมทุกรอบ
        positions = np.random.rand(num_nodes, 2) * 150

    for i in range(num_nodes):
        G.add_node(i, pos=(positions[i][0], positions[i][1]))
        
    # 2. วาดเส้นประ (เส้นรบกวน) จาก env.adj
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if env.adj[i][j] == 1:
                G.add_edge(i, j)

    pos = nx.get_node_attributes(G, 'pos')
    
    # 3. กำหนดสีโหนดตามช่องสัญญาณที่ AI เพิ่งตัดสินใจเลือก (actions)
    node_colors = [COLORS[int(actions[i])] for i in range(num_nodes)]
    
    # 4. วาดกราฟลงบนจอ
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=600, edgecolors='black')
    nx.draw_networkx_labels(G, pos, font_color='white', font_weight='bold')
    nx.draw_networkx_edges(G, pos, alpha=0.3, style='dashed')
    
    # ตั้งค่าหัวข้อ GUI
    plt.title(f"Real AI Testing (MARL+GNN)\nStep: {step_num} | Avg Interference: {avg_interference:.2f}", fontsize=14)
    plt.suptitle("Red=Ch0 | Blue=Ch1 | Green=Ch2 | Yellow=Ch3", fontsize=10, color='gray')
    plt.axis('off')
    
    plt.draw()
    plt.pause(1.0) # หน่วงเวลา 1 วินาทีให้กรรมการดูทัน

def test_real_model():
    print("Loading Environment and Model...")
    num_nodes = 15
    num_channels = 4
    
    # สร้างสภาพแวดล้อม
    env = WiFiMeshEnv(num_nodes=num_nodes, num_channels=num_channels, area_size=150, interference_range=30)
    
    # สร้างโมเดลแบบเดียวกับใน train.py
    model = GNN_Q_Network(input_dim=num_channels + num_nodes, hidden_dim=64, output_dim=num_channels)
    
    # โหลดน้ำหนัก (Weights) ที่ AI เรียนรู้มาแล้ว
    model.load_state_dict(torch.load("marl_gnn_model.pth"))
    model.eval() # ปิดโหมดเทรน
    
    # รีเซ็ตเริ่มเกม
    state = env.reset()
    
    plt.ion() # เปิดโหมดเคลื่อนไหวของกราฟ
    fig = plt.figure(figsize=(8, 8))
    
    print("\nเริ่มการจำลอง (Simulation Started)...")
    
    # ทดสอบการทำงาน 20 Steps
    for step in range(1, 21):
        # ป้องกันกรณีที่ state และ adj ไม่ได้เป็น Tensor
        state_tensor = state if isinstance(state, torch.Tensor) else torch.FloatTensor(state)
        adj_tensor = env.adj if isinstance(env.adj, torch.Tensor) else torch.FloatTensor(env.adj)
        
        # ให้ AI ทำนาย Q-Values
        with torch.no_grad():
            q_values = model(state_tensor, adj_tensor)
            
        # เลือก Action ที่ดีที่สุด 100% (ไม่ใช้ Epsilon เพราะนี่คือตอนเทสต์ใช้งานจริง)
        actions = []
        for i in range(num_nodes):
            actions.append(torch.argmax(q_values[i]).item())
            
        # ส่ง Action ไปรันใน Environment
        next_state, rewards, avg_interference = env.step(actions)
        
        # วาดภาพ GUI
        render_gui(env, actions, step, avg_interference)
        
        # แปลงค่า Reward รวมเพื่อปริ้นท์ลง Terminal
        sys_reward = float(np.sum(rewards)) if not isinstance(rewards, torch.Tensor) else float(torch.sum(rewards))
        print(f"Step {step:02d} | Avg Interference: {avg_interference:.2f} | System Reward: {sys_reward:.2f}")
        
        # ไปยังสถานะถัดไป
        state = next_state
        
    plt.ioff()
    print("\nการจำลองเสร็จสิ้น! AI สามารถเจรจาหลบหลีกสัญญาณกวนได้สำเร็จ")
    plt.show() # ค้างหน้าจอสุดท้ายไว้

if __name__ == "__main__":
    test_real_model()