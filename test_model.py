import torch
import numpy as np
from wifi_env import WiFiMeshEnv      # ไฟล์ Environment ของเรา
from marl_gnn import GNN_Q_Network    # ไฟล์ Model ของเรา

# 1. ตั้งค่าให้ตรงกับตอนที่ Train โมเดลมา
num_nodes = 15
num_channels = 4
area_size = 150
interference_range = 30

# 2. สร้าง Environment และ สร้างโครงสร้าง Model เปล่าๆ
env = WiFiMeshEnv(num_nodes=num_nodes, num_channels=num_channels, area_size=area_size, interference_range=interference_range)
model = GNN_Q_Network(input_dim=num_channels, hidden_dim=32, output_dim=num_channels)

# 3. โหลดสมอง (Weights) ที่เซฟไว้มาใส่ใน Model
model.load_state_dict(torch.load("marl_gnn_model.pth"))
model.eval() # สั่งให้โมเดลเข้าสู่โหมด "ทดสอบ" (ปิดการทำงานบางอย่างที่ใช้ตอนเทรนเท่านั้น)
print("โหลดโมเดล 'marl_gnn_model.pth' สำเร็จ!")

# --- 4. เริ่มการทดสอบ (Inference) ---
state = env.reset()
print(f"ช่องสัญญาณเริ่มต้น (สุ่ม): {env.channels}")

with torch.no_grad():
    # ให้ GNN+MARL มองกราฟและคิดหา Q-values
    q_values = model(state, env.adj)
    
    # เลือก Action ที่ดีที่สุด 100% (ไม่มีการสุ่ม Epsilon แล้ว)
    actions = [torch.argmax(q_values[i]).item() for i in range(num_nodes)]

# นำ Action ไปใช้จริงใน Environment
next_state, rewards, avg_interference = env.step(actions)

print(f"ช่องสัญญาณที่ Agent เลือกให้: {env.channels}")
print(f"ผลลัพธ์การจัดสรร -> Average Reward: {np.mean(rewards):.2f} | Average Interference: {avg_interference:.2f}")