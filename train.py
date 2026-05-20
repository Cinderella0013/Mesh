import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# project modules
from wifi_env import WiFiMeshEnv
from marl_gnn import GNN_Q_Network

# ตั้งค่าเริ่มต้น
num_nodes = 15
num_channels = 4
episodes = 10000           
steps_per_episode = 20   
epsilon = 1.0
epsilon_decay = 0.995     
epsilon_min = 0.05

# สร้าง Env และ Model
env = WiFiMeshEnv(num_nodes=num_nodes, num_channels=num_channels, area_size=150, interference_range=30)
model = GNN_Q_Network(input_dim=num_channels + num_nodes, hidden_dim=64, output_dim=num_channels)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# เก็บค่าสถิติเพื่อพล็อตกราฟ
history_reward = []
history_interference = []

print("Starting Training...")
state = env.reset()

for ep in range(episodes):
    state = env.reset() 
    ep_reward = 0
    ep_interference = 0
    
    for step in range(steps_per_episode):
        # Forward Pass
        q_values = model(state, env.adj)
        
        # เลือก Action
        actions = []
        for i in range(num_nodes):
            if np.random.rand() < epsilon:
                actions.append(np.random.randint(num_channels))
            else:
                actions.append(torch.argmax(q_values[i]).item())
                
        # นำ Action ไปรันใน Environment
        next_state, rewards, avg_interference = env.step(actions)
        
        # 4. คำนวณ Target Q-values
        with torch.no_grad():
            next_q_values = model(next_state, env.adj)
            max_next_q = torch.max(next_q_values, dim=1)[0]
        
  
        targets = q_values.clone().detach()
        for i in range(num_nodes):
            # Q_target = Reward + (Discount * Max_Next_Q)
            targets[i, actions[i]] = rewards[i] + 0.9 * max_next_q[i]

        # 5. Backpropagation อัปเดตโมเดล
        loss = criterion(q_values, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        state = next_state
        ep_reward += np.mean(rewards)
        ep_interference += avg_interference
        
    # เก็บค่าเฉลี่ย
    history_reward.append(ep_reward / steps_per_episode)
    history_interference.append(ep_interference / steps_per_episode)
    
    # ลดค่าการสุ่มเมื่อจบ
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    
    if (ep+1) % 50 == 0:
        print(f"Episode {ep+1}/{episodes} | Avg Reward: {history_reward[-1]:.2f} | Avg Interference: {history_interference[-1]:.2f}")

print("Training Complete!")

# บันทึกเฉพาะ State Dictionary (ค่าน้ำหนักที่เรียนรู้แล้ว)
torch.save(model.state_dict(), "marl_gnn_model.pth")
print("บันทึกโมเดลสำเร็จ 'marl_gnn_model.pth'")


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(history_reward, color='blue')
ax1.set_title("Average Reward per Episode")
ax1.set_xlabel("Episode")
ax1.set_ylabel("Reward")
ax1.grid(True)

ax2.plot(history_interference, color='red')
ax2.set_title("Average Interference per Node")
ax2.set_xlabel("Episode")
ax2.set_ylabel("Interference (Count)")
ax2.grid(True)

plt.tight_layout()
plt.show()