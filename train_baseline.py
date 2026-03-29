import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# นำเข้า Env ของเรา และ สมองแบบ Baseline
from wifi_env import WiFiMeshEnv
from marl_baseline import Baseline_Q_Network


num_nodes = 15
num_channels = 4
episodes = 10000          
steps_per_episode = 20    
epsilon = 1.0
epsilon_decay = 0.995     
epsilon_min = 0.05

# สร้าง Env และ Model แบบ Baseline
env = WiFiMeshEnv(num_nodes=num_nodes, num_channels=num_channels, area_size=150, interference_range=30)

model = Baseline_Q_Network(input_dim=num_channels + num_nodes, hidden_dim=64, output_dim=num_channels)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

history_reward = []
history_interference = []

print("🚀 Starting Baseline MARL Training (No GNN)...")

for ep in range(episodes):
    state = env.reset() 
    ep_reward = 0
    ep_interference = 0
    
    for step in range(steps_per_episode):
        # ส่ง state ไปให้โมเดลประมวลผล เพื่อหา Q-values
        q_values = model(state, env.adj)
        
        actions = []
        for i in range(num_nodes):
            if np.random.rand() < epsilon:
                actions.append(np.random.randint(num_channels))
            else:
                actions.append(torch.argmax(q_values[i]).item())
                
        next_state, rewards, avg_interference = env.step(actions)
        
        with torch.no_grad():
            next_q_values = model(next_state, env.adj)
            max_next_q = torch.max(next_q_values, dim=1)[0]
        
        targets = q_values.clone().detach()
        for i in range(num_nodes):
            targets[i, actions[i]] = rewards[i] + 0.9 * max_next_q[i]

        loss = criterion(q_values, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        state = next_state
        ep_reward += np.mean(rewards)
        ep_interference += avg_interference
        
    history_reward.append(ep_reward / steps_per_episode)
    history_interference.append(ep_interference / steps_per_episode)
    
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    
    if (ep+1) % 500 == 0:
        print(f"Episode {ep+1}/{episodes} | Avg Reward: {history_reward[-1]:.2f} | Avg Interference: {history_interference[-1]:.2f}")

print("✅ Baseline Training Complete!")

# บันทึกโมเดลคู่แข่ง
torch.save(model.state_dict(), "marl_baseline_model.pth")
print("บันทึกโมเดลคู่แข่งสำเร็จเป็นไฟล์ 'marl_baseline_model.pth'")

# พล็อตกราฟผลการเทรนของ Baseline
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(history_reward, color='orange') 
ax1.set_title("Baseline MARL: Average Reward")
ax1.set_xlabel("Episode")
ax1.set_ylabel("Reward")
ax1.grid(True)

ax2.plot(history_interference, color='purple')
ax2.set_title("Baseline MARL: Average Interference")
ax2.set_xlabel("Episode")
ax2.set_ylabel("Interference (Count)")
ax2.grid(True)

plt.tight_layout()
plt.show()