import torch
import numpy as np
import matplotlib.pyplot as plt
from wifi_env import WiFiMeshEnv
from marl_gnn import GNN_Q_Network
from marl_baseline import Baseline_Q_Network


num_nodes = 15
num_channels = 4
area_size = 150
interference_range = 30
episodes_test = 100      
steps_per_episode = 20    

env = WiFiMeshEnv(num_nodes=num_nodes, num_channels=num_channels, area_size=area_size, interference_range=interference_range)


model_gnn = GNN_Q_Network(input_dim=num_channels + num_nodes, hidden_dim=64, output_dim=num_channels)
model_gnn.load_state_dict(torch.load("marl_gnn_model.pth"))
model_gnn.eval()


model_baseline = Baseline_Q_Network(input_dim=num_channels + num_nodes, hidden_dim=64, output_dim=num_channels)
model_baseline.load_state_dict(torch.load("marl_baseline_model.pth"))
model_baseline.eval()


def evaluate_policy(policy_type):
    total_interferences = []
    total_rewards = []
    
    for _ in range(episodes_test):
        state = env.reset()
        
        
        for step in range(steps_per_episode):
            with torch.no_grad():
                if policy_type == "Baseline_MARL":
                    q_values = model_baseline(state)
                    actions = [torch.argmax(q_values[i]).item() for i in range(num_nodes)]
                    
                elif policy_type == "Proposed_GNN":
                    q_values = model_gnn(state, env.adj)
                    actions = [torch.argmax(q_values[i]).item() for i in range(num_nodes)]
                    
            next_state, rewards, avg_interference = env.step(actions)
            state = next_state 
            
        
        total_rewards.append(np.mean(rewards))
        total_interferences.append(avg_interference)
        
    return np.mean(total_rewards), np.mean(total_interferences)


print("กำลังรันการทดสอบเปรียบเทียบ ")

reward_base, intf_base = evaluate_policy("Baseline_MARL")
reward_gnn, intf_gnn = evaluate_policy("Proposed_GNN")

print("\nผลการทดสอบเฉลี่ย ")

print(f"2. Baseline MARL (Paper) -> Reward: {reward_base:.2f} | Interference: {intf_base:.2f}")
print(f"3. Proposed MARL+GNN (NW12) -> Reward: {reward_gnn:.2f} | Interference: {intf_gnn:.2f}")


labels = [ 'Baseline MARL', 'Proposed MARL+GNN']
rewards = [ reward_base, reward_gnn]
interferences = [ intf_base, intf_gnn]

x = np.arange(len(labels))
width = 0.35

fig, ax1 = plt.subplots(figsize=(10, 6))

rects1 = ax1.bar(x - width/2, rewards, width, label='System Reward (Higher is Better)', color='royalblue')
rects2 = ax1.bar(x + width/2, interferences, width, label='Interference Level (Lower is Better)', color='crimson')

ax1.set_ylabel('Scores')
ax1.set_title('Final Performance Comparison (After 20 steps of negotiation)')
ax1.set_xticks(x)
ax1.set_xticklabels(labels)
ax1.legend()

ax1.bar_label(rects1, padding=3, fmt='%.2f')
ax1.bar_label(rects2, padding=3, fmt='%.2f')

fig.tight_layout()
plt.show()