import torch
import torch.nn as nn
import torch.nn.functional as F

class Baseline_Q_Network(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Baseline_Q_Network, self).__init__()
        # สมองแบบดั้งเดิม (MLP) ไม่มีการแยกสายข้อมูลเพื่อนบ้าน
        self.fc1 = nn.Linear(input_dim, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc_policy = nn.Linear(hidden_dim, output_dim)

    def forward(self, state, adj=None):
        # จุดสำคัญ: รับ adj (Adjacency Matrix) เข้ามา แต่ "ไม่ใช้งานเลย"
        # ทำให้โหนดตัดสินใจจากข้อมูลของตัวเองล้วนๆ เหมือนคนตาบอดคลำช้าง
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        q_values = self.fc_policy(x)
        return q_values