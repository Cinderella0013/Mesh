import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class GNN_Q_Network(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNN_Q_Network, self).__init__()
        # แยกสมองเป็น 2 เส้นทาง: เส้นทางของ "ตัวเอง" และ เส้นทางของ "เพื่อนบ้าน"
        self.fc_self = nn.Linear(input_dim, hidden_dim)
        self.fc_neighbor = nn.Linear(input_dim, hidden_dim)
        
        # เลเยอร์ตัดสินใจ (รับข้อมูลตัวเอง + ข้อมูลเพื่อนบ้านมารวมกัน = hidden_dim * 2)
        self.fc_policy = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, state, adj):
        # 1. ประมวลผลข้อมูลตัวเอง (Identity & Current Channel)
        h_self = F.relu(self.fc_self(state))
        
        # 2. ประมวลผลข้อมูลเพื่อนบ้าน (Message Passing)
        # เอา Adjacency Matrix คูณ State ของคนอื่นมารวมกัน
        neighbor_state = torch.matmul(adj, state) 
        h_neighbor = F.relu(self.fc_neighbor(neighbor_state))
        
        # 3. รวมข้อมูลแบบต่อติดกัน (Concatenation) 
        # ทำให้ AI รู้ชัดเจนว่า "นี่คือตัวเอง" และ "นั่นคือเพื่อนบ้าน" ไม่กลืนเป็นเนื้อเดียวกัน
        h_combined = torch.cat([h_self, h_neighbor], dim=1)
        
        # 4. ส่งออกเป็นคะแนนความน่าจะเป็นของแต่ละช่องสัญญาณ (Q-values)
        q_values = self.fc_policy(h_combined)
        return q_values