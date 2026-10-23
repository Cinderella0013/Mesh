import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class GNN_Q_Network(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNN_Q_Network, self).__init__()

        self.fc_self = nn.Linear(input_dim, hidden_dim)
        self.fc_neighbor = nn.Linear(input_dim, hidden_dim)

        self.fc_policy = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, state, adj):

        h_self = F.relu(self.fc_self(state))

        neighbor_state = torch.matmul(adj, state) 
        h_neighbor = F.relu(self.fc_neighbor(neighbor_state))
        
        h_combined = torch.cat([h_self, h_neighbor], dim=1)
        

        q_values = self.fc_policy(h_combined)
        return q_values


# =============================================================================
# GAT upgrade: แทนที่ fc_self/fc_neighbor (average เพื่อนบ้านแบบเท่ากันหมด)
# ด้วย Graph Attention ที่เรียนรู้น้ำหนัก alpha ต่อคู่โหนด + กลไกลดความผันผวน
# ของ attention ในช่วง early training (ดู gat_stability_utils.py คู่กัน)
# =============================================================================
class GATLayer(nn.Module):
    """
    Single-hop GAT layer (dense adjacency, self-loop เติมอัตโนมัติ)
    แก้ over-smoothing / action-differentiation collapse:
      1. multi-head CONCAT (ไม่ใช่ average) รักษา representational capacity
      2. strong residual: h_out = ELU(concat_heads + W_res @ state)
         คงเอกลักษณ์ของโหนดไว้ ไม่ให้ attention เฉลี่ยจนภาพเบลอ
      3. ชั้นเดียว (shallow) ลด over-smoothing
    out_dim ที่รับเข้ามา = มิติผลลัพธ์รวมหลัง concat (per-head = out_dim // num_heads)
    """

    def __init__(self, in_dim, out_dim, num_heads=4, dropout=0.2,
                 leaky_slope=0.2, temperature=1.0, edge_dim=0):
        super().__init__()
        assert out_dim % num_heads == 0, "out_dim ต้องหารด้วย num_heads ลงตัว (concat)"
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        self.out_dim = out_dim
        self.edge_dim = edge_dim
        # register_buffer แทน plain attribute เพื่อให้ค่า temperature ที่ anneal
        # ระหว่างเทรน (ผ่าน TemperatureScheduler) ถูกบันทึก/โหลดคู่กับ state_dict
        # อัตโนมัติ — ถ้าเป็น plain float ค่านี้จะหายไปตอน torch.save/load
        self.register_buffer('temperature', torch.tensor(float(temperature)))

        self.W = nn.Parameter(torch.empty(num_heads, in_dim, self.head_dim))
        self.a_src = nn.Parameter(torch.empty(num_heads, self.head_dim))
        self.a_dst = nn.Parameter(torch.empty(num_heads, self.head_dim))
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.a_src.unsqueeze(0))
        nn.init.xavier_uniform_(self.a_dst.unsqueeze(0))

        if edge_dim > 0:
            # edge-conditioned attention: score เพิ่มพจน์ a_e^T (W_e e_ij) ต่อ head
            # ทำให้ attention เห็น R_ij / load / same-channel รายคู่โดยตรง
            # (ข้อมูลที่ node feature ไม่มีทางมี เพราะเป็นคุณสมบัติของ "คู่" โหนด)
            self.a_edge = nn.Parameter(torch.empty(num_heads, edge_dim))
            nn.init.xavier_uniform_(self.a_edge)

        # residual projection: state (in_dim) -> out_dim ให้บวกกับ concat ได้
        self.W_res = nn.Linear(in_dim, out_dim, bias=False)

        self.leaky_relu = nn.LeakyReLU(leaky_slope)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, state, adj, edge_feat=None):
        N = state.size(0)
        eye = torch.eye(N, device=adj.device, dtype=adj.dtype)
        adj_self = torch.clamp(adj + eye, max=1.0)  # เติม self-loop

        Wh = torch.einsum('ni,hio->hno', state, self.W)          # [heads,N,head_dim]
        src_score = torch.einsum('hno,ho->hn', Wh, self.a_src).unsqueeze(2)
        dst_score = torch.einsum('hno,ho->hn', Wh, self.a_dst).unsqueeze(1)
        score = src_score + dst_score
        if self.edge_dim > 0 and edge_feat is not None:
            score = score + torch.einsum('nme,he->hnm', edge_feat, self.a_edge)
        # หารด้วยค่า scalar (ไม่ใช่ tensor ใน graph) — temperature เป็น buffer ที่
        # scheduler แก้ inplace (fill_) ถ้า div ด้วย tensor ตรง ๆ backward จะพังเมื่อ
        # fill_ เกิดระหว่าง forward กับ backward
        e = self.leaky_relu(score) / float(self.temperature)

        mask = (adj_self == 0).unsqueeze(0)
        e = e.masked_fill(mask, float('-inf'))
        alpha = F.softmax(e, dim=-1)                              # [heads,N,N]
        alpha = torch.nan_to_num(alpha, nan=0.0)
        alpha = self.attn_dropout(alpha)

        h_prime = torch.einsum('hnm,hmo->hno', alpha, Wh)         # [heads,N,head_dim]
        h_cat = h_prime.permute(1, 0, 2).reshape(N, self.out_dim)  # multi-head CONCAT
        h_out = F.elu(h_cat + self.W_res(state))                   # residual + ELU
        return h_out, alpha


class GAT_Q_Network(nn.Module):
    """
    Drop-in ทดแทน GNN_Q_Network เดิม: constructor และ forward(state, adj)
    ยังใช้แบบเดียวกันได้ (return_attention=False คืนค่า q_values เฉย ๆ
    เหมือนเดิม ใช้กับ app.py/test_*.py ได้โดยไม่ต้องแก้ไข)
    ตั้ง return_attention=True เมื่อต้องใช้ attention weights ไป regularize
    ตอนเทรน (ดู train_gat.py)
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_heads=4,
                 temperature=1.0, dropout=0.2, edge_dim=0):
        super().__init__()
        self.edge_dim = edge_dim
        self.gat = GATLayer(input_dim, hidden_dim, num_heads=num_heads,
                             temperature=temperature, dropout=dropout, edge_dim=edge_dim)
        self.fc_policy = nn.Linear(hidden_dim, output_dim)

    def forward(self, state, adj, edge_feat=None, return_attention=False):
        h, alpha = self.gat(state, adj, edge_feat=edge_feat)
        q_values = self.fc_policy(h)
        if return_attention:
            return q_values, alpha
        return q_values

    def attention_params(self):
        """param group ของ attention (W, a_src, a_dst) สำหรับ two-timescale LR/clipping"""
        return [p for n, p in self.named_parameters()
                if n.startswith('gat.') and ('a_src' in n or 'a_dst' in n or n.endswith('.W'))]

    def other_params(self):
        attn_ids = {id(p) for p in self.attention_params()}
        return [p for p in self.parameters() if id(p) not in attn_ids]

