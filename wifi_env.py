import numpy as np
import torch

class WiFiMeshEnv:
    def __init__(self, num_nodes=10, num_channels=3, area_size=100, interference_range=40):
        self.num_nodes = num_nodes
        self.num_channels = num_channels
        
        # สุ่มตำแหน่งโหนด
        self.node_pos = np.random.rand(num_nodes, 2) * area_size
        self.channels = np.random.randint(0, num_channels, num_nodes)
        
        # สร้าง Adjacency Matrix
        self.adj = self._build_adj(interference_range)

    def _build_adj(self, threshold):
        adj = np.zeros((self.num_nodes, self.num_nodes))
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                dist = np.linalg.norm(self.node_pos[i] - self.node_pos[j])
                if dist < threshold and i != j:
                    adj[i, j] = 1
        return torch.FloatTensor(adj)

    def step(self, actions):
        rewards = np.zeros(self.num_nodes)
        interferences = np.zeros(self.num_nodes)
        
        for i in range(self.num_nodes):
            old_channel = self.channels[i]
            new_channel = actions[i]
            
            # คำนวณ Interference จากเพื่อนบ้านที่ใช้ช่องเดียวกัน
            interference_count = 0
            for j in range(self.num_nodes):
                if self.adj[i, j] == 1 and new_channel == actions[j]:
                    interference_count += 1
            
            # จำลอง Throughput แบบง่าย
            base_throughput = 10.0 
            throughput = max(0, base_throughput - (interference_count * 3.0))
            
            # ปิด Switch Cost เป็น 0 ไปเลย เพื่อให้ Agent กล้ากระโดดหนีเพื่อน
            switch_cost = 0  
            
            # เพิ่มบทลงโทษ 
            alpha = 1.0
            beta = 5.0  
            
            # สูตรใหม่: Reward = Throughput - (Beta * Interference)
            rewards[i] = (alpha * throughput) - (beta * interference_count) - switch_cost
            interferences[i] = interference_count
            
        # อัปเดตช่องสัญญาณใหม่
        self.channels = np.array(actions)
        
        # คืนค่า State ปัจจุบัน (One-hot channels), Rewards, และสถิติ
        next_state = self._get_state()
        return next_state, rewards, np.mean(interferences)

    def _get_state(self):
        # สร้าง State ที่มีขนาด = (จำนวนช่องสัญญาณ + จำนวนโหนด)
        state = torch.zeros((self.num_nodes, self.num_channels + self.num_nodes))
        for i in range(self.num_nodes):
            state[i, self.channels[i]] = 1           # 1. ข้อมูลช่องสัญญาณที่ใช้อยู่ (One-hot)
            state[i, self.num_channels + i] = 1      # 2. ข้อมูล Node ID (One-hot) เพื่อให้ AI รู้ว่าตัวเองคือใคร
        return state
        
    def reset(self):
        self.channels = np.random.randint(0, self.num_channels, self.num_nodes)
        return self._get_state()