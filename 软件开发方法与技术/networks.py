import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict

class ActorNetwork(nn.Module):
    """Actor网络基类"""
    def __init__(self, state_dim: int, hidden_dim: int = 512):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = F.relu(self.fc4(x))
        return x

class DiscreteActorNetwork(ActorNetwork):
    """离散动作策略网络 πθ(ad | s)"""
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super(DiscreteActorNetwork, self).__init__(state_dim, hidden_dim)
        self.action_head = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, state):
        features = super().forward(state)
        action_logits = self.action_head(features)
        # 添加梯度裁剪防止梯度爆炸
        action_logits = torch.clamp(action_logits, -10, 10)
        return action_logits
    
    def get_action(self, state, deterministic=False):
        """获取离散动作"""
        logits = self.forward(state)
        
        # 检查数值稳定性
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print(f"Warning: logits contain nan/inf: {logits}")
            # 使用均匀分布作为fallback
            logits = torch.zeros_like(logits)
        
        if deterministic:
            action = torch.argmax(logits, dim=-1)
        else:
            # 使用更稳定的softmax计算
            logits = logits - torch.max(logits, dim=-1, keepdim=True)[0]  # 数值稳定性
            probs = F.softmax(logits, dim=-1)
            
            # 检查概率的有效性
            if torch.isnan(probs).any() or torch.isinf(probs).any() or (probs < 0).any():
                print(f"Warning: invalid probabilities: {probs}")
                # 使用均匀分布作为fallback
                probs = torch.ones_like(probs) / probs.shape[-1]
            
            # 确保概率和为1
            probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-8)
            
            action = torch.multinomial(probs, 1).squeeze(-1)
        return action
    
    def get_log_prob(self, state, action):
        """获取动作的对数概率"""
        logits = self.forward(state)
        
        # 检查数值稳定性
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print(f"Warning: logits contain nan/inf in get_log_prob: {logits}")
            logits = torch.zeros_like(logits)
        
        # 使用更稳定的log_softmax
        logits = logits - torch.max(logits, dim=-1, keepdim=True)[0]  # 数值稳定性
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs.gather(-1, action.unsqueeze(-1)).squeeze(-1)

class ContinuousActorNetwork(ActorNetwork):
    """连续动作策略网络 πχ(ac | s) 和 πξ(p | s)"""
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128, 
                 min_action: float = 0.0, max_action: float = 1.0):
        super(ContinuousActorNetwork, self).__init__(state_dim, hidden_dim)
        self.action_dim = action_dim
        self.min_action = min_action
        self.max_action = max_action
        
        # 均值和标准差头
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)
        
        # 初始化标准差
        self.log_std_head.weight.data.fill_(-0.5)
        self.log_std_head.bias.data.fill_(-0.5)
        
    def forward(self, state):
        features = super().forward(state)
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        
        # 添加梯度裁剪和数值稳定性
        mean = torch.clamp(mean, -10, 10)
        log_std = torch.clamp(log_std, -20, 2)  # 限制标准差范围
        
        # 检查数值稳定性
        if torch.isnan(mean).any() or torch.isinf(mean).any():
            print(f"Warning: mean contains nan/inf: {mean}")
            mean = torch.zeros_like(mean)
        
        if torch.isnan(log_std).any() or torch.isinf(log_std).any():
            print(f"Warning: log_std contains nan/inf: {log_std}")
            log_std = torch.ones_like(log_std) * -0.5
        
        return mean, log_std
    
    def get_action(self, state, deterministic=False):
        """获取连续动作"""
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        
        if deterministic:
            action = mean
        else:
            noise = torch.randn_like(mean)
            action = mean + std * noise
        
        # 应用动作范围约束
        action = torch.clamp(action, self.min_action, self.max_action)
        return action
    
    def get_log_prob(self, state, action):
        """获取动作的对数概率"""
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        
        # 检查数值稳定性
        if torch.isnan(mean).any() or torch.isinf(mean).any() or torch.isnan(std).any() or torch.isinf(std).any():
            print(f"Warning: mean or std contains nan/inf in get_log_prob")
            return torch.zeros(action.shape[0], device=action.device)
        
        # 计算正态分布的对数概率
        log_prob = -0.5 * ((action - mean) / std) ** 2 - log_std - 0.5 * np.log(2 * np.pi)
        
        # 检查结果的有效性
        if torch.isnan(log_prob).any() or torch.isinf(log_prob).any():
            print(f"Warning: log_prob contains nan/inf: {log_prob}")
            log_prob = torch.zeros_like(log_prob)
        
        return log_prob.sum(dim=-1)

class TaskAllocationNetwork(ContinuousActorNetwork):
    """任务分配网络 πχ(ac | s)"""
    def __init__(self, state_dim: int, hidden_dim: int = 128):
        super(TaskAllocationNetwork, self).__init__(
            state_dim, action_dim=3, hidden_dim=hidden_dim, 
            min_action=0.0, max_action=1.0
        )
    
    def get_action(self, state, deterministic=False):
        """获取任务分配动作，确保和为1"""
        action = super().get_action(state, deterministic)
        
        # 使用softmax确保分配比例和为1
        if not deterministic:
            action = F.softmax(action, dim=-1)
        else:
            # 对于确定性动作，直接归一化
            action = action / (action.sum(dim=-1, keepdim=True) + 1e-8)
        
        return action

class PowerAllocationNetwork(ContinuousActorNetwork):
    """功率分配网络 πξ(p | s)"""
    def __init__(self, state_dim: int, max_power_u: float = 18.0, 
                 max_power_n: float = 18.0, max_power_b: float = 70.0,
                 hidden_dim: int = 128):
        super(PowerAllocationNetwork, self).__init__(
            state_dim, action_dim=3, hidden_dim=hidden_dim,
            min_action=0.0, max_action=1.0
        )
        self.max_powers = torch.tensor([max_power_u, max_power_n, max_power_b])
    
    def get_action(self, state, deterministic=False):
        """获取功率分配动作"""
        # 获取0-1范围的功率比例
        power_ratio = super().get_action(state, deterministic)
        
        # 缩放到实际功率范围
        action = power_ratio * self.max_powers.to(state.device)
        return action

class CriticNetwork(nn.Module):
    """价值网络 Vω(s)"""
    def __init__(self, state_dim: int, hidden_dim: int = 512):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.value_head = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = F.relu(self.fc4(x))
        value = self.value_head(x)
        
        # 添加梯度裁剪和数值稳定性
        value = torch.clamp(value, -100, 100)
        
        # 检查数值稳定性
        if torch.isnan(value).any() or torch.isinf(value).any():
            print(f"Warning: value contains nan/inf: {value}")
            value = torch.zeros_like(value)
        
        return value

class HybridActorCritic(nn.Module):
    """混合Actor-Critic网络"""
    def __init__(self, state_dim: int, discrete_action_dim: int, 
                 continuous_action_dim: int, hidden_dim: int = 128,
                 max_power_u: float = 18.0, max_power_n: float = 18.0, 
                 max_power_b: float = 70.0):
        super(HybridActorCritic, self).__init__()
        
        # 离散动作策略网络
        self.discrete_actor = DiscreteActorNetwork(state_dim, discrete_action_dim, hidden_dim)
        
        # 连续动作策略网络
        self.task_allocation_actor = TaskAllocationNetwork(state_dim, hidden_dim)
        self.power_allocation_actor = PowerAllocationNetwork(
            state_dim, max_power_u, max_power_n, max_power_b, hidden_dim
        )
        
        # 价值网络
        self.critic = CriticNetwork(state_dim, hidden_dim)
        
    def forward(self, state):
        """前向传播"""
        discrete_logits = self.discrete_actor(state)
        task_mean, task_log_std = self.task_allocation_actor(state)
        power_mean, power_log_std = self.power_allocation_actor(state)
        value = self.critic(state)
        
        return {
            'discrete_logits': discrete_logits,
            'task_mean': task_mean,
            'task_log_std': task_log_std,
            'power_mean': power_mean,
            'power_log_std': power_log_std,
            'value': value
        }
    
    def get_action(self, state, deterministic=False):
        """获取完整动作"""
        # 获取离散动作
        discrete_action = self.discrete_actor.get_action(state, deterministic)
        
        # 获取连续动作
        task_allocation = self.task_allocation_actor.get_action(state, deterministic)
        power_allocation = self.power_allocation_actor.get_action(state, deterministic)
        
        return {
            'offload_decision': discrete_action,
            'task_allocation': task_allocation,
            'power_allocation': power_allocation
        }
    
    def get_log_probs(self, state, action):
        """获取所有动作的对数概率"""
        discrete_log_prob = self.discrete_actor.get_log_prob(
            state, action['offload_decision']
        )
        task_log_prob = self.task_allocation_actor.get_log_prob(
            state, action['task_allocation']
        )
        power_log_prob = self.power_allocation_actor.get_log_prob(
            state, action['power_allocation']
        )
        
        return {
            'discrete_log_prob': discrete_log_prob,
            'task_log_prob': task_log_prob,
            'power_log_prob': power_log_prob
        }
    
    def get_value(self, state):
        """获取状态价值"""
        return self.critic(state)

class ReplayBuffer:
    """经验回放缓冲区"""
    def __init__(self, capacity: int = 100000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        
    def push(self, state, action, reward, next_state, done, log_probs):
        """存储经验"""
        # 全部转为numpy或标量
        def to_numpy(x):
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy()
            return x
        def to_scalar(x):
            if isinstance(x, torch.Tensor):
                return x.item()
            return x
        # action和log_probs为dict
        action_np = {k: to_numpy(v) if k != 'offload_decision' else to_scalar(v) for k, v in action.items()}
        log_probs_np = {k: to_numpy(v) for k, v in log_probs.items()}
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        
        self.buffer[self.position] = {
            'state': to_numpy(state),
            'action': action_np,
            'reward': float(reward),
            'next_state': to_numpy(next_state),
            'done': bool(done),
            'log_probs': log_probs_np
        }
        
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int):
        """采样经验"""
        batch = np.random.choice(len(self.buffer), batch_size, replace=False)
        
        states = np.stack([self.buffer[i]['state'].squeeze() for i in batch])
        actions = [self.buffer[i]['action'] for i in batch]
        rewards = np.array([self.buffer[i]['reward'] for i in batch], dtype=np.float32)
        next_states = np.stack([self.buffer[i]['next_state'].squeeze() for i in batch])
        dones = np.array([self.buffer[i]['done'] for i in batch], dtype=np.float32)
        log_probs = [self.buffer[i]['log_probs'] for i in batch]
        
        return states, actions, rewards, next_states, dones, log_probs
    
    def __len__(self):
        return len(self.buffer) 