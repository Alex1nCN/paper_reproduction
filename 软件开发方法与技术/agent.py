import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple
from networks import HybridActorCritic, ReplayBuffer

class PPOHybridAgent:
    """PPO混合Actor-Critic智能体"""
    
    def __init__(self, 
                 state_dim: int,
                 discrete_action_dim: int,
                 continuous_action_dim: int,
                 hidden_dim: int = 128,
                 lr: float = 3e-4,
                 gamma: float = 0.64,
                 gae_lambda: float = 0.95,
                 clip_ratio: float = 0.1,
                 value_loss_coef: float = 0.5,
                 entropy_coef: float = 0.01,
                 max_grad_norm: float = 0.5,
                 device: str = 'cpu'):
        
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        
        # 创建网络
        self.policy = HybridActorCritic(
            state_dim, discrete_action_dim, continuous_action_dim, 
            hidden_dim
        ).to(device)
        
        # 创建目标网络
        self.target_policy = HybridActorCritic(
            state_dim, discrete_action_dim, continuous_action_dim, 
            hidden_dim
        ).to(device)
        self.target_policy.load_state_dict(self.policy.state_dict())
        
        # 优化器
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # 经验缓冲区
        self.replay_buffer = ReplayBuffer(capacity=10000)
        
        # 训练统计
        self.training_stats = {
            'policy_loss': [],
            'value_loss': [],
            'entropy_loss': [],
            'total_loss': [],
            'advantages': [],
            'returns': []
        }
    
    def compute_gae(self, rewards: torch.Tensor, values: torch.Tensor, 
                   dones: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算广义优势估计 (GAE)"""
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        
        gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
        
        return advantages, returns
    
    def compute_advantages(self, states: torch.Tensor, rewards: torch.Tensor, 
                          dones: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算优势函数和回报"""
        with torch.no_grad():
            values = self.policy.get_value(states).squeeze(-1)
        
        advantages, returns = self.compute_gae(rewards, values, dones)
        
        # 标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
    
    def update_policy(self, states: torch.Tensor, actions: List[Dict], 
                     old_log_probs: List[Dict], advantages: torch.Tensor, 
                     returns: torch.Tensor, num_epochs: int = 10):
        """更新策略网络"""
        
        # 确保不保留计算图
        advantages = advantages.detach()
        returns = returns.detach()
        states = states.detach()
        
        # 分离动作和对数概率
        discrete_actions = torch.tensor([a['offload_decision'] for a in actions], dtype=torch.long, device=states.device).view(-1)
        task_actions = torch.stack([torch.tensor(a['task_allocation'], dtype=torch.float32, device=states.device).view(-1) for a in actions])
        power_actions = torch.stack([torch.tensor(a['power_allocation'], dtype=torch.float32, device=states.device).view(-1) for a in actions])
        
        old_discrete_log_probs = torch.stack([lp['discrete_log_prob'] if torch.is_tensor(lp['discrete_log_prob']) else torch.tensor(lp['discrete_log_prob'], dtype=torch.float32, device=states.device) for lp in old_log_probs])
        old_task_log_probs = torch.stack([lp['task_log_prob'] if torch.is_tensor(lp['task_log_prob']) else torch.tensor(lp['task_log_prob'], dtype=torch.float32, device=states.device) for lp in old_log_probs])
        old_power_log_probs = torch.stack([lp['power_log_prob'] if torch.is_tensor(lp['power_log_prob']) else torch.tensor(lp['power_log_prob'], dtype=torch.float32, device=states.device) for lp in old_log_probs])
        
        # 检查旧对数概率的数值稳定性
        if torch.isnan(old_discrete_log_probs).any() or torch.isinf(old_discrete_log_probs).any():
            print(f"Warning: old_discrete_log_probs contains nan/inf: {old_discrete_log_probs}")
            old_discrete_log_probs = torch.zeros_like(old_discrete_log_probs)
        
        if torch.isnan(old_task_log_probs).any() or torch.isinf(old_task_log_probs).any():
            print(f"Warning: old_task_log_probs contains nan/inf: {old_task_log_probs}")
            old_task_log_probs = torch.zeros_like(old_task_log_probs)
        
        if torch.isnan(old_power_log_probs).any() or torch.isinf(old_power_log_probs).any():
            print(f"Warning: old_power_log_probs contains nan/inf: {old_power_log_probs}")
            old_power_log_probs = torch.zeros_like(old_power_log_probs)
        
        for epoch in range(num_epochs):
            # 获取新的动作和对数概率
            new_log_probs = self.policy.get_log_probs(states, {
                'offload_decision': discrete_actions,
                'task_allocation': task_actions,
                'power_allocation': power_actions
            })
            
            # 检查新对数概率的数值稳定性
            if torch.isnan(new_log_probs['discrete_log_prob']).any() or torch.isinf(new_log_probs['discrete_log_prob']).any():
                print(f"Warning: new_discrete_log_prob contains nan/inf: {new_log_probs['discrete_log_prob']}")
                new_log_probs['discrete_log_prob'] = torch.zeros_like(new_log_probs['discrete_log_prob'])
            
            if torch.isnan(new_log_probs['task_log_prob']).any() or torch.isinf(new_log_probs['task_log_prob']).any():
                print(f"Warning: new_task_log_prob contains nan/inf: {new_log_probs['task_log_prob']}")
                new_log_probs['task_log_prob'] = torch.zeros_like(new_log_probs['task_log_prob'])
            
            if torch.isnan(new_log_probs['power_log_prob']).any() or torch.isinf(new_log_probs['power_log_prob']).any():
                print(f"Warning: new_power_log_prob contains nan/inf: {new_log_probs['power_log_prob']}")
                new_log_probs['power_log_prob'] = torch.zeros_like(new_log_probs['power_log_prob'])
            
            # 计算比率
            discrete_ratio = torch.exp(new_log_probs['discrete_log_prob'] - old_discrete_log_probs)
            task_ratio = torch.exp(new_log_probs['task_log_prob'] - old_task_log_probs)
            power_ratio = torch.exp(new_log_probs['power_log_prob'] - old_power_log_probs)
            
            # 检查比率的数值稳定性
            if torch.isnan(discrete_ratio).any() or torch.isinf(discrete_ratio).any():
                print(f"Warning: discrete_ratio contains nan/inf: {discrete_ratio}")
                discrete_ratio = torch.ones_like(discrete_ratio)
            
            if torch.isnan(task_ratio).any() or torch.isinf(task_ratio).any():
                print(f"Warning: task_ratio contains nan/inf: {task_ratio}")
                task_ratio = torch.ones_like(task_ratio)
            
            if torch.isnan(power_ratio).any() or torch.isinf(power_ratio).any():
                print(f"Warning: power_ratio contains nan/inf: {power_ratio}")
                power_ratio = torch.ones_like(power_ratio)
            
            # 计算裁剪后的目标
            discrete_clipped_ratio = torch.clamp(discrete_ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
            task_clipped_ratio = torch.clamp(task_ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
            power_clipped_ratio = torch.clamp(power_ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
            
            # 策略损失
            discrete_policy_loss = -torch.min(
                discrete_ratio * advantages,
                discrete_clipped_ratio * advantages
            ).mean()
            
            task_policy_loss = -torch.min(
                task_ratio * advantages,
                task_clipped_ratio * advantages
            ).mean()
            
            power_policy_loss = -torch.min(
                power_ratio * advantages,
                power_clipped_ratio * advantages
            ).mean()
            
            policy_loss = discrete_policy_loss + task_policy_loss + power_policy_loss
            
            # 价值损失 - 统一维度
            values = self.policy.get_value(states).view(-1)
            returns_flat = returns.view(-1)
            value_loss = F.mse_loss(values, returns_flat)
            
            # 熵损失 (鼓励探索)
            discrete_entropy = self._compute_entropy(new_log_probs['discrete_log_prob'])
            task_entropy = self._compute_entropy(new_log_probs['task_log_prob'])
            power_entropy = self._compute_entropy(new_log_probs['power_log_prob'])
            entropy_loss = -(discrete_entropy + task_entropy + power_entropy).mean()
            
            # 总损失
            total_loss = (policy_loss + 
                         self.value_loss_coef * value_loss + 
                         self.entropy_coef * entropy_loss)
            
            # 检查损失的数值稳定性
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print(f"Warning: total_loss is nan/inf: {total_loss}")
                continue
            
            # 反向传播
            self.optimizer.zero_grad()
            total_loss.backward()
            
            # 梯度裁剪
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            
            self.optimizer.step()
            
            # 记录统计信息
            self.training_stats['policy_loss'].append(policy_loss.item())
            self.training_stats['value_loss'].append(value_loss.item())
            self.training_stats['entropy_loss'].append(entropy_loss.item())
            self.training_stats['total_loss'].append(total_loss.item())
            self.training_stats['advantages'].append(advantages.mean().item())
            self.training_stats['returns'].append(returns.mean().item())
    
    def _compute_entropy(self, log_probs: torch.Tensor) -> torch.Tensor:
        """计算熵"""
        probs = torch.exp(log_probs)
        entropy = -torch.sum(probs * log_probs, dim=-1)
        return entropy
    
    def select_action(self, state: torch.Tensor, deterministic: bool = False) -> Dict:
        """选择动作"""
        with torch.no_grad():
            action = self.policy.get_action(state, deterministic)
        return action
    
    def store_transition(self, state: torch.Tensor, action: Dict, reward: float, 
                        next_state: torch.Tensor, done: bool, log_probs: Dict):
        """存储经验"""
        self.replay_buffer.push(state, action, reward, next_state, done, log_probs)
    
    def train(self, batch_size: int = 64, num_epochs: int = 10):
        """训练智能体"""
        if len(self.replay_buffer) < batch_size:
            return
        
        # 采样经验
        states, actions, rewards, next_states, dones, old_log_probs = self.replay_buffer.sample(batch_size)
        # 全部转为新的tensor，保证无autograd依赖
        device = self.device
        states = torch.tensor(states, dtype=torch.float32, device=device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=device)
        dones = torch.tensor(dones, dtype=torch.float32, device=device)
        # actions和old_log_probs保持原始dict列表
        
        # 计算优势函数
        advantages, returns = self.compute_advantages(states, rewards, dones)
        
        # 更新策略
        self.update_policy(states, actions, old_log_probs, advantages, returns, num_epochs)
        
        # 更新目标网络
        self._update_target_network()
    
    def _update_target_network(self, tau: float = 0.01):
        """软更新目标网络"""
        for target_param, param in zip(self.target_policy.parameters(), self.policy.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
    
    def save_model(self, path: str):
        """保存模型"""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'target_policy_state_dict': self.target_policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_stats': self.training_stats
        }, path)
    
    def load_model(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.target_policy.load_state_dict(checkpoint['target_policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_stats = checkpoint['training_stats']
    
    def get_training_stats(self) -> Dict:
        """获取训练统计信息"""
        return self.training_stats
    
    def reset_stats(self):
        """重置统计信息"""
        for key in self.training_stats:
            self.training_stats[key] = []

class A3CAgent:
    """A3C智能体 (用于比较)"""
    def __init__(self, state_dim: int, discrete_action_dim: int, 
                 continuous_action_dim: int, lr: float = 3e-4, device: str = 'cpu'):
        self.device = device
        self.policy = HybridActorCritic(state_dim, discrete_action_dim, continuous_action_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
    def select_action(self, state: torch.Tensor, deterministic: bool = False) -> Dict:
        with torch.no_grad():
            action = self.policy.get_action(state, deterministic)
        return action
    
    def update(self, states: torch.Tensor, actions: List[Dict], rewards: torch.Tensor, 
               next_states: torch.Tensor, dones: torch.Tensor):
        """A3C更新"""
        # 简化的A3C实现
        values = self.policy.get_value(states).squeeze(-1)
        next_values = self.policy.get_value(next_states).squeeze(-1)
        
        # 计算优势
        advantages = rewards + 0.64 * next_values * (1 - dones) - values
        
        # 策略损失
        log_probs = self.policy.get_log_probs(states, actions[0])  # 简化处理
        policy_loss = -(log_probs['discrete_log_prob'] * advantages.detach()).mean()
        
        # 价值损失
        value_loss = F.mse_loss(values, rewards + 0.64 * next_values * (1 - dones))
        
        # 总损失
        total_loss = policy_loss + 0.5 * value_loss
        
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

class A2CAgent:
    """A2C智能体 (用于比较)"""
    def __init__(self, state_dim: int, discrete_action_dim: int, 
                 continuous_action_dim: int, lr: float = 3e-4, device: str = 'cpu'):
        self.device = device
        self.policy = HybridActorCritic(state_dim, discrete_action_dim, continuous_action_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
    def select_action(self, state: torch.Tensor, deterministic: bool = False) -> Dict:
        with torch.no_grad():
            action = self.policy.get_action(state, deterministic)
        return action
    
    def update(self, states: torch.Tensor, actions: List[Dict], rewards: torch.Tensor, 
               next_states: torch.Tensor, dones: torch.Tensor):
        """A2C更新"""
        # 简化的A2C实现
        values = self.policy.get_value(states).squeeze(-1)
        next_values = self.policy.get_value(next_states).squeeze(-1)
        
        # 计算优势
        advantages = rewards + 0.64 * next_values * (1 - dones) - values
        
        # 策略损失
        log_probs = self.policy.get_log_probs(states, actions[0])  # 简化处理
        policy_loss = -(log_probs['discrete_log_prob'] * advantages.detach()).mean()
        
        # 价值损失
        value_loss = F.mse_loss(values, rewards + 0.64 * next_values * (1 - dones))
        
        # 总损失
        total_loss = policy_loss + 0.5 * value_loss
        
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step() 