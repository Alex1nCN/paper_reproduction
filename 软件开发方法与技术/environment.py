import numpy as np
import torch
from typing import Tuple, Dict, List
import math

class MECEnvironment:
    """
    Multi-Access Edge Computing Environment
    论文中的环境建模，包括状态空间、动作空间和奖励函数
    """
    
    def __init__(self, 
                 num_users: int = 5,
                 num_helpers: int = 3,
                 radius: float = 100.0,
                 helper_radius: Tuple[float, float] = (10.0, 50.0),
                 time_slot: float = 1.0,
                 max_power_u: float = 18.0,  # dBm
                 max_power_n: float = 18.0,  # dBm
                 max_power_b: float = 70.0,  # dBm
                 discount_factor: float = 0.64,
                 rho: float = 0.5,
                 max_energy: float = 1000.0):
        
        # 环境参数
        self.num_users = num_users
        self.num_helpers = num_helpers
        self.radius = radius
        self.helper_radius = helper_radius
        self.time_slot = time_slot
        self.max_power_u = max_power_u
        self.max_power_n = max_power_n
        self.max_power_b = max_power_b
        self.gamma = discount_factor
        self.rho = rho
        self.max_energy = max_energy
        
        # 设备位置初始化
        self.user_positions = self._initialize_user_positions()
        self.helper_positions = self._initialize_helper_positions()
        self.mec_position = np.array([0.0, 0.0])  # MEC服务器位于圆心
        
        # 能量参数
        self.user_energies = np.full(num_users, self.max_energy)
        self.helper_energies = np.full(num_helpers, self.max_energy)
        
        # 信道参数
        self.noise_power = -174.0  # dBm/Hz
        self.bandwidth = 1e6  # 1MHz
        self.path_loss_exponent = 3.0
        
        # 任务参数
        self.task_arrival_rate = 0.1  # 泊松分布参数
        self.task_size_range = (100, 1000)  # KB
        self.task_complexity_range = (100, 1000)  # cycles/bit
        
        # 社会关系强度矩阵 (随机初始化)
        self.social_strength = np.random.uniform(0.1, 1.0, (num_users, num_helpers))
        
        # 状态空间维度: 用户能量 + 辅助设备能量 + 5类信道增益
        # 5类信道: hUN(用户到MEC), hUB(用户到辅助), hNB(MEC到辅助), hBU(辅助到用户), hNU(MEC到用户)
        total_channels = (num_users +  # hUN
                         num_users * num_helpers +  # hUB
                         num_helpers +  # hNB
                         num_helpers * num_users +  # hBU
                         num_users)  # hNU
        self.state_dim = num_users + num_helpers + total_channels
        self.action_dim_discrete = 4  # 本地、D2D、辅助D2D、MEC
        self.action_dim_continuous = 6  # 3个分配比例 + 3个功率值
        
    def _initialize_user_positions(self) -> np.ndarray:
        """初始化用户设备位置"""
        positions = []
        for i in range(self.num_users):
            angle = 2 * np.pi * i / self.num_users
            r = np.random.uniform(0, self.radius)
            x = r * np.cos(angle)
            y = r * np.sin(angle)
            positions.append([x, y])
        return np.array(positions)
    
    def _initialize_helper_positions(self) -> np.ndarray:
        """初始化辅助设备位置"""
        positions = []
        for i in range(self.num_helpers):
            angle = 2 * np.pi * i / self.num_helpers
            r = np.random.uniform(self.helper_radius[0], self.helper_radius[1])
            x = r * np.cos(angle)
            y = r * np.sin(angle)
            positions.append([x, y])
        return np.array(positions)
    
    def _calculate_distance(self, pos1: np.ndarray, pos2: np.ndarray) -> float:
        """计算两点间距离"""
        return np.linalg.norm(pos1 - pos2)
    
    def _calculate_channel_gain(self, distance: float) -> float:
        """计算信道增益 (Shannon公式)"""
        # 路径损耗模型
        path_loss = 10 * self.path_loss_exponent * np.log10(distance + 1)
        # 瑞利衰落
        rayleigh_fading = np.random.exponential(1.0)
        return 10 ** (-path_loss / 10) * rayleigh_fading
    
    def _get_channel_gains(self) -> np.ndarray:
        """获取5类信道增益"""
        gains = []
        
        # hUN: 用户到MEC
        for user_pos in self.user_positions:
            dist = self._calculate_distance(user_pos, self.mec_position)
            gains.append(self._calculate_channel_gain(dist))
        
        # hUB: 用户到辅助设备
        for user_pos in self.user_positions:
            for helper_pos in self.helper_positions:
                dist = self._calculate_distance(user_pos, helper_pos)
                gains.append(self._calculate_channel_gain(dist))
        
        # hNB: MEC到辅助设备
        for helper_pos in self.helper_positions:
            dist = self._calculate_distance(self.mec_position, helper_pos)
            gains.append(self._calculate_channel_gain(dist))
        
        # hBU: 辅助设备到用户
        for helper_pos in self.helper_positions:
            for user_pos in self.user_positions:
                dist = self._calculate_distance(helper_pos, user_pos)
                gains.append(self._calculate_channel_gain(dist))
        
        # hNU: MEC到用户
        for user_pos in self.user_positions:
            dist = self._calculate_distance(self.mec_position, user_pos)
            gains.append(self._calculate_channel_gain(dist))
        
        return np.array(gains)
    
    def _calculate_transmission_rate(self, power: float, channel_gain: float) -> float:
        """计算传输速率 (Shannon公式)"""
        snr = (power * channel_gain) / (10 ** (self.noise_power / 10))
        return self.bandwidth * np.log2(1 + snr)
    
    def _generate_task(self) -> Dict:
        """生成任务"""
        task_size = np.random.uniform(*self.task_size_range) * 1024  # 转换为bits
        task_complexity = np.random.uniform(*self.task_complexity_range)
        return {
            'size': task_size,
            'complexity': task_complexity,
            'user_id': np.random.randint(0, self.num_users)
        }
    
    def _calculate_energy_consumption(self, task: Dict, action: Dict) -> float:
        """计算能耗"""
        # 简化的能耗模型
        local_energy = task['complexity'] * 1e-9  # 本地计算能耗
        transmission_energy = task['size'] / 1e6  # 传输能耗
        
        total_energy = 0
        if action['offload_decision'] == 0:  # 本地处理
            total_energy = local_energy
        else:  # 卸载处理
            total_energy = transmission_energy
            
        return total_energy
    
    def _calculate_charging_energy(self, action: Dict) -> float:
        """计算充电获取能量"""
        # 简化的无线充电模型
        power_allocation = np.array(action['power_allocation']).reshape(-1)
        charging_power = power_allocation[2]  # PB
        charging_efficiency = 0.7
        return charging_power * charging_efficiency * self.time_slot
    
    def reset(self) -> np.ndarray:
        """重置环境"""
        # 重置能量
        self.user_energies = np.full(self.num_users, self.max_energy)
        self.helper_energies = np.full(self.num_helpers, self.max_energy)
        
        # 重置步数计数器
        self.step_count = 0
        
        # 获取初始状态
        state = self._get_state()
        return state
    
    def _get_state(self) -> np.ndarray:
        """获取当前状态"""
        # 归一化能量
        normalized_user_energy = self.user_energies / self.max_energy
        normalized_helper_energy = self.helper_energies / self.max_energy
        
        # 检查能量值的数值稳定性
        if np.isnan(normalized_user_energy).any() or np.isinf(normalized_user_energy).any():
            print(f"Warning: normalized_user_energy contains nan/inf: {normalized_user_energy}")
            normalized_user_energy = np.clip(normalized_user_energy, 0, 1)
        
        if np.isnan(normalized_helper_energy).any() or np.isinf(normalized_helper_energy).any():
            print(f"Warning: normalized_helper_energy contains nan/inf: {normalized_helper_energy}")
            normalized_helper_energy = np.clip(normalized_helper_energy, 0, 1)
        
        # 获取信道增益
        channel_gains = self._get_channel_gains()
        
        # 检查信道增益的数值稳定性
        if np.isnan(channel_gains).any() or np.isinf(channel_gains).any():
            print(f"Warning: channel_gains contains nan/inf: {channel_gains}")
            channel_gains = np.clip(channel_gains, 1e-10, 1e10)
        
        # 组合状态
        state = np.concatenate([
            normalized_user_energy,
            normalized_helper_energy,
            channel_gains
        ])
        
        # 最终检查
        if np.isnan(state).any() or np.isinf(state).any():
            print(f"Warning: final state contains nan/inf: {state}")
            state = np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=0.0)
        
        return state
    
    def step(self, action: Dict) -> Tuple[np.ndarray, float, bool, Dict]:
        """环境步进"""
        # 解析动作
        offload_decision = action['offload_decision']  # 离散动作
        task_allocation = np.array(action['task_allocation']).reshape(-1)
        power_allocation = np.array(action['power_allocation']).reshape(-1)
        action['task_allocation'] = task_allocation
        action['power_allocation'] = power_allocation
        
        # 生成任务
        task = self._generate_task()
        
        # 计算能耗
        energy_consumption = self._calculate_energy_consumption(task, action)
        
        # 计算充电能量
        charging_energy = self._calculate_charging_energy(action)
        
        # 更新能量
        user_id = task['user_id']
        self.user_energies[user_id] = max(0, self.user_energies[user_id] - energy_consumption + charging_energy)
        
        # 计算奖励
        social_strength = self.social_strength[user_id, :].mean() if offload_decision > 0 else 0
        energy_efficiency = charging_energy - energy_consumption
        
        # 检查奖励组件的数值稳定性
        if np.isnan(social_strength) or np.isinf(social_strength):
            print(f"Warning: social_strength is nan/inf: {social_strength}")
            social_strength = 0.0
        
        if np.isnan(energy_efficiency) or np.isinf(energy_efficiency):
            print(f"Warning: energy_efficiency is nan/inf: {energy_efficiency}")
            energy_efficiency = 0.0
        
        reward = self.rho * social_strength + (1 - self.rho) * energy_efficiency
        
        # 检查最终奖励的数值稳定性
        if np.isnan(reward) or np.isinf(reward):
            print(f"Warning: reward is nan/inf: {reward}")
            reward = 0.0
        
        # 限制奖励范围，防止数值过大
        reward = np.clip(reward, -100, 100)
        
        # 检查约束条件
        constraints_violated = self._check_constraints(action)
        if constraints_violated:
            reward -= 10.0  # 惩罚违反约束的行为
        
        # 获取新状态
        next_state = self._get_state()
        
        # 检查是否结束 - 添加最大步数限制
        done = np.any(self.user_energies <= 0)
        
        # 添加episode长度限制，避免无限循环
        if not hasattr(self, 'step_count'):
            self.step_count = 0
        self.step_count += 1
        
        # 如果超过100步还没结束，强制结束
        if self.step_count >= 100:
            done = True
        
        info = {
            'energy_consumption': energy_consumption,
            'charging_energy': charging_energy,
            'social_strength': social_strength,
            'constraints_violated': constraints_violated,
            'step_count': self.step_count
        }
        
        return next_state, reward, done, info
    
    def _check_constraints(self, action: Dict) -> bool:
        """检查约束条件 C1-C8"""
        # C1: 功率约束
        power_allocation = action['power_allocation']
        if (power_allocation[0] > self.max_power_u or 
            power_allocation[1] > self.max_power_n or 
            power_allocation[2] > self.max_power_b):
            return True
        
        # C2: 任务分配约束
        task_allocation = action['task_allocation']
        if not np.isclose(np.sum(task_allocation), 1.0, atol=1e-6):
            return True
        
        # C3: 能量约束
        if np.any(self.user_energies <= 0):
            return True
        
        return False
    
    def get_action_space(self) -> Dict:
        """获取动作空间"""
        return {
            'discrete': self.action_dim_discrete,
            'continuous': self.action_dim_continuous
        }
    
    def get_state_space(self) -> int:
        """获取状态空间维度"""
        return self.state_dim 