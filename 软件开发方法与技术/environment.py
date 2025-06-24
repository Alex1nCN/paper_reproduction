import numpy as np
import torch
from typing import Tuple, Dict, List
import math

class MECEnvironment:
    """
    Multi-Access Edge Computing Environment
    实现论文中的环境建模，包括状态空间、动作空间和奖励函数
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
        
        # 任务参数增强
        self.task_priority_levels = ['low', 'medium', 'high']
        self.task_priority_weights = {'low': 1.0, 'medium': 1.5, 'high': 2.0}
        self.task_arrival_rates = {'low': 0.2, 'medium': 0.1, 'high': 0.05}
        self.task_queue = []  # 任务队列
        self.max_queue_size = 10
        
        # 任务大小和复杂度范围
        self.task_size_range = (100, 1000)  # KB
        self.task_complexity_range = (100, 1000)  # cycles/bit
        
        # 能量模型参数
        self.cpu_freq_levels = np.linspace(0.5, 2.0, 5)  # GHz
        self.cpu_power_coefficient = 1e-9  # 能耗系数
        self.transmission_power_coefficient = 1e-6
        self.battery_efficiency = 0.9
        self.energy_harvest_rate = 0.1  # 能量收集速率
        
        # 社会关系参数
        self.social_decay_factor = 0.95  # 社会关系衰减因子
        self.social_increase_rate = 0.1  # 社会关系增强率
        self.cooperation_history = np.zeros((num_users, num_helpers))
        self.social_strength = np.random.uniform(0.1, 1.0, (num_users, num_helpers))  # 初始化社会关系矩阵
        
        # 性能统计
        self.stats = {
            'completed_tasks': 0,
            'failed_tasks': 0,
            'total_energy_consumption': 0,
            'total_delay': 0,
            'cooperation_count': 0,
            'constraint_violations': {'power': 0, 'task': 0, 'energy': 0},
            'average_queue_length': 0,
            'task_completion_rate': 0
        }
        
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
        
        # 新增奖励权重参数
        self.social_weight = rho  # 社会关系权重
        self.energy_weight = 0.3  # 能耗权重
        self.delay_weight = 0.2   # 延迟权重
        self.charging_weight = 0.1  # 充电奖励权重
        
        # 用于动态归一化的统计
        self.max_energy_consumption = 1.0
        self.max_delay = 1.0
        self.energy_history = []
        self.delay_history = []
        self.history_window = 1000  # 统计窗口大小
        
        # 约束惩罚权重
        self.power_penalty = 5.0
        self.task_penalty = 3.0
        self.energy_penalty = 8.0
        
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
        """生成具有优先级的任务"""
        # 根据不同优先级的到达率生成任务
        priority = np.random.choice(self.task_priority_levels, p=[0.6, 0.3, 0.1])
        if np.random.random() < self.task_arrival_rates[priority]:
            task_size = np.random.uniform(*self.task_size_range) * 1024  # bits
            task_complexity = np.random.uniform(*self.task_complexity_range)
            deadline = task_size * task_complexity * 1e-9 * 2  # 简单的截止时间估计
            
            task = {
                'size': task_size,
                'complexity': task_complexity,
                'priority': priority,
                'deadline': deadline,
                'arrival_time': self.step_count,
                'user_id': np.random.randint(0, self.num_users)
            }
            
            # 将任务加入队列
            if len(self.task_queue) < self.max_queue_size:
                self.task_queue.append(task)
        
        # 如果队列非空，返回最高优先级的任务
        if self.task_queue:
            highest_priority_task = max(self.task_queue, 
                                      key=lambda x: self.task_priority_weights[x['priority']])
            self.task_queue.remove(highest_priority_task)
            return highest_priority_task
        
        # 如果队列为空，生成一个普通任务
        return {
            'size': np.random.uniform(*self.task_size_range) * 1024,
            'complexity': np.random.uniform(*self.task_complexity_range),
            'priority': 'medium',
            'deadline': float('inf'),
            'arrival_time': self.step_count,
            'user_id': np.random.randint(0, self.num_users)
        }
    
    def _calculate_energy_consumption(self, task: Dict, action: Dict) -> float:
        """增强的能耗计算模型"""
        if action['offload_decision'] == 0:  # 本地处理
            # 选择CPU频率
            cpu_freq = self.cpu_freq_levels[np.random.randint(len(self.cpu_freq_levels))]
            # 计算能耗：P = α * f^3 * t
            processing_time = task['size'] * task['complexity'] / (cpu_freq * 1e9)
            local_energy = self.cpu_power_coefficient * (cpu_freq ** 3) * processing_time
            
            # 考虑电池效率
            return local_energy / self.battery_efficiency
        
        else:  # 卸载处理
            # 传输能耗：P = β * size * (1/rate)
            channel_gain = self._get_channel_gains()[0]
            power = action['power_allocation'][0]
            transmission_rate = self._calculate_transmission_rate(power, channel_gain)
            transmission_energy = self.transmission_power_coefficient * task['size'] / transmission_rate
            
            # 考虑电池效率
            return transmission_energy / self.battery_efficiency
    
    def _calculate_charging_energy(self, action: Dict) -> float:
        """计算充电获取能量"""
        # 简化的无线充电模型
        power_allocation = np.array(action['power_allocation']).reshape(-1)
        charging_power = power_allocation[2]  # PB
        charging_efficiency = 0.7
        return charging_power * charging_efficiency * self.time_slot
    
    def _calculate_delay(self, task: Dict, action: Dict) -> float:
        """计算任务处理延迟"""
        if action['offload_decision'] == 0:  # 本地处理
            # 本地处理延迟 = 任务大小 * 复杂度 / 本地计算能力
            local_computing_capacity = 1e9  # 1GHz
            delay = task['size'] * task['complexity'] / local_computing_capacity
        else:  # 卸载处理
            # 传输延迟 = 任务大小 / 传输速率
            channel_gain = self._get_channel_gains()[0]  # 简化：使用第一个信道
            power = action['power_allocation'][0]
            transmission_rate = self._calculate_transmission_rate(power, channel_gain)
            delay = task['size'] / transmission_rate
        
        return delay
    
    def _update_statistics(self, energy: float, delay: float):
        """更新统计信息用于动态归一化"""
        self.energy_history.append(energy)
        self.delay_history.append(delay)
        
        # 保持固定窗口大小
        if len(self.energy_history) > self.history_window:
            self.energy_history.pop(0)
        if len(self.delay_history) > self.history_window:
            self.delay_history.pop(0)
        
        # 更新最大值
        if len(self.energy_history) > 0:
            self.max_energy_consumption = max(max(self.energy_history), 1e-6)
        if len(self.delay_history) > 0:
            self.max_delay = max(max(self.delay_history), 1e-6)
    
    def _check_constraints(self, action: Dict) -> Tuple[bool, Dict]:
        """检查约束条件并返回具体违反信息"""
        violations = {
            'power': False,
            'task': False,
            'energy': False
        }
        
        # C1: 功率约束
        power_allocation = action['power_allocation']
        if (power_allocation[0] > self.max_power_u or 
            power_allocation[1] > self.max_power_n or 
            power_allocation[2] > self.max_power_b):
            violations['power'] = True
        
        # C2: 任务分配约束
        task_allocation = action['task_allocation']
        if not np.isclose(np.sum(task_allocation), 1.0, atol=1e-6):
            violations['task'] = True
        
        # C3: 能量约束
        if np.any(self.user_energies <= 0):
            violations['energy'] = True
        
        return any(violations.values()), violations
    
    def reset(self) -> np.ndarray:
        """重置环境"""
        # 重置能量
        self.user_energies = np.full(self.num_users, self.max_energy)
        self.helper_energies = np.full(self.num_helpers, self.max_energy)
        
        # 重置步数计数器
        self.step_count = 0
        
        # 重置统计信息
        self.energy_history = []
        self.delay_history = []
        self.max_energy_consumption = 1.0
        self.max_delay = 1.0
        
        # 重置任务队列
        self.task_queue = []
        
        # 重置统计信息
        self.stats = {
            'completed_tasks': 0,
            'failed_tasks': 0,
            'total_energy_consumption': 0,
            'total_delay': 0,
            'cooperation_count': 0,
            'constraint_violations': {'power': 0, 'task': 0, 'energy': 0},
            'average_queue_length': 0,
            'task_completion_rate': 0
        }
        
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
        """环境步进增强版"""
        # 更新步数（移到函数开始处）
        if not hasattr(self, 'step_count'):
            self.step_count = 0
        self.step_count += 1
        
        # 解析动作
        offload_decision = action['offload_decision']  # 离散动作
        task_allocation = np.array(action['task_allocation']).reshape(-1)
        power_allocation = np.array(action['power_allocation']).reshape(-1)
        action['task_allocation'] = task_allocation
        action['power_allocation'] = power_allocation
        
        # 生成任务
        task = self._generate_task()
        
        # 计算能耗和延迟
        energy_consumption = self._calculate_energy_consumption(task, action)
        delay = self._calculate_delay(task, action)
        
        # 判断任务是否成功完成
        success = delay <= task['deadline']
        
        # 计算充电能量
        charging_energy = self._calculate_charging_energy(action)
        
        # 更新能量（考虑能量收集）
        user_id = task['user_id']
        energy_harvested = self.energy_harvest_rate * self.time_slot
        self.user_energies[user_id] = max(0, min(
            self.max_energy,
            self.user_energies[user_id] - energy_consumption + charging_energy + energy_harvested
        ))
        
        # 更新社会关系
        if action['offload_decision'] > 0:
            helper_ids = [i for i, alloc in enumerate(action['task_allocation']) if alloc > 0]
            cooperation_level = action['task_allocation'].max()
            self._update_social_relations(user_id, helper_ids, success, cooperation_level)
            if success:
                self.stats['cooperation_count'] += 1
        
        # 更新统计信息
        self._update_stats(task, energy_consumption, delay, success)
        
        # 归一化
        norm_energy_consumption = np.clip(energy_consumption / self.max_energy_consumption, 0, 1)
        norm_delay = np.clip(delay / self.max_delay, 0, 1)
        
        # 归一化充电能量
        norm_charging = charging_energy / self.max_power_b
        
        # 计算基础奖励
        reward = (
            self.social_weight * self.cooperation_history[user_id, :].mean() +           # 社会关系奖励
            self.energy_weight * (1 - norm_energy_consumption) +  # 能耗奖励
            self.delay_weight * (1 - norm_delay) +          # 延迟奖励
            self.charging_weight * norm_charging            # 充电奖励
        )
        
        # 检查约束并应用惩罚
        constraints_violated, violations = self._check_constraints(action)
        if constraints_violated:
            penalty = 0
            if violations['power']:
                penalty += self.power_penalty
            if violations['task']:
                penalty += self.task_penalty
            if violations['energy']:
                penalty += self.energy_penalty
            reward -= penalty
        
        # 限制奖励范围
        reward = np.clip(reward, -100, 100)
        
        # 获取新状态
        next_state = self._get_state()
        
        # 检查是否结束
        done = np.any(self.user_energies <= 0)
        
        # 如果超过100步还没结束，强制结束
        if self.step_count >= 100:
            done = True
        
        # 更新统计信息
        self._update_stats(task, energy_consumption, delay, success)
        
        info = {
            'energy_consumption': energy_consumption,
            'charging_energy': charging_energy,
            'social_strength': self.cooperation_history[user_id, :].mean(),
            'delay': delay,
            'norm_energy': norm_energy_consumption,
            'norm_delay': norm_delay,
            'constraints_violated': constraints_violated,
            'violations': violations,
            'step_count': self.step_count,
            'task_priority': task['priority'],
            'task_success': success,
            'energy_harvested': energy_harvested,
            'queue_length': len(self.task_queue),
            'completion_rate': self.stats['task_completion_rate'],
            'cooperation_count': self.stats['cooperation_count'],
            'average_queue_length': self.stats['average_queue_length']
        }
        
        return next_state, reward, done, info
    
    def get_action_space(self) -> Dict:
        """获取动作空间"""
        return {
            'discrete': self.action_dim_discrete,
            'continuous': self.action_dim_continuous
        }
    
    def get_state_space(self) -> int:
        """获取状态空间维度"""
        return self.state_dim 

    def _update_social_relations(self, user_id: int, helper_ids: List[int], 
                               success: bool, cooperation_level: float):
        """更新社会关系强度"""
        for helper_id in helper_ids:
            if success:
                # 成功合作增强社会关系
                self.social_strength[user_id, helper_id] = min(
                    1.0,
                    self.social_strength[user_id, helper_id] * (1 + self.social_increase_rate * cooperation_level)
                )
                self.cooperation_history[user_id, helper_id] += 1
            else:
                # 失败减弱社会关系
                self.social_strength[user_id, helper_id] *= self.social_decay_factor
        
        # 定期衰减所有社会关系
        if self.step_count % 100 == 0:
            self.social_strength *= self.social_decay_factor

    def _update_stats(self, task: Dict, energy: float, delay: float, success: bool):
        """更新性能统计"""
        self.stats['total_energy_consumption'] += energy
        self.stats['total_delay'] += delay
        
        if success and delay <= task['deadline']:
            self.stats['completed_tasks'] += 1
        else:
            self.stats['failed_tasks'] += 1
        
        total_tasks = self.stats['completed_tasks'] + self.stats['failed_tasks']
        if total_tasks > 0:
            self.stats['task_completion_rate'] = self.stats['completed_tasks'] / total_tasks
        
        # 修复除零错误：确保step_count至少为1
        current_step = max(1, self.step_count)
        self.stats['average_queue_length'] = (
            self.stats['average_queue_length'] * (current_step - 1) + len(self.task_queue)
        ) / current_step 