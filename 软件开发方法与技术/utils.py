import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple
import os
from datetime import datetime

def preprocess_state(state: np.ndarray) -> torch.Tensor:
    """预处理状态"""
    return torch.FloatTensor(state).unsqueeze(0)

def postprocess_action(action: Dict) -> Dict:
    """后处理动作"""
    processed_action = {}
    for key, value in action.items():
        if isinstance(value, torch.Tensor):
            processed_action[key] = value.cpu().numpy()
        else:
            processed_action[key] = value
    return processed_action

def calculate_metrics(rewards: List[float], energy_consumptions: List[float], 
                     charging_energies: List[float], social_strengths: List[float]) -> Dict:
    """计算评估指标"""
    metrics = {
        'mean_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'total_reward': np.sum(rewards),
        'mean_energy_consumption': np.mean(energy_consumptions),
        'mean_charging_energy': np.mean(charging_energies),
        'energy_efficiency': np.mean(charging_energies) - np.mean(energy_consumptions),
        'mean_social_strength': np.mean(social_strengths),
        'convergence_step': _find_convergence_step(rewards)
    }
    return metrics

def _find_convergence_step(rewards: List[float], window_size: int = 50, 
                          threshold: float = 0.01) -> int:
    """找到收敛步数"""
    if len(rewards) < window_size:
        return len(rewards)
    
    for i in range(window_size, len(rewards)):
        window_rewards = rewards[i-window_size:i]
        if np.std(window_rewards) < threshold:
            return i
    
    return len(rewards)

def plot_training_curves(training_stats: Dict, save_path: str = None):
    """绘制训练曲线"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Training Curves', fontsize=16)
    
    # 损失曲线
    axes[0, 0].plot(training_stats['policy_loss'], label='Policy Loss')
    axes[0, 0].set_title('Policy Loss')
    axes[0, 0].set_xlabel('Update Step')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    
    axes[0, 1].plot(training_stats['value_loss'], label='Value Loss', color='orange')
    axes[0, 1].set_title('Value Loss')
    axes[0, 1].set_xlabel('Update Step')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    
    axes[0, 2].plot(training_stats['total_loss'], label='Total Loss', color='red')
    axes[0, 2].set_title('Total Loss')
    axes[0, 2].set_xlabel('Update Step')
    axes[0, 2].set_ylabel('Loss')
    axes[0, 2].legend()
    
    # 优势函数和回报
    axes[1, 0].plot(training_stats['advantages'], label='Advantages', color='green')
    axes[1, 0].set_title('Advantages')
    axes[1, 0].set_xlabel('Update Step')
    axes[1, 0].set_ylabel('Advantage')
    axes[1, 0].legend()
    
    axes[1, 1].plot(training_stats['returns'], label='Returns', color='purple')
    axes[1, 1].set_title('Returns')
    axes[1, 1].set_xlabel('Update Step')
    axes[1, 1].set_ylabel('Return')
    axes[1, 1].legend()
    
    # 熵损失
    axes[1, 2].plot(training_stats['entropy_loss'], label='Entropy Loss', color='brown')
    axes[1, 2].set_title('Entropy Loss')
    axes[1, 2].set_xlabel('Update Step')
    axes[1, 2].set_ylabel('Loss')
    axes[1, 2].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_comparison_results(results: Dict[str, List[float]], save_path: str = None):
    """绘制算法比较结果"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 收敛曲线
    for algorithm, rewards in results.items():
        # 计算移动平均
        window_size = 50
        if len(rewards) >= window_size:
            moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
            axes[0].plot(moving_avg, label=algorithm)
        else:
            axes[0].plot(rewards, label=algorithm)
    
    axes[0].set_title('Convergence Comparison')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Average Reward')
    axes[0].legend()
    axes[0].grid(True)
    
    # 最终性能比较
    final_rewards = [np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards) 
                    for rewards in results.values()]
    algorithms = list(results.keys())
    
    bars = axes[1].bar(algorithms, final_rewards, color=['blue', 'orange', 'green', 'red'])
    axes[1].set_title('Final Performance Comparison')
    axes[1].set_ylabel('Average Reward (Last 100 Episodes)')
    axes[1].grid(True, axis='y')
    
    # 添加数值标签
    for bar, value in zip(bars, final_rewards):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_robustness_analysis(robustness_results: Dict, save_path: str = None):
    """绘制鲁棒性分析结果"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Robustness Analysis', fontsize=16)
    
    # 不同设备数的影响
    if 'device_count' in robustness_results:
        device_counts = list(robustness_results['device_count'].keys())
        rewards = list(robustness_results['device_count'].values())
        axes[0, 0].plot(device_counts, rewards, 'o-', linewidth=2, markersize=8)
        axes[0, 0].set_title('Impact of Device Count')
        axes[0, 0].set_xlabel('Number of Devices')
        axes[0, 0].set_ylabel('Average Reward')
        axes[0, 0].grid(True)
    
    # 不同社会关系权重的影响
    if 'social_weight' in robustness_results:
        weights = list(robustness_results['social_weight'].keys())
        rewards = list(robustness_results['social_weight'].values())
        axes[0, 1].plot(weights, rewards, 's-', linewidth=2, markersize=8)
        axes[0, 1].set_title('Impact of Social Weight (ρ)')
        axes[0, 1].set_xlabel('Social Weight')
        axes[0, 1].set_ylabel('Average Reward')
        axes[0, 1].grid(True)
    
    # 不同能量约束的影响
    if 'energy_constraint' in robustness_results:
        constraints = list(robustness_results['energy_constraint'].keys())
        rewards = list(robustness_results['energy_constraint'].values())
        axes[1, 0].plot(constraints, rewards, '^-', linewidth=2, markersize=8)
        axes[1, 0].set_title('Impact of Energy Constraint')
        axes[1, 0].set_xlabel('Energy Constraint (mJ)')
        axes[1, 0].set_ylabel('Average Reward')
        axes[1, 0].grid(True)
    
    # 不同功率约束的影响
    if 'power_constraint' in robustness_results:
        constraints = list(robustness_results['power_constraint'].keys())
        rewards = list(robustness_results['power_constraint'].values())
        axes[1, 1].plot(constraints, rewards, 'd-', linewidth=2, markersize=8)
        axes[1, 1].set_title('Impact of Power Constraint')
        axes[1, 1].set_xlabel('Power Constraint (dBm)')
        axes[1, 1].set_ylabel('Average Reward')
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def create_experiment_log(experiment_name: str, config: Dict, results: Dict) -> str:
    """创建实验日志"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"experiments/{experiment_name}_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    
    # 保存配置
    config_path = os.path.join(log_dir, "config.txt")
    with open(config_path, 'w') as f:
        f.write("Experiment Configuration:\n")
        f.write("=" * 50 + "\n")
        for key, value in config.items():
            f.write(f"{key}: {value}\n")
    
    # 保存结果
    results_path = os.path.join(log_dir, "results.txt")
    with open(results_path, 'w') as f:
        f.write("Experiment Results:\n")
        f.write("=" * 50 + "\n")
        for key, value in results.items():
            f.write(f"{key}: {value}\n")
    
    return log_dir

def save_training_data(training_data: Dict, save_path: str):
    """保存训练数据"""
    np.savez(save_path, **training_data)

def load_training_data(load_path: str) -> Dict:
    """加载训练数据"""
    return np.load(load_path, allow_pickle=True)

def calculate_constraint_violation_rate(constraint_violations: List[bool]) -> float:
    """计算约束违反率"""
    if not constraint_violations:
        return 0.0
    return sum(constraint_violations) / len(constraint_violations)

def calculate_energy_efficiency(charging_energies: List[float], 
                              energy_consumptions: List[float]) -> float:
    """计算能量效率"""
    if not charging_energies or not energy_consumptions:
        return 0.0
    
    total_charging = sum(charging_energies)
    total_consumption = sum(energy_consumptions)
    
    if total_consumption == 0:
        return 0.0
    
    return (total_charging - total_consumption) / total_consumption

def print_training_progress(episode: int, total_episodes: int, 
                          episode_reward: float, avg_reward: float,
                          energy_consumption: float, charging_energy: float):
    """打印训练进度"""
    progress = (episode + 1) / total_episodes * 100
    print(f"Episode {episode+1}/{total_episodes} ({progress:.1f}%) | "
          f"Reward: {episode_reward:.3f} | Avg Reward: {avg_reward:.3f} | "
          f"Energy: {energy_consumption:.3f} | Charging: {charging_energy:.3f}")

def set_random_seed(seed: int):
    """设置随机种子"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def calculate_model_parameters(model):
    """计算模型参数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def print_model_info(model, model_name="Model"):
    """打印模型信息"""
    total_params, trainable_params = calculate_model_parameters(model)
    print(f"{model_name} 信息:")
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    print(f"  模型大小: {total_params * 4 / 1024 / 1024:.2f} MB (float32)")
    
    # 估算GPU内存使用
    if torch.cuda.is_available():
        # 模型参数 + 梯度 + 优化器状态 + 激活值
        estimated_memory = total_params * 4 * 3 / 1024 / 1024  # 粗略估算
        print(f"  估算GPU内存: {estimated_memory:.2f} MB") 