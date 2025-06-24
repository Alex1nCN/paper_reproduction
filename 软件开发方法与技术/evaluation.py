import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from typing import Dict, List, Tuple

from environment import MECEnvironment
from agent import PPOHybridAgent
from utils import (
    preprocess_state, postprocess_action, calculate_metrics,
    plot_training_curves, plot_comparison_results, plot_robustness_analysis,
    save_training_data, set_random_seed
)

def evaluate_trained_model(model_path: str, env_config: Dict, 
                          num_episodes: int = 100) -> Dict:
    """评估训练好的模型"""
    print(f"评估模型: {model_path}")
    
    # 创建环境
    env = MECEnvironment(**env_config)
    
    # 创建智能体并加载模型
    agent = PPOHybridAgent(
        state_dim=env.get_state_space(),
        discrete_action_dim=env.get_action_space()['discrete'],
        continuous_action_dim=env.get_action_space()['continuous'],
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    agent.load_model(model_path)
    
    # 评估数据收集
    episode_rewards = []
    episode_energy_consumptions = []
    episode_charging_energies = []
    episode_social_strengths = []
    episode_constraint_violations = []
    episode_actions = []
    
    # 评估循环
    for episode in tqdm(range(num_episodes), desc="Evaluating"):
        state = env.reset()
        episode_reward = 0
        episode_energy_consumption = 0
        episode_charging_energy = 0
        episode_social_strength = 0
        episode_constraint_violation = 0
        episode_action_count = {'local': 0, 'd2d': 0, 'helper_d2d': 0, 'mec': 0}
        step_count = 0
        
        while True:
            # 预处理状态
            state_tensor = preprocess_state(state).to(agent.device)
            
            # 选择动作 (确定性)
            action = agent.select_action(state_tensor, deterministic=True)
            
            # 执行动作
            next_state, reward, done, info = env.step(postprocess_action(action))
            
            # 更新统计
            episode_reward += reward
            episode_energy_consumption += info['energy_consumption']
            episode_charging_energy += info['charging_energy']
            episode_social_strength += info['social_strength']
            episode_constraint_violation += int(info['constraints_violated'])
            
            # 统计动作分布
            offload_decision = action['offload_decision'].item()
            if offload_decision == 0:
                episode_action_count['local'] += 1
            elif offload_decision == 1:
                episode_action_count['d2d'] += 1
            elif offload_decision == 2:
                episode_action_count['helper_d2d'] += 1
            else:
                episode_action_count['mec'] += 1
            
            step_count += 1
            state = next_state
            
            if done:
                break
        
        # 记录episode统计
        episode_rewards.append(episode_reward)
        episode_energy_consumptions.append(episode_energy_consumption)
        episode_charging_energies.append(episode_charging_energy)
        episode_social_strengths.append(episode_social_strength / step_count)
        episode_constraint_violations.append(episode_constraint_violation / step_count)
        episode_actions.append(episode_action_count)
    
    # 计算评估指标
    metrics = calculate_metrics(episode_rewards, episode_energy_consumptions,
                              episode_charging_energies, episode_social_strengths)
    metrics['constraint_violation_rate'] = np.mean(episode_constraint_violations)
    
    # 计算动作分布
    action_distribution = {
        'local': np.mean([ep['local'] for ep in episode_actions]),
        'd2d': np.mean([ep['d2d'] for ep in episode_actions]),
        'helper_d2d': np.mean([ep['helper_d2d'] for ep in episode_actions]),
        'mec': np.mean([ep['mec'] for ep in episode_actions])
    }
    
    return {
        'episode_rewards': episode_rewards,
        'episode_energy_consumptions': episode_energy_consumptions,
        'episode_charging_energies': episode_charging_energies,
        'episode_social_strengths': episode_social_strengths,
        'episode_constraint_violations': episode_constraint_violations,
        'episode_actions': episode_actions,
        'metrics': metrics,
        'action_distribution': action_distribution
    }

def analyze_action_patterns(evaluation_results: Dict) -> Dict:
    """分析动作模式"""
    action_distribution = evaluation_results['action_distribution']
    episode_actions = evaluation_results['episode_actions']
    
    # 计算动作选择的稳定性
    action_std = {
        'local': np.std([ep['local'] for ep in episode_actions]),
        'd2d': np.std([ep['d2d'] for ep in episode_actions]),
        'helper_d2d': np.std([ep['helper_d2d'] for ep in episode_actions]),
        'mec': np.std([ep['mec'] for ep in episode_actions])
    }
    
    # 计算主要策略
    total_actions = sum(action_distribution.values())
    if total_actions > 0:
        primary_strategy = max(action_distribution, key=action_distribution.get)
        primary_ratio = action_distribution[primary_strategy] / total_actions
    else:
        primary_strategy = 'none'
        primary_ratio = 0.0
    
    return {
        'action_distribution': action_distribution,
        'action_std': action_std,
        'primary_strategy': primary_strategy,
        'primary_ratio': primary_ratio
    }

def plot_evaluation_results(evaluation_results: Dict, save_path: str = None):
    """绘制评估结果"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Model Evaluation Results', fontsize=16)
    
    # 奖励分布
    rewards = evaluation_results['episode_rewards']
    axes[0, 0].hist(rewards, bins=20, alpha=0.7, color='blue')
    axes[0, 0].axvline(np.mean(rewards), color='red', linestyle='--', label=f'Mean: {np.mean(rewards):.3f}')
    axes[0, 0].set_title('Reward Distribution')
    axes[0, 0].set_xlabel('Reward')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].legend()
    
    # 能量效率
    energy_efficiencies = [ec - cc for ec, cc in zip(
        evaluation_results['episode_charging_energies'],
        evaluation_results['episode_energy_consumptions']
    )]
    axes[0, 1].hist(energy_efficiencies, bins=20, alpha=0.7, color='green')
    axes[0, 1].axvline(np.mean(energy_efficiencies), color='red', linestyle='--', 
                      label=f'Mean: {np.mean(energy_efficiencies):.3f}')
    axes[0, 1].set_title('Energy Efficiency Distribution')
    axes[0, 1].set_xlabel('Energy Efficiency')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()
    
    # 社会关系强度
    social_strengths = evaluation_results['episode_social_strengths']
    axes[0, 2].hist(social_strengths, bins=20, alpha=0.7, color='orange')
    axes[0, 2].axvline(np.mean(social_strengths), color='red', linestyle='--',
                      label=f'Mean: {np.mean(social_strengths):.3f}')
    axes[0, 2].set_title('Social Strength Distribution')
    axes[0, 2].set_xlabel('Social Strength')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].legend()
    
    # 动作分布
    action_dist = evaluation_results['action_distribution']
    actions = list(action_dist.keys())
    values = list(action_dist.values())
    bars = axes[1, 0].bar(actions, values, color=['blue', 'orange', 'green', 'red'])
    axes[1, 0].set_title('Action Distribution')
    axes[1, 0].set_ylabel('Average Count per Episode')
    
    # 添加数值标签
    for bar, value in zip(bars, values):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                       f'{value:.1f}', ha='center', va='bottom')
    
    # 约束违反率
    constraint_violations = evaluation_results['episode_constraint_violations']
    axes[1, 1].hist(constraint_violations, bins=20, alpha=0.7, color='red')
    axes[1, 1].axvline(np.mean(constraint_violations), color='blue', linestyle='--',
                      label=f'Mean: {np.mean(constraint_violations):.3f}')
    axes[1, 1].set_title('Constraint Violation Rate')
    axes[1, 1].set_xlabel('Violation Rate')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].legend()
    
    # 奖励时间序列
    axes[1, 2].plot(rewards, alpha=0.7, color='purple')
    axes[1, 2].axhline(np.mean(rewards), color='red', linestyle='--',
                      label=f'Mean: {np.mean(rewards):.3f}')
    axes[1, 2].set_title('Reward Time Series')
    axes[1, 2].set_xlabel('Episode')
    axes[1, 2].set_ylabel('Reward')
    axes[1, 2].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def generate_evaluation_report(evaluation_results: Dict, model_path: str) -> str:
    """生成评估报告"""
    report = []
    report.append("=" * 60)
    report.append("MODEL EVALUATION REPORT")
    report.append("=" * 60)
    report.append(f"Model Path: {model_path}")
    report.append(f"Evaluation Episodes: {len(evaluation_results['episode_rewards'])}")
    report.append("")
    
    # 性能指标
    metrics = evaluation_results['metrics']
    report.append("PERFORMANCE METRICS:")
    report.append("-" * 30)
    for key, value in metrics.items():
        report.append(f"{key}: {value:.4f}")
    report.append("")
    
    # 动作分析
    action_analysis = analyze_action_patterns(evaluation_results)
    report.append("ACTION PATTERN ANALYSIS:")
    report.append("-" * 30)
    report.append("Action Distribution:")
    for action, count in action_analysis['action_distribution'].items():
        report.append(f"  {action}: {count:.2f}")
    report.append(f"Primary Strategy: {action_analysis['primary_strategy']}")
    report.append(f"Primary Strategy Ratio: {action_analysis['primary_ratio']:.3f}")
    report.append("")
    
    # 稳定性分析
    report.append("STABILITY ANALYSIS:")
    report.append("-" * 30)
    for action, std in action_analysis['action_std'].items():
        report.append(f"{action} std: {std:.2f}")
    report.append("")
    
    # 约束分析
    constraint_rate = metrics['constraint_violation_rate']
    report.append("CONSTRAINT ANALYSIS:")
    report.append("-" * 30)
    report.append(f"Constraint Violation Rate: {constraint_rate:.4f}")
    if constraint_rate < 0.05:
        report.append("Status: GOOD (Low violation rate)")
    elif constraint_rate < 0.1:
        report.append("Status: ACCEPTABLE (Moderate violation rate)")
    else:
        report.append("Status: POOR (High violation rate)")
    
    return "\n".join(report)

def run_comprehensive_evaluation(model_path: str, env_config: Dict, 
                               output_dir: str = "evaluation_results"):
    """运行综合评估"""
    print("开始综合评估...")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置随机种子
    set_random_seed(42)
    
    # 运行评估
    evaluation_results = evaluate_trained_model(model_path, env_config, num_episodes=100)
    
    # 保存评估结果
    save_training_data(evaluation_results, os.path.join(output_dir, "evaluation_results.npz"))
    
    # 绘制评估结果
    plot_evaluation_results(evaluation_results, os.path.join(output_dir, "evaluation_plots.png"))
    
    # 生成评估报告
    report = generate_evaluation_report(evaluation_results, model_path)
    report_path = os.path.join(output_dir, "evaluation_report.txt")
    with open(report_path, 'w') as f:
        f.write(report)
    
    # 打印报告
    print(report)
    
    print(f"\n评估结果已保存到: {output_dir}")
    
    return evaluation_results

def main():
    """主函数"""
    # 环境配置
    env_config = {
        'num_users': 5,
        'num_helpers': 3,
        'radius': 100.0,
        'helper_radius': (10.0, 50.0),
        'time_slot': 1.0,
        'max_power_u': 18.0,
        'max_power_n': 18.0,
        'max_power_b': 70.0,
        'discount_factor': 0.64,
        'rho': 0.5
    }
    
    # 模型路径 (需要先训练模型)
    model_path = "experiments/mec_optimization_*/ppo_model.pth"
    
    # 查找最新的模型文件
    import glob
    model_files = glob.glob(model_path)
    if not model_files:
        print("未找到训练好的模型文件。请先运行 main.py 训练模型。")
        return
    
    latest_model = max(model_files, key=os.path.getctime)
    print(f"使用模型: {latest_model}")
    
    # 运行综合评估
    evaluation_results = run_comprehensive_evaluation(latest_model, env_config)

if __name__ == "__main__":
    main() 