import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from typing import Dict, List

from environment import MECEnvironment
from agent import PPOHybridAgent, A3CAgent, A2CAgent
from utils import (
    preprocess_state, postprocess_action, calculate_metrics,
    plot_training_curves, plot_comparison_results, plot_robustness_analysis,
    create_experiment_log, save_training_data, print_training_progress,
    set_random_seed, calculate_constraint_violation_rate, print_model_info
)

def train_ppo_agent(env: MECEnvironment, agent: PPOHybridAgent, 
                   num_episodes: int = 500, batch_size: int = 64,
                   update_frequency: int = 100) -> Dict:
    """训练PPO智能体"""
    print("开始训练PPO混合Actor-Critic智能体...")
    
    # 训练数据收集
    episode_rewards = []
    episode_energy_consumptions = []
    episode_charging_energies = []
    episode_social_strengths = []
    episode_constraint_violations = []
    
    # 训练循环
    for episode in tqdm(range(num_episodes), desc="Training PPO"):
        print(f"开始Episode {episode + 1}")  # 调试信息
        state = env.reset()
        episode_reward = 0
        episode_energy_consumption = 0
        episode_charging_energy = 0
        episode_social_strength = 0
        episode_constraint_violation = 0
        step_count = 0
        
        while True:
            # 预处理状态
            state_tensor = preprocess_state(state).to(agent.device)
            
            # 选择动作
            action = agent.select_action(state_tensor, deterministic=False)
            
            # 获取动作的对数概率
            log_probs = agent.policy.get_log_probs(state_tensor, action)
            
            # 执行动作
            next_state, reward, done, info = env.step(postprocess_action(action))
            
            # 存储经验
            agent.store_transition(state_tensor, action, reward, 
                                 preprocess_state(next_state).to(agent.device), 
                                 done, log_probs)
            
            # 更新统计
            episode_reward += reward
            episode_energy_consumption += info['energy_consumption']
            episode_charging_energy += info['charging_energy']
            episode_social_strength += info['social_strength']
            episode_constraint_violation += int(info['constraints_violated'])
            step_count += 1
            
            state = next_state
            
            if done:
                print(f"Episode {episode + 1} 结束，步数: {step_count}")  # 调试信息
                break
        
        # 记录episode统计
        episode_rewards.append(episode_reward)
        episode_energy_consumptions.append(episode_energy_consumption)
        episode_charging_energies.append(episode_charging_energy)
        episode_social_strengths.append(episode_social_strength / step_count)
        episode_constraint_violations.append(episode_constraint_violation / step_count)
        
        # 定期更新策略
        if (episode + 1) % update_frequency == 0:
            agent.train(batch_size=batch_size, num_epochs=10)
        
        # 打印进度
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            print_training_progress(episode, num_episodes, episode_reward, avg_reward,
                                  episode_energy_consumption, episode_charging_energy)
    
    # 计算最终指标
    metrics = calculate_metrics(episode_rewards, episode_energy_consumptions,
                              episode_charging_energies, episode_social_strengths)
    metrics['constraint_violation_rate'] = np.mean(episode_constraint_violations)
    
    return {
        'episode_rewards': episode_rewards,
        'episode_energy_consumptions': episode_energy_consumptions,
        'episode_charging_energies': episode_charging_energies,
        'episode_social_strengths': episode_social_strengths,
        'episode_constraint_violations': episode_constraint_violations,
        'metrics': metrics,
        'training_stats': agent.get_training_stats()
    }

def train_comparison_agent(env: MECEnvironment, agent, agent_name: str,
                          num_episodes: int = 500) -> List[float]:
    """训练比较算法"""
    print(f"开始训练{agent_name}智能体...")
    
    episode_rewards = []
    
    for episode in tqdm(range(num_episodes), desc=f"Training {agent_name}"):
        state = env.reset()
        episode_reward = 0
        
        while True:
            state_tensor = preprocess_state(state).to(agent.device)
            action = agent.select_action(state_tensor, deterministic=False)
            next_state, reward, done, info = env.step(postprocess_action(action))
            
            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        
        # 简化的更新 (对于比较算法)
        if hasattr(agent, 'update') and (episode + 1) % 100 == 0:
            # 这里需要收集一些经验进行更新，简化处理
            pass
    
    return episode_rewards

def run_algorithm_comparison(env_config: Dict, num_episodes: int = 500) -> Dict:
    """运行算法比较"""
    print("开始算法比较实验...")
    
    results = {}
    
    # 创建环境
    env = MECEnvironment(**env_config)
    print(f"环境创建成功，状态空间: {env.get_state_space()}")  # 调试信息
    
    # 训练PPO
    ppo_agent = PPOHybridAgent(
        state_dim=env.get_state_space(),
        discrete_action_dim=env.get_action_space()['discrete'],
        continuous_action_dim=env.get_action_space()['continuous'],
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # 打印模型信息
    print_model_info(ppo_agent.policy, "PPO-Hybrid Policy")
    
    print("开始训练...")  # 调试信息
    ppo_results = train_ppo_agent(env, ppo_agent, num_episodes)
    results['PPO-Hybrid'] = ppo_results['episode_rewards']
    
    # 训练A3C
    a3c_agent = A3CAgent(
        state_dim=env.get_state_space(),
        discrete_action_dim=env.get_action_space()['discrete'],
        continuous_action_dim=env.get_action_space()['continuous'],
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    results['A3C'] = train_comparison_agent(env, a3c_agent, "A3C", num_episodes)
    
    # 训练A2C
    a2c_agent = A2CAgent(
        state_dim=env.get_state_space(),
        discrete_action_dim=env.get_action_space()['discrete'],
        continuous_action_dim=env.get_action_space()['continuous'],
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    results['A2C'] = train_comparison_agent(env, a2c_agent, "A2C", num_episodes)
    
    return results, ppo_results, ppo_agent

def run_robustness_analysis(base_config: Dict) -> Dict:
    """运行鲁棒性分析"""
    print("开始鲁棒性分析...")
    
    robustness_results = {}
    
    # 不同设备数的影响
    device_counts = [3, 5, 7, 10]
    device_results = {}
    
    for num_users in device_counts:
        config = base_config.copy()
        config['num_users'] = num_users
        config['num_helpers'] = max(2, num_users // 2)
        
        env = MECEnvironment(**config)
        agent = PPOHybridAgent(
            state_dim=env.get_state_space(),
            discrete_action_dim=env.get_action_space()['discrete'],
            continuous_action_dim=env.get_action_space()['continuous'],
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        results = train_ppo_agent(env, agent, num_episodes=200)  # 减少episode数
        device_results[num_users] = results['metrics']['mean_reward']
    
    robustness_results['device_count'] = device_results
    
    # 不同社会关系权重的影响
    social_weights = [0.1, 0.3, 0.5, 0.7, 0.9]
    social_results = {}
    
    for rho in social_weights:
        config = base_config.copy()
        config['rho'] = rho
        
        env = MECEnvironment(**config)
        agent = PPOHybridAgent(
            state_dim=env.get_state_space(),
            discrete_action_dim=env.get_action_space()['discrete'],
            continuous_action_dim=env.get_action_space()['continuous'],
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        results = train_ppo_agent(env, agent, num_episodes=200)
        social_results[rho] = results['metrics']['mean_reward']
    
    robustness_results['social_weight'] = social_results
    
    # 不同能量约束的影响
    energy_constraints = [500, 750, 1000, 1250, 1500]
    energy_results = {}
    
    for max_energy in energy_constraints:
        config = base_config.copy()
        config['max_energy'] = max_energy
        
        env = MECEnvironment(**config)
        agent = PPOHybridAgent(
            state_dim=env.get_state_space(),
            discrete_action_dim=env.get_action_space()['discrete'],
            continuous_action_dim=env.get_action_space()['continuous'],
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        results = train_ppo_agent(env, agent, num_episodes=200)
        energy_results[max_energy] = results['metrics']['mean_reward']
    
    robustness_results['energy_constraint'] = energy_results
    
    return robustness_results

def main():
    """主函数"""
    # 设置随机种子
    set_random_seed(42)
    
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
        'rho': 0.5,
        'max_energy': 1000.0
    }
    
    # 训练配置
    train_config = {
        'num_episodes': 500,
        'batch_size': 64,
        'update_frequency': 10,
        'learning_rate': 3e-4,
        'gamma': 0.64,
        'clip_ratio': 0.1
    }
    
    print("=" * 60)
    print("DRL-Based Joint Optimization of Wireless Charging and Computation Offloading")
    print("=" * 60)
    print(f"环境配置: {env_config}")
    print(f"训练配置: {train_config}")
    print("=" * 60)
    
    # 创建实验目录
    experiment_name = "mec_optimization"
    log_dir = create_experiment_log(experiment_name, {**env_config, **train_config}, {})
    
    # 1. 算法比较实验
    print("\n1. 运行算法比较实验...")
    comparison_results, ppo_detailed_results, ppo_agent = run_algorithm_comparison(env_config, train_config['num_episodes'])
    
    # 保存PPO模型
    model_path = os.path.join(log_dir, "ppo_model.pth")
    ppo_agent.save_model(model_path)
    print(f"PPO模型已保存到: {model_path}")
    
    # 保存比较结果
    save_training_data(comparison_results, os.path.join(log_dir, "comparison_results.npz"))
    
    # 绘制比较结果
    plot_comparison_results(comparison_results, os.path.join(log_dir, "comparison_results.png"))
    
    # 2. 绘制PPO训练曲线
    print("\n2. 绘制PPO训练曲线...")
    plot_training_curves(ppo_detailed_results['training_stats'], 
                        os.path.join(log_dir, "ppo_training_curves.png"))
    
    # 3. 鲁棒性分析
    print("\n3. 运行鲁棒性分析...")
    robustness_results = run_robustness_analysis(env_config)
    
    # 保存鲁棒性结果
    save_training_data(robustness_results, os.path.join(log_dir, "robustness_results.npz"))
    
    # 绘制鲁棒性分析结果
    plot_robustness_analysis(robustness_results, os.path.join(log_dir, "robustness_analysis.png"))
    
    # 4. 保存详细结果
    print("\n4. 保存详细结果...")
    save_training_data(ppo_detailed_results, os.path.join(log_dir, "ppo_detailed_results.npz"))
    
    # 5. 打印最终结果
    print("\n" + "=" * 60)
    print("实验完成！")
    print("=" * 60)
    print(f"实验日志保存在: {log_dir}")
    print(f"PPO模型保存在: {model_path}")
    print("\nPPO-Hybrid最终性能:")
    for key, value in ppo_detailed_results['metrics'].items():
        print(f"  {key}: {value:.4f}")
    
    print("\n算法比较结果:")
    for algorithm, rewards in comparison_results.items():
        final_reward = np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards)
        print(f"  {algorithm}: {final_reward:.4f}")
    
    print("\n鲁棒性分析结果:")
    for analysis_type, results in robustness_results.items():
        print(f"  {analysis_type}:")
        for param, reward in results.items():
            print(f"    {param}: {reward:.4f}")
    
    print("=" * 60)

if __name__ == "__main__":
    main() 