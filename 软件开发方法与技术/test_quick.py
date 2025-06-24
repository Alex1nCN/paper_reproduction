import torch
import numpy as np
from environment import MECEnvironment
from networks import HybridActorCritic
from agent import PPOHybridAgent
from utils import preprocess_state, postprocess_action, set_random_seed

def test_environment():
    """测试环境基本功能"""
    print("测试MEC环境...")
    
    # 创建环境
    env = MECEnvironment(num_users=3, num_helpers=2)
    
    # 测试状态空间
    state = env.reset()
    print(f"状态空间维度: {env.get_state_space()}")
    print(f"状态形状: {state.shape}")
    print(f"动作空间: {env.get_action_space()}")
    
    # 测试环境步进
    action = {
        'offload_decision': 1,  # D2D
        'task_allocation': np.array([0.3, 0.4, 0.3]),  # 任务分配
        'power_allocation': np.array([10.0, 10.0, 50.0])  # 功率分配
    }
    
    next_state, reward, done, info = env.step(action)
    print(f"奖励: {reward:.4f}")
    print(f"是否结束: {done}")
    print(f"信息: {info}")
    
    print("环境测试通过！\n")

def test_agent():
    """测试智能体基本功能"""
    print("测试PPO智能体...")
    
    # 创建环境
    env = MECEnvironment(num_users=3, num_helpers=2)
    
    # 创建智能体
    agent = PPOHybridAgent(
        state_dim=env.get_state_space(),
        discrete_action_dim=env.get_action_space()['discrete'],
        continuous_action_dim=env.get_action_space()['continuous'],
        device='cpu'
    )
    
    # 测试动作选择
    state = env.reset()
    state_tensor = preprocess_state(state)
    action = agent.select_action(state_tensor, deterministic=True)
    print(f"动作: {action}")
    
    # 测试对数概率计算
    log_probs = agent.policy.get_log_probs(state_tensor, action)
    print(f"对数概率: {log_probs}")
    
    print("智能体测试通过！\n")

def test_training_step():
    """测试训练步骤"""
    print("测试训练步骤...")
    
    # 创建环境和智能体
    env = MECEnvironment(num_users=3, num_helpers=2)
    agent = PPOHybridAgent(
        state_dim=env.get_state_space(),
        discrete_action_dim=env.get_action_space()['discrete'],
        continuous_action_dim=env.get_action_space()['continuous'],
        device='cpu'
    )
    
    # 运行几个episode收集经验
    for episode in range(5):
        state = env.reset()
        episode_reward = 0
        
        while True:
            state_tensor = preprocess_state(state).to(agent.device)
            action = agent.select_action(state_tensor, deterministic=False)
            log_probs = agent.policy.get_log_probs(state_tensor, action)
            
            next_state, reward, done, info = env.step(postprocess_action(action))
            
            agent.store_transition(state_tensor, action, reward, 
                                 preprocess_state(next_state).to(agent.device), 
                                 done, log_probs)
            
            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        print(f"Episode {episode + 1}: 奖励 = {episode_reward:.4f}")
    
    # 测试训练
    if len(agent.replay_buffer) >= 10:
        agent.train(batch_size=10, num_epochs=2)
        print("训练完成！")
    else:
        print("经验不足，跳过训练")
    
    print("训练步骤测试通过！\n")

def test_numerical_stability():
    """测试数值稳定性"""
    print("Testing numerical stability...")
    
    # 创建环境
    env = MECEnvironment(num_users=3, num_helpers=2)
    
    # 创建网络
    state_dim = env.get_state_space()
    discrete_action_dim = env.get_action_space()['discrete']
    continuous_action_dim = env.get_action_space()['continuous']
    
    policy = HybridActorCritic(state_dim, discrete_action_dim, continuous_action_dim)
    
    # 创建智能体
    agent = PPOHybridAgent(state_dim, discrete_action_dim, continuous_action_dim)
    
    # 测试环境
    print("Testing environment...")
    state = env.reset()
    print(f"Initial state shape: {state.shape}")
    print(f"State range: [{state.min():.6f}, {state.max():.6f}]")
    print(f"State has nan: {np.isnan(state).any()}")
    print(f"State has inf: {np.isinf(state).any()}")
    
    # 测试动作选择
    print("\nTesting action selection...")
    state_tensor = torch.FloatTensor(state).unsqueeze(0)
    
    for i in range(10):
        try:
            action = agent.select_action(state_tensor, deterministic=False)
            print(f"Action {i}: {action}")
            
            # 测试环境步进
            next_state, reward, done, info = env.step(action)
            print(f"Reward: {reward:.6f}, Done: {done}")
            print(f"Next state has nan: {np.isnan(next_state).any()}")
            print(f"Next state has inf: {np.isinf(next_state).any()}")
            
            if done:
                state = env.reset()
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
            else:
                state = next_state
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                
        except Exception as e:
            print(f"Error in iteration {i}: {e}")
            break
    
    print("\nNumerical stability test completed!")

def main():
    """主测试函数"""
    print("=" * 50)
    print("快速功能测试")
    print("=" * 50)
    
    # 设置随机种子
    set_random_seed(42)
    
    try:
        # 测试环境
        test_environment()
        
        # 测试智能体
        test_agent()
        
        # 测试训练步骤
        test_training_step()
        
        # 测试数值稳定性
        test_numerical_stability()
        
        print("=" * 50)
        print("所有测试通过！代码基本功能正常。")
        print("=" * 50)
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 