import torch
import numpy as np
from environment import MECEnvironment
from agent import PPOHybridAgent
from utils import preprocess_state, postprocess_action, set_random_seed

def debug_environment():
    """调试环境"""
    print("=== 调试环境 ===")
    
    # 创建简单环境
    env = MECEnvironment(num_users=3, num_helpers=2)
    print(f"环境创建成功")
    print(f"状态空间: {env.get_state_space()}")
    print(f"动作空间: {env.get_action_space()}")
    
    # 测试reset
    print("测试reset...")
    state = env.reset()
    print(f"初始状态形状: {state.shape}")
    print(f"初始状态范围: [{state.min():.3f}, {state.max():.3f}]")
    
    # 测试step
    print("测试step...")
    action = {
        'offload_decision': 1,
        'task_allocation': np.array([0.3, 0.4, 0.3]),
        'power_allocation': np.array([10.0, 10.0, 50.0])
    }
    
    next_state, reward, done, info = env.step(action)
    print(f"Step结果: reward={reward:.3f}, done={done}")
    print(f"信息: {info}")
    
    print("环境测试通过！\n")

def debug_agent():
    """调试智能体"""
    print("=== 调试智能体 ===")
    
    env = MECEnvironment(num_users=3, num_helpers=2)
    agent = PPOHybridAgent(
        state_dim=env.get_state_space(),
        discrete_action_dim=env.get_action_space()['discrete'],
        continuous_action_dim=env.get_action_space()['continuous'],
        device='cpu'
    )
    
    print("智能体创建成功")
    
    # 测试动作选择
    state = env.reset()
    state_tensor = preprocess_state(state)
    action = agent.select_action(state_tensor, deterministic=True)
    print(f"动作: {action}")
    
    # 测试对数概率
    log_probs = agent.policy.get_log_probs(state_tensor, action)
    print(f"对数概率: {log_probs}")
    
    print("智能体测试通过！\n")

def debug_training_step():
    """调试训练步骤"""
    print("=== 调试训练步骤 ===")
    
    env = MECEnvironment(num_users=3, num_helpers=2)
    agent = PPOHybridAgent(
        state_dim=env.get_state_space(),
        discrete_action_dim=env.get_action_space()['discrete'],
        continuous_action_dim=env.get_action_space()['continuous'],
        device='cpu'
    )
    
    print("开始收集经验...")
    
    # 收集几个经验
    for i in range(5):
        print(f"收集经验 {i+1}/5")
        state = env.reset()
        step_count = 0
        
        while True:
            state_tensor = preprocess_state(state).to(agent.device)
            action = agent.select_action(state_tensor, deterministic=False)
            log_probs = agent.policy.get_log_probs(state_tensor, action)
            
            next_state, reward, done, info = env.step(postprocess_action(action))
            
            agent.store_transition(state_tensor, action, reward, 
                                 preprocess_state(next_state).to(agent.device), 
                                 done, log_probs)
            
            step_count += 1
            state = next_state
            
            if done:
                print(f"  Episode结束，步数: {step_count}")
                break
        
        print(f"  缓冲区大小: {len(agent.replay_buffer)}")
    
    # 测试训练
    print("测试训练...")
    if len(agent.replay_buffer) >= 10:
        agent.train(batch_size=10, num_epochs=2)
        print("训练完成！")
    else:
        print(f"经验不足，当前只有{len(agent.replay_buffer)}个经验")
    
    print("训练步骤测试通过！\n")

def main():
    """主调试函数"""
    print("开始调试...")
    set_random_seed(42)
    
    try:
        debug_environment()
        debug_agent()
        debug_training_step()
        
        print("所有调试测试通过！")
        
    except Exception as e:
        print(f"调试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 