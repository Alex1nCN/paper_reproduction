# 📡 无线充电与计算卸载联合优化的深度强化学习方法

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.13.1%2B-orange)
![CUDA](https://img.shields.io/badge/CUDA-11.7-green)
![License](https://img.shields.io/badge/License-MIT-blue)

## 📍 项目简介
本项目复现了论文《DRL-Based Joint Optimization of Wireless Charging and Computation Offloading for Multi-Access Edge Computing》提出的混合Actor-Critic算法，用于解决边缘计算环境中的以下关键问题：

1. **无线充电决策**：动态优化设备充电时机与功率
2. **计算任务卸载**：智能分配本地/边缘/云计算资源
3. **资源约束优化**：满足能量、功率和时延约束条件

## ⚡ 硬件说明
- **GPU**: NVIDIA GeForce RTX 4060 Ti (已启用CUDA加速)
- **CPU**: AMD Ryzen 5 7500F (6核12线程)
- **推荐内存**: 16GB DDR5 或更高

## 项目结构
drl-mec-optimization/
├── 📂 saved_models/         # 训练好的模型
├── 📜 environment.py        # MEC环境建模
│   ├── class EdgeEnv        # 边缘计算环境
│   ├── step()               # 状态转换
│   └── reward_function()    # 多目标奖励
├── 📜 networks.py           # 神经网络
│   ├── HybridActor()        # 混合动作策略网络
│   └── Critic()             # 价值评估网络
├── 📜 agent.py              # PPO智能体
│   ├── collect_rollouts()   # 数据收集
│   └── ppo_update()         # 策略优化核心
├── 📜 utils.py              # 支持工具
│   ├── ReplayBuffer()       # 经验回放池
│   └── perf_monitor()       # 资源监控
├── 📜 main.py               # 训练主循环
├── 📜 evaluation.py         # 性能评估
└── 📜 requirements.txt      # 依赖库

MIT License
Copyright (c) 2023 河海大学计算机与软件学院

允许商用/修改/私有化 需保留版权声明 禁止恶意使用
