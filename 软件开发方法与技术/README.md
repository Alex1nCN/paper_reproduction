# DRL-Based Joint Optimization of Wireless Charging and Computation Offloading

## 项目概述
本项目复现了论文《DRL-Based Joint Optimization of Wireless Charging and Computation Offloading for Multi-Access Edge Computing》中提出的混合Actor-Critic强化学习算法，用于联合优化无线充电和计算卸载决策。

## 主要特性
- **混合Actor-Critic架构**: 包含离散动作策略网络和连续动作策略网络
- **PPO-Clip策略更新**: 使用PPO算法进行稳定的策略更新
- **GAE优势估计**: 采用广义优势估计提高训练稳定性
- **多约束优化**: 满足能量、功率、时延等多种约束条件

## 环境建模
- **状态空间**: 设备剩余能量、无线信道增益(5类信道)
- **动作空间**: 任务卸载决策(离散) + 任务分配比例(连续) + 功率分配(连续)
- **奖励函数**: 社会关系强度与能量效率的加权组合

## 安装依赖
```bash
pip install -r requirements.txt
```

## 运行训练
```bash
python main.py
```

## 项目结构
- `environment.py`: MEC环境建模
- `networks.py`: 神经网络架构
- `agent.py`: 强化学习智能体
- `utils.py`: 工具函数
- `main.py`: 主训练脚本
- `evaluation.py`: 性能评估脚本 