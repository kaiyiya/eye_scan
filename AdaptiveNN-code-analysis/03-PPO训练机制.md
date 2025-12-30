# 03 - PPO 训练机制详解

本文件详细解析 AdaptiveNN 如何使用 PPO (Proximal Policy Optimization) 训练策略网络。

---

## 🎯 训练策略概述

AdaptiveNN 使用**混合训练策略**：

1. **监督学习**: 训练 Glance Net、Focus Net 和分类头
2. **强化学习 (PPO)**: 训练策略网络（Policy Network）

关键点：策略网络的训练是**无监督**的，不需要标注注视点位置！

---

## 📊 训练流程概览

```
每个 batch:
  ↓
1. 前向传播（所有网络）
  ↓
2. 计算监督学习损失（Glance/Focus/分类）
  ↓
3. 反向传播更新 Glance/Focus 网络（策略网络梯度被跳过）
  ↓
4. 每隔 update_policy_freq 个 batch:
  ↓
   a. 计算奖励 (Reward)
  ↓
   b. 计算优势 (Advantage) - GAE
  ↓
   c. PPO 更新策略网络（只更新策略网络参数）
```

---

## 🔍 阶段 1: 监督学习损失

### 1.1 Focus Net 正则化损失

```python
outputs_reg_focus_net = expected_outputs['outputs_reg_focus_net']
loss_reg_focus_net = criterion(outputs_reg_focus_net, targets)
loss_reg_focus_net = args.loss_reg_focus_net_weight * loss_reg_focus_net
```

**说明**:
- 对整张图像进行 Focus Net 处理（下采样）
- 提供额外的监督信号
- 权重通常设为 2.0

### 1.2 知识蒸馏损失 (Knowledge Distillation)

```python
out_teacher = expected_outputs['x_focus'][-1].detach()  # 最终分类结果作为教师

# Glance 结果向最终结果学习
loss_KD = F.kl_div(
    F.log_softmax(expected_outputs['x_glance'][-1] / args.kd_temp, dim=1),
    F.softmax(out_teacher / args.kd_temp, dim=1), 
    reduction='batchmean'
) * (args.kd_temp ** 2)

# 每个中间步骤也向最终结果学习
loss_KD += sum([
    F.kl_div(
        F.log_softmax(_x_focus / args.kd_temp, dim=1),
        F.softmax(out_teacher / args.kd_temp, dim=1), 
        reduction='batchmean'
    ) * (args.kd_temp ** 2)
    for _x_focus in expected_outputs['x_focus'][:-1]
])
```

**目的**: 
- 让中间步骤的分类结果向最终结果学习
- 使用温度缩放 (temperature scaling) 软化概率分布

### 1.3 Glance 和 Focus 分类损失

```python
# Glance 阶段的分类损失
loss_glance = criterion(expected_outputs['x_glance'][-1], targets)

# 所有 Focus 步骤的分类损失
loss_focus = sum([
    criterion(_x_focus, targets) 
    for _x_focus in expected_outputs['x_focus']
])
```

### 1.4 总监督学习损失

```python
loss = loss_reg_focus_net + loss_focus + loss_glance + loss_KD * args.kd_alpha
```

**反向传播**: 
- 更新 Glance Net、Focus Net 和分类头
- **跳过策略网络的梯度** (`skip_policy_net=True`)

---

## 🎮 阶段 2: PPO 更新（每隔 update_policy_freq 个 batch）

### 2.1 收集 PPO 数据

```python
# 每 update_policy_freq 个 batch 收集一次
if (data_iter_step + 1) % args.update_policy_freq == 0:
    
    # 合并多个 batch 的数据
    expected_outputs_ppo = {
        'x_glance': [], 
        'x_focus': [], 
        'actions': [],              # 动作（注视点位置）
        'actions_logprobs': [],     # 动作的对数概率
        '_state_values': [],        # 状态价值
        'states': [],               # 状态特征
    }
    
    # 将多个 batch 的数据拼接起来
    for key in ['x_glance', 'x_focus', 'actions', 'actions_logprobs', '_state_values', 'states']:
        for index in range(len(ppo_update_collect_dict[0][key])):
            expected_outputs_ppo[key].append(
                torch.cat([
                    ppo_update_collect_dict[m][key][index] 
                    for m in range(args.update_policy_freq)
                ], dim=0)
            )
```

**数据形状**:
- `actions`: `(B*update_policy_freq, seq_l, 2)` - 每个步骤的动作
- `actions_logprobs`: `(B*update_policy_freq, seq_l)` - 每个步骤的对数概率
- `_state_values`: `(B*update_policy_freq, seq_l, 1)` - 每个步骤的状态价值
- `states`: `seq_l` 个元素，每个 `(B*update_policy_freq, 49, 384)`

---

### 2.2 计算奖励 (Reward)

```python
_list_all_outputs = expected_outputs_ppo['x_glance'] + expected_outputs_ppo['x_focus']
# 包含: [x_glance, x_focus[0], x_focus[1], x_focus[2], x_focus[3]]

# 计算单步奖励：当前步骤损失 - 下一步骤损失
list_to_compute_reward = [
    (F.cross_entropy(_list_all_outputs[focus_step_index].detach(), targets, reduction='none')
     - F.cross_entropy(_list_all_outputs[focus_step_index + 1].detach(), targets, reduction='none')
    ).unsqueeze(-1)
    for focus_step_index in range(args.seq_l)
]

mb_rewards = torch.cat(list_to_compute_reward, dim=1)  # (B, seq_l)
```

**奖励定义**:
- **奖励 = 损失减少量**
- 如果执行动作后分类损失减小 → 正奖励 ✅
- 如果执行动作后分类损失增大 → 负奖励 ❌

**直观理解**:
- 策略网络的目标是选择能够**改善分类结果**的注视点
- 通过分类损失的减少作为奖励信号，无需人工标注

---

### 2.3 计算优势 (Advantage) - GAE

GAE (Generalized Advantage Estimation) 用于估计动作的优势：

```python
old_states = expected_outputs_ppo['states']
old_actions = expected_outputs_ppo['actions']
old_actions_logprobs = torch.cat(expected_outputs_ppo['actions_logprobs'], dim=1).detach()  # (B, seq_l)
old_state_values = torch.cat(expected_outputs_ppo['_state_values'], dim=1).detach()  # (B, seq_l)

# GAE 计算
mb_values = old_state_values  # (B, seq_l)
mb_advs = torch.zeros_like(mb_rewards)  # (B, seq_l)
lastgaelam = torch.zeros((mb_advs.shape[0], ), device=model.device)

# 从后往前计算
for t in reversed(range(args.seq_l)):
    if t == args.seq_l - 1:  # 最后一步
        nextnonterminal = 0.0  # 没有下一步
        nextvalues = 0.0
    else:
        nextnonterminal = 1.0
        nextvalues = mb_values[:, t+1]  # 下一步的状态价值
    
    # TD 误差
    delta = mb_rewards[:, t] + args.gamma * nextvalues * nextnonterminal - mb_values[:, t]
    
    # GAE
    mb_advs[:, t] = lastgaelam = delta + args.gamma * args.ppo_lam * nextnonterminal * lastgaelam

mb_returns = mb_advs + mb_values  # 回报 = 优势 + 价值

# 优势归一化（可选）
if args.adv_normalization:
    mb_advs = (mb_advs - mb_advs.mean()) / (mb_advs.std() + 1e-8)

advantages = mb_advs
```

**GAE 公式**:
```
δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
A_t = δ_t + (γλ) * δ_{t+1} + (γλ)² * δ_{t+2} + ...
```

**参数**:
- `gamma` (默认 0.7): 折扣因子
- `ppo_lam` (默认 0.84): GAE 参数 λ

---

### 2.4 PPO 更新策略网络

```python
# 只更新策略网络参数
for name, param in model.named_parameters():
    if 'policy_net_patch' in name:
        param.requires_grad_(True)
    else:
        param.requires_grad_(False)

# PPO 更新
num_ppo_update_iters = args.num_ppo_update_iters  # 通常为 1
ppo_total_batch_size = args.batch_size * args.update_policy_freq
mini_batch_size = int(ppo_total_batch_size / num_ppo_update_iters)

for _ in range(args.ppo_update_steps):  # 通常为 2-5 次
    # 随机打乱数据
    inds = torch.randperm(ppo_total_batch_size)
    
    for i in range(num_ppo_update_iters):
        batch_index = inds[torch.arange(i*mini_batch_size, (i+1)*mini_batch_size)]
        
        # 重新评估旧动作
        logprobs, state_values, dist_entropy = model(
            old_states=old_states, 
            old_actions=old_actions, 
            seq_l=args.seq_l,
            flag='evaluate_policy_net', 
            batch_index=batch_index, 
            ppo_std_this_iter=ppo_std_this_iter
        )
        
        # 计算重要性采样比率
        ratios = torch.exp(logprobs - old_actions_logprobs[batch_index].detach())
        
        # PPO 裁剪目标
        surr1 = ratios * advantages[batch_index]
        surr2 = torch.clamp(ratios, 1 - args.eps_clip, 1 + args.eps_clip) * advantages[batch_index]
        
        # PPO 损失
        loss = (
            -torch.min(surr1, surr2).mean()           # 策略损失（裁剪）
            + mse_loss(state_values, mb_returns[batch_index].detach())  # 价值损失
            - 0.01 * dist_entropy.mean()              # 熵正则化
        )
        
        # 反向传播（只更新策略网络）
        optimizer.zero_grad()
        loss_scaler(
            loss, optimizer, 
            model=model, 
            skip_backbones=True,      # 跳过 backbone
            skip_policy_net=False,    # 更新策略网络
            clip_grad=args.ppo_clip_grid,
            parameters=model.parameters()
        )

# 恢复参数设置
for name, param in model.named_parameters():
    if 'policy_net_patch' in name:
        param.requires_grad_(False)
    else:
        param.requires_grad_(True)
```

**PPO 损失组成**:

1. **策略损失** (裁剪目标):
   ```
   L^CLIP = min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t)
   ```
   - `r_t`: 新旧策略比率
   - `eps_clip` (默认 0.2): 裁剪范围

2. **价值损失**:
   ```
   L^VF = (V(s_t) - R_t)²
   ```
   - 价值函数学习估计回报

3. **熵正则化**:
   ```
   L^ENT = -H(π)
   ```
   - 鼓励探索，防止策略过早收敛

---

## 🔄 完整训练循环

```python
for epoch in range(epochs):
    for batch in data_loader:
        # === 阶段 1: 监督学习 ===
        outputs = model(batch)
        
        # 计算监督损失
        loss_supervised = compute_supervised_loss(outputs, targets)
        
        # 反向传播（跳过策略网络）
        loss_supervised.backward()
        optimizer.step()  # 只更新 Glance/Focus 网络
        
        # 收集 PPO 数据
        ppo_data.append(outputs)
        
        # === 阶段 2: PPO 更新（每 update_policy_freq 个 batch） ===
        if batch_idx % update_policy_freq == 0:
            # 计算奖励和优势
            rewards = compute_rewards(ppo_data, targets)
            advantages = compute_gae(rewards, values)
            
            # PPO 更新
            for ppo_step in range(ppo_update_steps):
                loss_ppo = compute_ppo_loss(ppo_data, advantages)
                loss_ppo.backward()  # 只更新策略网络
                optimizer.step()
            
            ppo_data.clear()
```

---

## ⚙️ 关键超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `update_policy_freq` | 10 | 每隔多少个 batch 更新一次策略网络 |
| `ppo_update_steps` | 2-5 | 每次 PPO 更新的步数 |
| `gamma` | 0.7 | 折扣因子（奖励衰减） |
| `ppo_lam` | 0.84 | GAE 参数 λ |
| `eps_clip` | 0.2 | PPO 裁剪范围 |
| `ppo_std_start` | 0.25 | 动作噪声初始标准差 |
| `ppo_std_end` | 0.10 | 动作噪声最终标准差 |

---

## 🎓 设计亮点

1. **自奖励机制**:
   - 不需要标注注视点位置
   - 通过分类损失减少作为奖励

2. **混合训练**:
   - 监督学习训练特征提取网络
   - 强化学习训练决策网络

3. **交替更新**:
   - 策略网络和特征网络交替更新
   - 避免训练不稳定

4. **渐进式探索**:
   - 训练初期噪声大（0.25），后期小（0.1）
   - 从探索到利用的平滑过渡

---

## 📝 下一步

阅读 `04-关键组件解析.md` 了解各个辅助函数的实现细节。


