# 重要发现：AdaptiveNN确实生成注视点序列！

## 🎯 核心发现

**AdaptiveNN在中间过程中确实生成了一序列的注视点位置（眼动序列/扫描路径序列）！**

虽然AdaptiveNN的最终目标是图像分类，但在处理过程中，它通过Policy Network生成了一序列的注视点坐标，这就是**眼动序列**！

---

## 🔍 代码证据

### 关键代码片段

从`AdaptiveNN-main/models/dynamic_deitS.py`的`forward_backbone`方法中：

```python
def forward_backbone(self, imgs, seq_l=4, ppo_std_this_iter=None):
    # ...
    expected_outputs = {
        'actions': [],  # 存储注视点序列！
        'actions_logprobs': [],
        'states': [],
        # ...
    }
    
    # Glance阶段（全局快速浏览）
    # ...
    
    # Focus循环（生成注视点序列）
    updated_features = global_features
    for focus_step_index in range(seq_l):  # 默认循环4次
        # ...
        
        # 步骤1：Policy Network输出注视点位置
        _actions = self.policy_net_patch(...)
        # 输出: (B, 2) - [x, y] 坐标，范围 [0, 1]
        
        # 步骤2：采样动作（训练时添加噪声）
        if self.training:
            # 添加探索噪声
            dist = MultivariateNormal(actions, scale_tril=cov_mat)
            actions = dist.sample()  # (B, 2)
            actions_logprobs = dist.log_prob(actions)
        else:
            actions = _actions  # 推理时直接使用
        
        # 步骤3：保存注视点位置到序列中！
        actions = actions.detach()
        expected_outputs['actions'].append(actions.detach())  # ⭐ 关键！
        
        # 步骤4：根据注视点位置提取patch
        x_patch = self.get_img_patches(imgs, actions, ...)
        
        # 步骤5：Focus处理
        # ...
        
    # 返回结果
    return expected_outputs
```

### 序列结构

**`expected_outputs['actions']`的结构：**

```python
expected_outputs['actions'] = [
    actions_step0,  # (B, 2) - 第1个注视点位置
    actions_step1,  # (B, 2) - 第2个注视点位置
    actions_step2,  # (B, 2) - 第3个注视点位置
    actions_step3,  # (B, 2) - 第4个注视点位置
]
# 总共4个注视点位置（如果seq_l=4）

# 可以转换为序列格式：
scanpath_sequence = torch.stack(expected_outputs['actions'], dim=1)
# 输出: (B, seq_l, 2) = (B, 4, 2)
# 这就是一个完整的扫描路径序列！
```

---

## 📊 与ScanDMM的对比

### AdaptiveNN的注视点序列

| 属性 | AdaptiveNN |
|------|-----------|
| **序列长度** | 固定4步（seq_l=4，可调） |
| **坐标格式** | 2D归一化坐标 (x, y) ∈ [0, 1] |
| **生成方式** | Policy Network（策略网络） |
| **训练方式** | PPO强化学习 |
| **用途** | 中间过程，用于提取局部特征 |
| **最终目标** | 图像分类 |
| **序列形式** | `actions`列表：[(B,2), (B,2), (B,2), (B,2)] |

### ScanDMM的扫描路径序列

| 属性 | ScanDMM |
|------|---------|
| **序列长度** | 固定30步（T_max=30，可调） |
| **坐标格式** | 3D球面坐标 (x, y, z) |
| **生成方式** | Emitter（发射器） |
| **训练方式** | 变分推断（Pyro） |
| **用途** | 最终输出，预测眼动轨迹 |
| **最终目标** | 扫描路径预测 |
| **序列形式** | `scanpaths`: (B, T, 3) |

---

## 💡 关键洞察

### 1. AdaptiveNN确实有序列生成能力！

**重要发现：**
- ✅ AdaptiveNN通过Policy Network生成注视点序列
- ✅ 这个序列是时间序列（按时间步生成）
- ✅ 每个注视点位置都是基于当前状态动态选择的
- ✅ 序列可以提取出来：`expected_outputs['actions']`

### 2. 两者的相似性更高了！

**相似点：**
- ✅ 都生成注视点序列
- ✅ 都是时间序列模型
- ✅ 都基于当前状态预测下一个注视点
- ✅ 都有状态更新机制

**差异：**
- AdaptiveNN：序列是中间过程，最终目标是分类
- ScanDMM：序列是最终输出，最终目标是序列预测

### 3. 迁移的可行性更高了！

**这个发现说明：**
- ✅ AdaptiveNN已经有生成注视点序列的机制
- ✅ Policy Network就是"生成注视点的网络"
- ✅ PPO训练就是"训练序列生成"的方法
- ✅ 只需要改变训练目标（从分类准确率 → 序列预测准确率）

---

## 🚀 对迁移方案的影响

### 方案一：保留DMM + 融合AdaptiveNN（推荐）

**这个发现增强了这个方案的可行性：**
- ✅ AdaptiveNN的Policy Network就是生成注视点的机制
- ✅ 可以将Policy Network的输出（注视点序列）作为监督信号
- ✅ 或者直接用Policy Network替代Emitter

### 方案二：用RL替换DMM（现在更可行了！）

**这个发现使RL替换方案更可行：**

**关键洞察：**
- ✅ AdaptiveNN的Policy Network已经在做"序列生成"（生成注视点序列）
- ✅ 只是最终目标不同（分类 vs 序列预测）
- ✅ 我们可以将Policy Network的输出作为最终输出（而不是中间过程）
- ✅ 奖励函数设计更清晰了：可以使用序列预测误差作为奖励

**新的理解：**
```
AdaptiveNN（当前）：
Policy Network → 注视点序列 → 提取特征 → 分类

AdaptiveNN应用到ScanDMM（可能）：
Policy Network → 注视点序列 → 这就是最终输出！
```

---

## 🔄 重新理解AdaptiveNN

### 原来理解（不完整）

```
AdaptiveNN = 分类任务
- Glance阶段：快速浏览
- Focus阶段：局部聚焦
- 最终输出：类别概率
```

### 新理解（完整）

```
AdaptiveNN = 分类任务 + 序列生成（隐藏的！）
- Glance阶段：快速浏览
- Focus阶段：
  - Policy Network → 生成注视点序列（隐藏的输出）
  - 根据注视点提取特征
  - 特征融合
- 最终输出：类别概率
- 中间输出：注视点序列（可提取！）
```

---

## 📈 迁移方案的新理解

### 方案A：直接提取AdaptiveNN的序列

**思路：**
- 使用AdaptiveNN的Policy Network生成注视点序列
- 将`expected_outputs['actions']`作为最终输出
- 用扫描路径预测的损失训练Policy Network

**优点：**
- ✅ 直接复用AdaptiveNN的序列生成机制
- ✅ 只需要改训练目标

**挑战：**
- ⚠️ 序列长度固定为4步（可能需要扩展）
- ⚠️ 坐标格式是2D（需要转换为3D球面坐标）

### 方案B：借鉴机制，重新设计

**思路：**
- 借鉴AdaptiveNN的Policy Network机制
- 适配到360度图像和扫描路径预测任务
- 设计新的奖励函数（序列预测误差）

**优点：**
- ✅ 可以灵活设计序列长度
- ✅ 可以适配3D球面坐标
- ✅ 可以针对扫描路径预测优化

---

## 🎯 关键结论

### ✅ 重要发现总结

1. **AdaptiveNN确实生成注视点序列**
   - 通过Policy Network生成
   - 存储在`expected_outputs['actions']`中
   - 序列长度：seq_l（默认4步）

2. **这个序列就是眼动序列/扫描路径**
   - 格式：(B, seq_l, 2) - 2D归一化坐标
   - 生成方式：基于当前状态动态选择
   - 训练方式：PPO强化学习

3. **迁移的可行性大大提高**
   - AdaptiveNN已经有序列生成能力
   - 只需要改变训练目标
   - Policy Network可以复用或借鉴

4. **两个方案的可行性都提升了**
   - 方案一（融合）：可以更直接地融合Policy Network
   - 方案二（RL替换）：可行性大大提升，因为AdaptiveNN已经在做类似的事情

---

## 💭 对询问师姐的建议更新

**可以在询问师姐时提到这个发现：**

"我仔细研究了AdaptiveNN的源代码，发现一个有趣的细节：虽然AdaptiveNN的最终目标是分类，但在中间过程中，它通过Policy Network生成了一序列的注视点位置（存储在`expected_outputs['actions']`中），这个序列就是眼动序列！这个发现让我觉得迁移到扫描路径预测任务可能更可行，因为AdaptiveNN本身就有序列生成的能力，只是最终目标不同。不知道师姐您怎么看这个发现？"

---

## 📝 下一步建议

### 1. 深入研究AdaptiveNN的序列生成机制

- 分析Policy Network如何生成序列
- 分析状态如何更新
- 分析PPO如何训练序列生成

### 2. 设计迁移方案

- 方案A：直接提取序列（简单但有限制）
- 方案B：借鉴机制重新设计（灵活但复杂）

### 3. 与师姐讨论

- 分享这个发现
- 讨论迁移策略
- 获取专业建议

---

**这个发现非常重要！它大大提升了迁移的可行性！** 🎉

