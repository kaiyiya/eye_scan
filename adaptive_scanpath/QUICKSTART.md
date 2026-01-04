# ✅ 问题已解决 - 数据准备说明

## 📝 回答你的问题

> **Q**: 训练数据难道不是开源数据集获取的吗,不用我额外提供吧?

**A**: 你说得对！有多个开源数据集可以使用，我为你准备了3种方案：

---

## 🎯 三种数据获取方案

### ⭐ 方案1: 合成数据（最简单，推荐新手）

**完全自动生成，无需下载！**

```bash
# 一条命令生成100个样本（1分钟完成）
python prepare_data.py --num_samples 100
```

**特点**：
- ✅ 完全自动，无需下载
- ✅ 模拟真实眼动模式
- ✅ 适合快速验证和测试
- ✅ 5分钟内可以开始训练

**立即试试**：
```bash
cd E:\eye_scan\eye_scan\adaptive_scanpath
python prepare_data.py --num_samples 100
python quickstart.py
```

---

### 🔥 方案2: Salient360! 数据集（推荐）

**专业级360度图像眼动数据集**

- **官网**: https://salient360.di.fc.ul.pt/
- **数据量**: 40+张360度全景图像
- **标注**: 多个人工标注的眼动扫描路径
- **费用**: 完全免费（学术研究用）
- **许可证**: 学术用途免费

**获取步骤**：
1. 访问官网注册账号（免费）
2. 下载数据集（约2GB）
3. 解压到 `data/salient360/` 目录
4. 运行转换脚本

```bash
# 查看详细下载指南
python prepare_data.py --mode info
```

---

### 📚 方案3: 其他开源数据集

#### SALICON
- **官网**: http://salicon.net/
- **数据量**: 10,000张图像
- **格式**: 2D图像 + 眼动数据
- **适用**: 预训练，然后迁移到360度

#### OSIE
- **GitHub**: https://github.com/NUS-VIP/osie
- **数据量**: 700张图像
- **质量**: 高质量人工标注

---

## 🚀 立即开始（推荐流程）

### 今天：使用合成数据验证
```bash
# 1. 生成数据（1分钟）
python prepare_data.py --num_samples 100

# 2. 快速训练（5分钟）
python quickstart.py

# 3. 查看结果
ls checkpoints/
```

### 本周：下载数据集
- 访问 Salient360! 官网
- 注册并下载数据集
- 在真实数据上训练

---

## 📊 数据对比

| 方案 | 时间 | 质量 | 难度 | 推荐度 |
|------|------|------|------|--------|
| 合成数据 | 1分钟 | ⭐⭐⭐ | ⭐ | ⭐⭐⭐⭐⭐ |
| Salient360! | 30分钟 | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| SALICON | 1小时 | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ |

---

## 💡 我的建议

**对于快速验证和开发**：
- ✅ 使用合成数据
- ✅ 5分钟内就能跑起来
- ✅ 验证代码和模型架构

**对于学术研究/论文**：
- ✅ 使用 Salient360! 数据集
- ✅ 真实的眼动数据
- ✅ 可重复的研究结果

**最佳实践**：
1. 先用合成数据快速验证（1天）
2. 再用Salient360!微调（1周）
3. 最后用SALICON大规模预训练（可选）

---

## 📖 完整文档

- **数据准备指南**: [DATA_GUIDE.md](DATA_GUIDE.md)
- **项目README**: [README.md](README.md)
- **配置说明**: [config.py](config.py)

---

## 🎉 现在就开始吧！

```bash
# 进入项目目录
cd E:\eye_scan\eye_scan\adaptive_scanpath

# 准备数据（100个样本，1分钟）
python prepare_data.py --num_samples 100

# 快速训练（5个epoch，约5分钟）
python quickstart.py

# 完成！
```

**不需要任何额外操作，所有数据都是自动生成的！** 🎊

---

## ❓ 还有问题？

- 如何下载Salient360!? → 查看 [DATA_GUIDE.md](DATA_GUIDE.md)
- 如何使用自己的数据? → 查看 [DATA_GUIDE.md](DATA_GUIDE.md) 的"自定义数据集"部分
- 合成数据够用吗？ → 用于验证足够，发表论文建议用真实数据

**准备好了吗？运行 `python prepare_data.py` 开始吧！** 🚀
