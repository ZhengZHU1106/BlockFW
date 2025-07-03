# BNaT特征使用指南

## 概述

BNaT（Blockchain Network Attack Traffic）数据集包含21个特征，用于检测区块链网络中的入侵行为。在训练过程中，3个分类特征被one-hot编码，将特征数扩展为28个。本指南说明如何正确使用API进行检测。

## 特征说明

### 21个原始特征

| # | 特征名 | 类型 | 描述 |
|---|--------|------|------|
| 0 | duration | 数值 | 连接持续时间（秒） |
| 1 | protocol_type | 分类 | 协议类型（tcp, udp, icmp） |
| 2 | service | 分类 | 网络服务（http, ssh等） |
| 3 | src_bytes | 数值 | 源到目标的字节数 |
| 4 | dst_bytes | 数值 | 目标到源的字节数 |
| 5 | flag | 分类 | 连接状态标志 |
| 6-20 | 统计特征 | 数值 | 各种网络流量统计指标 |

### One-hot编码

训练时，3个分类特征被编码：
- `protocol_type`: 3个类别 → 2个二进制特征（drop_first）
- `service`: N个类别 → N-1个二进制特征
- `flag`: M个类别 → M-1个二进制特征

总计：18个数值特征 + 10个编码特征 = 28个特征

## API使用方法

### 1. 使用列表格式（推荐）

```python
import requests

# 按照特征顺序创建列表
features = [
    0,        # duration
    'tcp',    # protocol_type
    'http',   # service
    408,      # src_bytes
    0,        # dst_bytes
    'OTH',    # flag
    14,       # count
    13,       # srv_count
    0,        # serror_rate
    0.64,     # same_srv_rate
    0.36,     # diff_srv_rate
    0,        # srv_serror_rate
    0.31,     # srv_diff_host_rate
    14,       # dst_host_count
    13,       # dst_host_srv_count
    0.64,     # dst_host_same_srv_rate
    0.36,     # dst_host_diff_srv_rate
    0.21,     # dst_host_same_src_port_rate
    0,        # dst_host_serror_rate
    0.31,     # dst_host_srv_diff_host_rate
    0         # dst_host_srv_serror_rate
]

response = requests.post(
    "http://127.0.0.1:8000/api/detection/single",
    json={"features": features}
)
```

### 2. 批量检测

```python
# 多个样本
batch_features = [
    [0, 'tcp', 'http', 408, 0, 'OTH', ...],  # 样本1
    [0, 'tcp', 'ssh', 200, 150, 'SF', ...],  # 样本2
    # 更多样本...
]

response = requests.post(
    "http://127.0.0.1:8000/api/detection/batch",
    json={"features_list": batch_features}
)
```

## 攻击类型

BNaT数据集包含5种标签：
- **Normal**: 正常流量
- **DoS**: 拒绝服务攻击
- **BP**: 暴力破解攻击（Brute Password）
- **FoT**: 交易洪水攻击（Flooding of Transactions）
- **MitM**: 中间人攻击（Man in the Middle）

## 重要提示

### 1. 分类特征的值

确保分类特征使用正确的值：
- `protocol_type`: 使用小写（'tcp', 'udp', 'icmp'）
- `service`: 使用小写（'http', 'ssh', 'other'等）
- `flag`: 使用大写（'OTH', 'SF', 'S1'等）

### 2. 类别映射调整

如果检测结果不准确，可能需要调整`bnat_preprocessor.py`中的类别映射：

```python
self.encoding_map = {
    'protocol_type': ['tcp', 'udp', 'icmp'],  # 添加训练数据中的所有协议
    'service': ['http', 'other', 'ssh', ...],  # 添加所有服务类型
    'flag': ['OTH', 'SF', 'S1', ...]  # 添加所有标志类型
}
```

### 3. 查看训练时的类别

要确定训练时使用的确切类别，可以：

1. 查看训练notebook中的`pd.get_dummies()`输出
2. 检查训练后的特征名称
3. 分析训练数据的唯一值

## 故障排除

### 问题1: "特征预处理失败"

**原因**: 分类特征值不在预定义的类别列表中
**解决**: 检查输入的分类特征值，确保拼写正确

### 问题2: 检测结果不准确

**原因**: One-hot编码映射与训练时不一致
**解决**: 
1. 检查训练notebook中的实际类别
2. 更新`bnat_preprocessor.py`中的`encoding_map`
3. 确保类别顺序与训练时一致

### 问题3: API返回特征维度错误

**原因**: 输入特征数量不是21个
**解决**: 确保提供完整的21个特征，按照正确顺序

## 示例代码

完整的测试脚本请参考：`test_bnat_features.py`

```bash
python test_bnat_features.py
```

这个脚本演示了：
- 单次检测
- 批量检测
- 不同攻击类型的样本
- 获取模型信息

## 总结

正确使用BNaT特征的关键是：
1. 提供21个原始特征（按正确顺序）
2. 分类特征使用正确的字符串值
3. API会自动进行one-hot编码转换
4. 如有问题，检查类别映射配置