# BlockFW - 区块链防火墙系统

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-orange.svg)](https://xgboost.readthedocs.io)
[![Solidity](https://img.shields.io/badge/Solidity-0.8+-purple.svg)](https://docs.soliditylang.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> 🛡️ 基于人工智能和区块链技术的分布式防火墙系统

BlockFW 集成AI网络入侵检测、智能合约治理和实时API服务，为区块链网络提供智能化的安全防护解决方案。

## ✨ 核心特性

- 🤖 **AI智能检测**: XGBoost GPU模型，98.10%准确率，支持5种攻击类型
- ⛓️ **区块链治理**: 多签名智能合约，去中心化规则管理
- 🚄 **高性能API**: FastAPI异步处理，响应时间<100ms
- 📊 **实时监控**: 检测统计、历史记录、攻击分析
- 🔄 **自动联动**: AI检测结果自动触发智能合约

## 🏗️ 系统架构

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   AI检测层      │    │  区块链治理层   │    │   API服务层     │
│                 │    │                 │    │                 │
│ XGBoost GPU     │◄──►│ Smart Contract  │◄──►│ FastAPI         │
│ 98.10% 准确率   │    │ 多签名机制      │    │ RESTful API     │
│ 21→28维特征     │    │ 去中心化治理    │    │ 自动文档        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🚀 快速开始

### 环境要求
- Python 3.8+
- Git

### 安装与运行

```bash
# 克隆项目
git clone https://github.com/your-username/BlockFW.git
cd BlockFW

# 安装依赖
cd backend
pip install -r requirements.txt

# 启动API服务
python app.py
```

API文档: http://127.0.0.1:8000/docs

### 快速测试

```bash
# 运行完整测试
cd backend
python test_step2.py

# 检查配置
python check_api_config.py
```

## 📚 API 使用示例

### 单次入侵检测

```python
import requests

# 21维BNaT特征
features = [
    0,        # duration
    'tcp',    # protocol_type  
    'http',   # service
    408,      # src_bytes
    0,        # dst_bytes
    'OTH',    # flag
    14, 13, 0, 0.64, 0.36, 0, 0.31, 14, 13, 0.64, 0.36, 0.21, 0, 0.31, 0
]

response = requests.post(
    "http://127.0.0.1:8000/api/detection/single",
    json={"features": features}
)

result = response.json()
print(f"攻击检测: {result['is_attack']}")
print(f"攻击类型: {result['predicted_label']}")
print(f"置信度: {result['confidence']:.2%}")
```

### 获取检测统计

```python
response = requests.get("http://127.0.0.1:8000/api/detection/statistics")
stats = response.json()['data']
print(f"总检测次数: {stats['total_detections']}")
print(f"攻击检测率: {stats['attack_rate']:.2%}")
```

## 📁 项目结构

```
BlockFW/
├── backend/                    # 后端API服务
│   ├── app.py                 # FastAPI主应用
│   ├── api/detection.py       # AI检测API端点
│   ├── services/              # 核心服务模块
│   │   ├── ai_detection_service_v2.py
│   │   └── bnat_preprocessor.py
│   └── models/schemas.py      # 数据模型
├── ml/                        # 机器学习模块
│   ├── ml/models/            # 训练好的模型文件
│   └── src/inference_service.py
├── contracts/                 # 智能合约
│   └── FirewallRules.sol
├── scripts/                   # 工具脚本
└── frontend/                  # 前端界面（开发中）
```

## 📊 技术指标

| 组件 | 技术栈 | 性能指标 |
|------|--------|----------|
| AI模型 | XGBoost GPU | 准确率98.10% |
| 特征处理 | 21维→28维 | <10ms转换时间 |
| API响应 | FastAPI | <100ms响应时间 |
| 智能合约 | Solidity | Gas优化设计 |

## 🎯 支持的攻击类型

- **BP**: 暴力破解攻击 (Brute Password)
- **DoS**: 拒绝服务攻击
- **FoT**: 交易洪水攻击 (Flooding of Transactions)
- **MitM**: 中间人攻击 (Man in the Middle)
- **Normal**: 正常流量

## 🔧 开发状态

- ✅ **第一阶段**: AI检测模块 (已完成)
- ✅ **第二阶段**: 智能合约开发 (已完成)  
- ✅ **第三阶段**: 后端API服务 (步骤1&2已完成)
- 🔄 **当前**: 区块链交互服务开发中
- 📋 **计划**: 前端界面开发

## 📖 文档

- [API文档](http://127.0.0.1:8000/docs) - 在线API文档
- [BNaT特征指南](backend/BNAT_FEATURE_GUIDE.md) - 特征使用说明
- [AI检测模块总结](AI_DETECTION_MODULE_SUMMARY.md) - 完整技术文档

## 🤝 贡献

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

## 🔗 相关资源

- [BNaT 数据集](https://dohaison.github.io/BNaT/#/)
- [FastAPI 文档](https://fastapi.tiangolo.com/)
- [XGBoost 文档](https://xgboost.readthedocs.io/)
- [Solidity 文档](https://docs.soliditylang.org/)

---

**⭐ 如果这个项目对您有帮助，请给我们一个星标！**