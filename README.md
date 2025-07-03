# BlockFW - 区块链防火墙系统

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-orange.svg)](https://xgboost.readthedocs.io)

一个基于人工智能和区块链技术的分布式防火墙系统，集成AI网络入侵检测、智能合约治理和实时API服务。

## 🚀 项目概览

BlockFW 采用三层架构设计：
- **AI检测层**: 基于XGBoost的高性能网络入侵检测模型
- **区块链治理层**: 智能合约实现的去中心化防火墙规则管理
- **API服务层**: FastAPI提供的统一接口服务

## ✨ 核心特性

- 🤖 **智能检测**: XGBoost GPU模型，准确率98.10%
- ⛓️ **区块链治理**: 多签名机制，去中心化规则管理
- 🚄 **高性能API**: FastAPI异步处理，支持高并发
- 📊 **实时监控**: 检测统计、历史记录、攻击分析
- 🔄 **自动联动**: AI检测结果自动触发智能合约

## 📋 快速开始

### 环境要求

- Python 3.8+
- Node.js 16+ (前端开发)
- Git

### 安装依赖

```bash
# 克隆项目
git clone <repository-url>
cd BlockFW

# 安装后端依赖
cd backend
pip install -r requirements.txt

# 安装机器学习依赖
cd ../ml
pip install -r requirements.txt
```

### 启动服务

```bash
# 启动后端API服务
cd backend
python app.py
```

API文档: http://127.0.0.1:8000/docs

### 测试系统

```bash
# 运行完整测试
cd backend
python test_step2.py

# 快速配置检查
python check_api_config.py
```

## 📚 API使用示例

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
    # ... 其余15个数值特征
]

response = requests.post(
    "http://127.0.0.1:8000/api/detection/single",
    json={"features": features}
)

result = response.json()
print(f"是否攻击: {result['is_attack']}")
print(f"攻击类型: {result['predicted_label']}")
print(f"置信度: {result['confidence']:.2%}")
```

### 获取检测统计

```python
response = requests.get(
    "http://127.0.0.1:8000/api/detection/statistics"
)
stats = response.json()['data']
print(f"总检测次数: {stats['total_detections']}")
print(f"攻击检测率: {stats['attack_rate']:.2%}")
```

## 🏗️ 项目结构

```
BlockFW/
├── backend/                      # 后端API服务
│   ├── app.py                    # FastAPI主应用
│   ├── api/detection.py          # AI检测API
│   ├── services/
│   │   ├── ai_detection_service_v2.py
│   │   └── bnat_preprocessor.py
│   └── models/schemas.py         # 数据模型
├── ml/                           # AI检测模块
│   ├── ml/models/               # 训练好的模型
│   └── src/inference_service.py
├── contracts/                    # 智能合约
│   └── FirewallRules.sol
└── scripts/                      # 工具脚本
    └── test_contract_features.py
```

## 🎯 开发状态

- ✅ **第一阶段**: AI检测模块 (已完成)
- ✅ **第二阶段**: 智能合约开发 (已完成)  
- ✅ **第三阶段**: 后端API服务 (步骤1&2已完成)
- 🔄 **当前**: 准备步骤3 - 区块链交互服务
- 📋 **下一步**: 前端界面开发

## 📊 技术规格

| 组件 | 技术栈 | 性能指标 |
|------|--------|----------|
| AI模型 | XGBoost GPU | 准确率98.10% |
| 特征处理 | 21维→28维 | <10ms转换时间 |
| API响应 | FastAPI | <100ms响应时间 |
| 合约 | Solidity | Gas优化 |

## 🔧 开发指南

### 添加新的检测模型

1. 在`ml/src/`目录下创建新的模型文件
2. 在`backend/services/`中创建对应的服务类
3. 在`backend/api/`中添加新的API端点
4. 更新数据模型和文档

### 扩展智能合约功能

1. 修改`contracts/FirewallRules.sol`
2. 更新`scripts/contract_info.json`
3. 在后端服务中添加Web3交互逻辑
4. 编写测试验证新功能

## 🧪 测试

```bash
# 运行所有测试
cd backend
python test_step2.py

# 测试特定功能
python -c "
import requests
response = requests.get('http://127.0.0.1:8000/health')
print(response.json())
"
```

## 📖 文档

- [API文档](http://127.0.0.1:8000/docs) - 在线API文档
- [BNaT特征指南](backend/BNAT_FEATURE_GUIDE.md) - 特征使用说明
- [项目详细总结](AI_DETECTION_MODULE_SUMMARY.md) - 完整技术文档

## 🤝 贡献

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

## 🔗 相关资源

- [BNaT 数据集](https://dohaison.github.io/BNaT/#/)
- [FastAPI 文档](https://fastapi.tiangolo.com/)
- [XGBoost 文档](https://xgboost.readthedocs.io/)
- [Solidity 文档](https://docs.soliditylang.org/)

## 📞 联系方式

如有问题或建议，请创建 [Issue](https://github.com/your-repo/BlockFW/issues) 或联系项目维护者。

---

**⭐ 如果这个项目对您有帮助，请给我们一个星标！** 