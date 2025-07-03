# 项目结构说明

## 📁 目录结构

```
BlockFW/
├── backend/                            # 🔧 后端API服务
│   ├── app.py                         # FastAPI主应用入口
│   ├── api/                           # API路由模块
│   │   ├── __init__.py
│   │   └── detection.py               # AI检测API端点
│   ├── services/                      # 核心业务服务
│   │   ├── __init__.py
│   │   ├── ai_detection_service_v2.py # AI检测服务主模块
│   │   ├── bnat_preprocessor.py       # BNaT特征预处理器
│   │   ├── ai_blockchain_integration.py # AI-区块链集成服务
│   │   └── blockchain_service.py      # 区块链交互服务
│   ├── models/                        # 数据模型定义
│   │   ├── __init__.py
│   │   └── schemas.py                 # Pydantic数据模型
│   ├── requirements.txt               # 后端依赖包列表
│   ├── test_step2.py                  # API功能测试脚本
│   ├── check_api_config.py            # 配置检查脚本
│   ├── start_server.py                # 服务器启动脚本
│   ├── verify_step1.py                # 步骤1验证脚本
│   └── BNAT_FEATURE_GUIDE.md          # BNaT特征使用指南
├── ml/                                # 🤖 机器学习模块
│   ├── ml/models/                     # 训练好的模型文件
│   │   ├── xgboost_gpu_20250702_073122.pkl     # XGBoost GPU模型
│   │   ├── scaler_20250702_073122.pkl          # 特征标准化器  
│   │   └── label_encoder_20250702_073122.pkl   # 标签编码器
│   ├── src/                           # 推理服务源码
│   │   └── inference_service.py       # 模型推理服务
│   ├── Block模型训练.ipynb             # 模型训练Jupyter笔记本
│   └── requirements.txt               # ML模块依赖包
├── contracts/                         # ⛓️ 智能合约模块
│   └── FirewallRules.sol              # 防火墙规则智能合约
├── scripts/                           # 🛠️ 工具脚本集合
│   ├── attacker.py                    # 攻击流量模拟器（测试用）
│   ├── apply_firewall.py              # 防火墙规则应用脚本
│   ├── deploy_contract.py             # 智能合约部署脚本
│   ├── test_contract_features.py      # 合约功能测试脚本
│   ├── contract_info.json             # 合约部署信息
│   └── update_rules.py                # 规则更新脚本
├── frontend/                          # 🎨 前端界面（开发中）
│   ├── components/                    # React组件目录
│   └── pages/                         # 页面组件目录
├── .gitignore                         # Git忽略规则
├── LICENSE                            # MIT开源许可证
├── README.md                          # 项目说明文档
├── AI_DETECTION_MODULE_SUMMARY.md     # AI检测模块详细总结
├── STRUCTURE.md                       # 项目结构说明（本文档）
└── 2203.11076v4.pdf                   # 相关学术论文
```

## 📋 核心模块说明

### 🔧 后端API服务 (`backend/`)

| 文件 | 功能 | 说明 |
|------|------|------|
| `app.py` | FastAPI主应用 | API服务入口，路由配置 |
| `api/detection.py` | AI检测API | 单次/批量检测端点 |
| `services/ai_detection_service_v2.py` | AI检测核心服务 | 模型加载，特征处理，检测推理 |
| `services/bnat_preprocessor.py` | 特征预处理器 | 21维→28维特征转换 |
| `models/schemas.py` | 数据模型 | Pydantic请求/响应模型 |

### 🤖 机器学习模块 (`ml/`)

| 文件 | 功能 | 说明 |
|------|------|------|
| `ml/models/*.pkl` | 训练好的模型 | XGBoost模型、标准化器、编码器 |
| `src/inference_service.py` | 推理服务 | 独立的模型推理接口 |
| `Block模型训练.ipynb` | 训练笔记本 | 完整的模型训练流程 |

### ⛓️ 智能合约模块 (`contracts/`)

| 文件 | 功能 | 说明 |
|------|------|------|
| `FirewallRules.sol` | 智能合约 | 多签名端口封锁，攻击模式管理 |

### 🛠️ 工具脚本 (`scripts/`)

| 文件 | 功能 | 说明 |
|------|------|------|
| `attacker.py` | 攻击模拟 | 生成测试流量，验证检测能力 |
| `deploy_contract.py` | 合约部署 | 智能合约部署到区块链 |
| `test_contract_features.py` | 合约测试 | 验证智能合约功能 |
| `apply_firewall.py` | 规则应用 | 将区块链规则应用到防火墙 |

## 🚀 数据流向

```
用户请求 → FastAPI → AI检测服务 → BNaT预处理器 → XGBoost模型 → 检测结果
    ↓
检测结果 → 区块链服务 → 智能合约 → 端口封锁决策 → 防火墙规则更新
```

## 📊 文件重要性分级

### 🔥 核心文件（必需）
- `backend/app.py` - API入口
- `backend/services/ai_detection_service_v2.py` - AI检测核心
- `ml/ml/models/*.pkl` - 训练好的模型
- `contracts/FirewallRules.sol` - 智能合约

### ⭐ 重要文件
- `backend/api/detection.py` - API端点
- `backend/services/bnat_preprocessor.py` - 特征处理
- `backend/models/schemas.py` - 数据模型

### 📝 配置和文档
- `README.md` - 项目文档
- `requirements.txt` - 依赖管理
- `.gitignore` - Git配置
- `LICENSE` - 开源许可

### 🧪 测试和工具
- `backend/test_step2.py` - 功能测试
- `scripts/*.py` - 工具脚本
- `ml/Block模型训练.ipynb` - 训练笔记本

## 🔧 开发建议

1. **修改API功能**: 主要编辑 `backend/api/` 和 `backend/services/`
2. **调整AI模型**: 修改 `ml/src/` 和重新训练模型
3. **扩展智能合约**: 更新 `contracts/FirewallRules.sol`
4. **添加新功能**: 在对应模块下创建新文件
5. **测试验证**: 运行 `backend/test_step2.py` 确保功能正常