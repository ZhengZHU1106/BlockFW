# AI 入侵检测模块

基于 BNaT (Blockchain Network Attack Traffic) 数据集的区块链网络安全入侵检测系统。

## 功能特性

- 🚀 **多模型支持**: 支持随机森林、梯度提升、逻辑回归、SVM、神经网络等多种算法
- 📊 **数据处理**: 自动处理 BNaT 数据集，包括特征提取、标准化、标签编码
- 🔍 **实时检测**: 提供单次检测和批量检测功能
- 📈 **性能评估**: 自动生成模型性能报告和可视化图表
- 💾 **模型管理**: 自动保存和加载训练好的模型及预处理器

## 项目结构

```
ml/
├── src/                           # 核心模块
│   ├── data_processor.py         # 数据处理模块
│   ├── model_trainer.py          # 模型训练模块
│   └── inference_service.py      # 推理服务模块
├── data/                          # 数据集目录
├── models/                        # 训练好的模型
├── notebooks/                     # Jupyter 笔记本
├── train_intrusion_detection.py  # 完整训练脚本
├── test_modules.py               # 模块测试脚本
├── requirements.txt              # 依赖包
└── README.md                     # 说明文档
```

## 快速开始

### 1. 安装依赖

```bash
cd ml
pip install -r requirements.txt
```

### 2. 测试模块

```bash
python test_modules.py
```

### 3. 训练模型

```bash
# 使用 BNaT 数据集训练模型
python train_intrusion_detection.py --data_path ml/data/BNaT_dataset.csv
```

### 4. 使用训练好的模型

```python
from src.inference_service import IntrusionDetectionService

# 初始化检测服务
service = IntrusionDetectionService(
    model_path='ml/models/best_model.pkl',
    scaler_path='ml/models/scaler.pkl',
    encoder_path='ml/models/label_encoder.pkl'
)

# 单次检测
features = np.random.random(10)  # 你的特征向量
result = service.detect_single(features)
print(f"检测结果: {result}")
```

## 详细使用说明

### 数据处理模块

`BNaTDataProcessor` 类负责处理 BNaT 数据集：

```python
from src.data_processor import BNaTDataProcessor

processor = BNaTDataProcessor()

# 加载数据
data = processor.load_data('path_to_BNaT_dataset.csv')

# 数据预处理
data = processor.preprocess_data(data)

# 特征提取
X, features = processor.extract_features(data)

# 准备标签
y = processor.prepare_labels(data)

# 分割数据集
X_train, X_val, X_test, y_train, y_val, y_test = processor.split_data(X, y)
```

### 模型训练模块

`IntrusionDetectionTrainer` 类负责训练多种机器学习模型：

```python
from src.model_trainer import IntrusionDetectionTrainer

trainer = IntrusionDetectionTrainer()
trainer.define_models()

# 训练所有模型
results = trainer.train_all_models(X_train, y_train, X_val, y_val)

# 评估最佳模型
best_model_name = max(results.keys(), key=lambda k: results[k]['score'])
best_model = results[best_model_name]['model']
evaluation = trainer.evaluate_model(best_model, X_test, y_test, best_model_name)

# 保存模型
trainer.save_model(best_model, best_model_name, processor.scaler, processor.label_encoder)
```

### 推理服务模块

`IntrusionDetectionService` 类提供检测功能：

```python
from src.inference_service import IntrusionDetectionService

# 初始化服务
service = IntrusionDetectionService(
    model_path='path_to_model.pkl',
    scaler_path='path_to_scaler.pkl',
    encoder_path='path_to_encoder.pkl'
)

# 单次检测
result = service.detect_single(features)

# 批量检测
results = service.detect_batch(features_list)

# 从 DataFrame 检测
results = service.detect_from_dataframe(df)

# 获取检测摘要
summary = service.get_detection_summary(results)
```

## 支持的模型

1. **随机森林** (Random Forest)
2. **梯度提升** (Gradient Boosting)
3. **逻辑回归** (Logistic Regression)
4. **支持向量机** (SVM)
5. **神经网络** (Neural Network)

## 输出文件

训练完成后会生成以下文件：

- `models/best_model_YYYYMMDD_HHMMSS.pkl`: 最佳模型
- `models/scaler_YYYYMMDD_HHMMSS.pkl`: 标准化器
- `models/label_encoder_YYYYMMDD_HHMMSS.pkl`: 标签编码器
- `models/training_report.json`: 训练报告
- `models/model_comparison.png`: 模型对比图表
- `models/processed_data.pkl`: 处理后的数据

## 配置参数

训练脚本支持以下参数：

- `--data_path`: BNaT 数据集文件路径（必需）
- `--output_dir`: 模型输出目录（默认: ml/models）
- `--test_size`: 测试集比例（默认: 0.2）
- `--val_size`: 验证集比例（默认: 0.2）
- `--random_state`: 随机种子（默认: 42）

## 检测结果格式

检测结果包含以下信息：

```python
{
    'timestamp': '2024-01-01T12:00:00',
    'is_attack': True,
    'predicted_label': 'DDoS',
    'confidence': 0.95,
    'probabilities': {
        'normal': 0.05,
        'DDoS': 0.95
    },
    'features_used': 10
}
```

## 故障排除

### 常见问题

1. **模块导入错误**
   - 确保已安装所有依赖包
   - 检查 Python 路径设置

2. **数据加载失败**
   - 检查数据集文件路径
   - 确认文件格式（CSV 或 JSON）

3. **模型训练失败**
   - 检查内存是否足够
   - 减少参数搜索空间以加快训练

4. **推理服务错误**
   - 确认模型文件路径正确
   - 检查特征维度是否匹配

### 日志文件

训练过程中的日志会保存在 `ml/training.log` 文件中，可用于调试问题。

## 下一步计划

- [ ] 添加深度学习模型支持
- [ ] 实现模型自动更新机制
- [ ] 添加特征重要性分析
- [ ] 支持在线学习
- [ ] 集成到区块链智能合约

## 贡献

欢迎提交 Issue 和 Pull Request 来改进这个项目！

## 许可证

本项目采用 MIT 许可证。 