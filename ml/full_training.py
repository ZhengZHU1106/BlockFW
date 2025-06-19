#!/usr/bin/env python3
"""
完整训练脚本
支持 BNaT 数据集的完整训练，包括 Merged 和 DL 数据集
"""

import os
import sys
import logging
import time
import pandas as pd
import numpy as np
from datetime import datetime
import json
import glob
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def find_bnat_datasets():
    """查找 BNaT 数据集文件"""
    print("🔍 查找 BNaT 数据集...")
    
    # 可能的路径模式
    base_paths = [
        '.',
        'BlockFW',
        'BlockFW/BlockFW',
        '/content/BlockFW',
        '/content/BlockFW/BlockFW'
    ]
    
    dataset_patterns = [
        'ml/data/BNaT-master/Merged_dataset/*.csv',
        'ml/data/BNaT-master/DL_dataset/*.csv',
        'data/BNaT-master/Merged_dataset/*.csv',
        'data/BNaT-master/DL_dataset/*.csv',
        'BNaT-master/Merged_dataset/*.csv',
        'BNaT-master/DL_dataset/*.csv'
    ]
    
    found_datasets = {}
    
    for base_path in base_paths:
        if not os.path.exists(base_path):
            continue
            
        print(f"检查路径: {base_path}")
        
        for pattern in dataset_patterns:
            full_pattern = os.path.join(base_path, pattern)
            files = glob.glob(full_pattern)
            
            if files:
                dataset_type = 'Merged' if 'Merged' in pattern else 'DL'
                if dataset_type not in found_datasets:
                    found_datasets[dataset_type] = []
                found_datasets[dataset_type].extend(files)
                print(f"  ✓ 找到 {dataset_type} 数据集: {len(files)} 个文件")
                for file in files[:3]:  # 只显示前3个文件
                    print(f"    - {file}")
                if len(files) > 3:
                    print(f"    ... 还有 {len(files) - 3} 个文件")
    
    return found_datasets

def load_dataset_files(file_paths, dataset_type):
    """加载数据集文件"""
    print(f"\n📊 加载 {dataset_type} 数据集...")
    
    all_data = []
    total_rows = 0
    
    for file_path in file_paths:
        try:
            # 检查文件大小
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
            print(f"加载文件: {file_path} ({file_size:.1f} MB)")
            
            # 读取数据
            data = pd.read_csv(file_path, header=None)
            
            # 检查列数
            if len(data.columns) == 22:  # 21个特征 + 1个标签
                feature_cols = [f'feature_{i}' for i in range(21)]
                data.columns = feature_cols + ['label']
            elif len(data.columns) == 21:  # 只有特征，没有标签
                feature_cols = [f'feature_{i}' for i in range(21)]
                data.columns = feature_cols
                # 为 DL 数据集添加标签（假设都是正常流量）
                data['label'] = 'Normal'
            else:
                print(f"⚠️ 跳过文件 {file_path}: 列数不符合预期 ({len(data.columns)})")
                continue
            
            all_data.append(data)
            total_rows += len(data)
            print(f"  ✓ 加载成功: {len(data)} 行")
            
        except Exception as e:
            print(f"❌ 加载文件 {file_path} 失败: {e}")
    
    if not all_data:
        print(f"❌ 没有成功加载任何 {dataset_type} 数据")
        return None
    
    # 合并数据
    merged_data = pd.concat(all_data, ignore_index=True)
    print(f"✓ {dataset_type} 数据集合并完成: {merged_data.shape} (总行数: {total_rows})")
    
    return merged_data

def preprocess_data(data, dataset_name):
    """数据预处理"""
    print(f"\n🔧 预处理 {dataset_name} 数据...")
    
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    
    # 分离特征和标签
    X = data.drop('label', axis=1)
    y = data['label']
    
    print(f"特征数据形状: {X.shape}")
    print(f"标签数据形状: {y.shape}")
    
    # 检查数据类型
    categorical_features = []
    numerical_features = []
    
    for col in X.columns:
        if X[col].dtype == 'object' or X[col].dtype == 'string':
            categorical_features.append(col)
        else:
            numerical_features.append(col)
    
    print(f"发现 {len(categorical_features)} 个分类特征，{len(numerical_features)} 个数值特征")
    
    # 对分类特征进行编码
    if categorical_features:
        print("编码分类特征...")
        for col in categorical_features:
            unique_values = X[col].unique()
            print(f"  {col}: {unique_values[:5]} -> 编码完成")
            X[col] = pd.Categorical(X[col]).codes
    
    # 处理缺失值
    missing_values = X.isnull().sum()
    if missing_values.sum() > 0:
        print(f"发现缺失值: {missing_values[missing_values > 0]}")
        X = X.fillna(X.median())
        print("已用中位数填充缺失值")
    else:
        print("✓ 没有缺失值")
    
    # 编码标签
    print("编码标签...")
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    print(f"标签类别: {list(le.classes_)}")
    print(f"标签分布: {np.bincount(y_encoded)}")
    
    # 标准化
    print("标准化特征...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"✓ 标准化完成，特征形状: {X_scaled.shape}")
    
    return X_scaled, y_encoded, scaler, le

def train_models(X_scaled, y_encoded, dataset_name):
    """训练模型"""
    print(f"\n🤖 训练 {dataset_name} 模型...")
    
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report
    import time
    
    # 分割数据
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    print(f"训练集: {X_train.shape}")
    print(f"测试集: {X_test.shape}")
    
    results = {}
    
    # 1. 随机森林
    print("\n训练随机森林...")
    start_time = time.time()
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_time = time.time() - start_time
    
    y_pred_rf = rf.predict(X_test)
    rf_accuracy = accuracy_score(y_test, y_pred_rf)
    
    results['random_forest'] = {
        'model': rf,
        'time': rf_time,
        'accuracy': rf_accuracy,
        'gpu': False,
        'predictions': y_pred_rf,
        'true_labels': y_test
    }
    
    print(f"随机森林: {rf_time:.2f}秒, 准确率: {rf_accuracy:.4f}")
    
    # 2. XGBoost（GPU）
    try:
        import xgboost as xgb
        print("\n训练 XGBoost (GPU)...")
        
        start_time = time.time()
        xgb_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            tree_method='hist',
            device='cuda',
            random_state=42
        )
        xgb_model.fit(X_train, y_train)
        xgb_time = time.time() - start_time
        
        y_pred_xgb = xgb_model.predict(X_test)
        xgb_accuracy = accuracy_score(y_test, y_pred_xgb)
        
        results['xgboost'] = {
            'model': xgb_model,
            'time': xgb_time,
            'accuracy': xgb_accuracy,
            'gpu': True,
            'predictions': y_pred_xgb,
            'true_labels': y_test
        }
        
        print(f"XGBoost: {xgb_time:.2f}秒, 准确率: {xgb_accuracy:.4f}")
        
    except Exception as e:
        print(f"⚠️ XGBoost 训练失败: {e}")
    
    # 3. PyTorch 神经网络（GPU）
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        
        if torch.cuda.is_available():
            print("\n训练 PyTorch 神经网络 (GPU)...")
            
            class FullNN(nn.Module):
                def __init__(self, input_size, hidden_size, output_size):
                    super(FullNN, self).__init__()
                    self.fc1 = nn.Linear(input_size, hidden_size)
                    self.bn1 = nn.BatchNorm1d(hidden_size)
                    self.relu = nn.ReLU()
                    self.dropout = nn.Dropout(0.3)
                    self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
                    self.bn2 = nn.BatchNorm1d(hidden_size // 2)
                    self.fc3 = nn.Linear(hidden_size // 2, hidden_size // 4)
                    self.bn3 = nn.BatchNorm1d(hidden_size // 4)
                    self.fc4 = nn.Linear(hidden_size // 4, output_size)
                    
                def forward(self, x):
                    x = self.fc1(x)
                    x = self.bn1(x)
                    x = self.relu(x)
                    x = self.dropout(x)
                    x = self.fc2(x)
                    x = self.bn2(x)
                    x = self.relu(x)
                    x = self.dropout(x)
                    x = self.fc3(x)
                    x = self.bn3(x)
                    x = self.relu(x)
                    x = self.dropout(x)
                    x = self.fc4(x)
                    return x
            
            # 转换为张量
            X_train_tensor = torch.FloatTensor(X_train).cuda()
            y_train_tensor = torch.LongTensor(y_train).cuda()
            X_test_tensor = torch.FloatTensor(X_test).cuda()
            y_test_tensor = torch.LongTensor(y_test).cuda()
            
            # 创建模型
            input_size = X_train.shape[1]
            output_size = len(np.unique(y_encoded))
            model = FullNN(input_size, 256, output_size).cuda()
            
            # 优化器和损失函数
            optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
            criterion = nn.CrossEntropyLoss()
            
            # 训练
            start_time = time.time()
            model.train()
            for epoch in range(100):
                optimizer.zero_grad()
                outputs = model(X_train_tensor)
                loss = criterion(outputs, y_train_tensor)
                loss.backward()
                optimizer.step()
                
                if (epoch + 1) % 20 == 0:
                    print(f"  Epoch {epoch+1}/100, Loss: {loss.item():.4f}")
            
            torch.cuda.synchronize()
            nn_time = time.time() - start_time
            
            # 评估
            model.eval()
            with torch.no_grad():
                outputs = model(X_test_tensor)
                _, predicted = torch.max(outputs, 1)
                nn_accuracy = (predicted == y_test_tensor).float().mean().item()
            
            results['pytorch_nn'] = {
                'model': model,
                'time': nn_time,
                'accuracy': nn_accuracy,
                'gpu': True,
                'predictions': predicted.cpu().numpy(),
                'true_labels': y_test
            }
            
            print(f"PyTorch NN: {nn_time:.2f}秒, 准确率: {nn_accuracy:.4f}")
            
        else:
            print("⚠️ GPU 不可用，跳过 PyTorch 训练")
            
    except Exception as e:
        print(f"⚠️ PyTorch 训练失败: {e}")
    
    return results

def save_results(results, scaler, label_encoder, dataset_name):
    """保存结果"""
    print(f"\n💾 保存 {dataset_name} 结果...")
    
    # 创建模型目录
    os.makedirs('BlockFW/ml/models', exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 生成报告
    report = {
        'dataset_name': dataset_name,
        'training_time': datetime.now().isoformat(),
        'data_info': {
            'total_samples': len(label_encoder.classes_),
            'classes': list(label_encoder.classes_),
            'class_distribution': np.bincount(label_encoder.transform(label_encoder.classes_)).tolist()
        },
        'results': {
            name: {
                'time': result['time'],
                'accuracy': result['accuracy'],
                'gpu': result['gpu']
            } for name, result in results.items()
        },
        'best_model': max(results.keys(), key=lambda k: results[k]['accuracy']),
        'best_accuracy': max(result['accuracy'] for result in results.values())
    }
    
    report_path = f'BlockFW/ml/models/full_{dataset_name}_report_{timestamp}.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"报告已保存: {report_path}")
    
    # 尝试保存模型
    try:
        import joblib
        import torch
        
        for model_name, result in results.items():
            if model_name == 'pytorch_nn':
                model_path = f'BlockFW/ml/models/full_{dataset_name}_{model_name}_{timestamp}.pth'
                torch.save(result['model'].state_dict(), model_path)
            else:
                model_path = f'BlockFW/ml/models/full_{dataset_name}_{model_name}_{timestamp}.pkl'
                joblib.dump(result['model'], model_path)
            
            print(f"保存 {model_name}: {model_path}")
        
        # 保存预处理器
        scaler_path = f'BlockFW/ml/models/full_{dataset_name}_scaler_{timestamp}.pkl'
        joblib.dump(scaler, scaler_path)
        
        encoder_path = f'BlockFW/ml/models/full_{dataset_name}_encoder_{timestamp}.pkl'
        joblib.dump(label_encoder, encoder_path)
        
    except Exception as e:
        print(f"⚠️ 模型保存失败: {e}")
        print("但报告已成功保存")
    
    return report

def main():
    """主函数"""
    print("🚀 BNaT 完整训练")
    print("="*60)
    
    # 1. 查找数据集
    datasets = find_bnat_datasets()
    
    if not datasets:
        print("❌ 未找到任何 BNaT 数据集")
        print("\n请检查数据集路径，确保包含以下文件之一:")
        print("- ml/data/BNaT-master/Merged_dataset/*.csv")
        print("- ml/data/BNaT-master/DL_dataset/*.csv")
        return
    
    all_results = {}
    
    # 2. 处理每个数据集
    for dataset_type, file_paths in datasets.items():
        print(f"\n{'='*60}")
        print(f"处理 {dataset_type} 数据集")
        print(f"{'='*60}")
        
        # 加载数据
        data = load_dataset_files(file_paths, dataset_type)
        if data is None:
            continue
        
        # 预处理数据
        X_scaled, y_encoded, scaler, le = preprocess_data(data, dataset_type)
        
        # 训练模型
        results = train_models(X_scaled, y_encoded, dataset_type)
        
        if not results:
            print(f"❌ {dataset_type} 数据集训练失败")
            continue
        
        # 保存结果
        report = save_results(results, scaler, le, dataset_type)
        all_results[dataset_type] = report
        
        # 显示结果
        print(f"\n📊 {dataset_type} 训练结果:")
        for name, result in results.items():
            gpu_status = "GPU" if result['gpu'] else "CPU"
            print(f"  {name} ({gpu_status}): {result['time']:.2f}秒, 准确率: {result['accuracy']:.4f}")
        
        print(f"最佳模型: {report['best_model']}")
        print(f"最佳准确率: {report['best_accuracy']:.4f}")
    
    # 3. 总结
    print(f"\n{'='*60}")
    print("🎉 完整训练总结")
    print(f"{'='*60}")
    
    for dataset_type, report in all_results.items():
        print(f"\n{dataset_type} 数据集:")
        print(f"  最佳模型: {report['best_model']}")
        print(f"  最佳准确率: {report['best_accuracy']:.4f}")
        print(f"  训练时间: {report['training_time']}")
    
    print(f"\n所有模型文件已保存到 BlockFW/ml/models/ 目录")

if __name__ == "__main__":
    main() 