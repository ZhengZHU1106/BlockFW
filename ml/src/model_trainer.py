"""
入侵检测模型训练模块
基于 BNaT 数据集训练多种机器学习模型
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV, cross_val_score
import joblib
import os
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IntrusionDetectionTrainer:
    """入侵检测模型训练器"""
    
    def __init__(self, models_dir="ml/models"):
        self.models_dir = models_dir
        self.models = {}
        self.best_model = None
        self.best_score = 0
        
        # 确保模型目录存在
        os.makedirs(models_dir, exist_ok=True)
        
    def define_models(self):
        """定义要训练的模型"""
        self.models = {
            'random_forest': {
                'model': RandomForestClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10]
                }
            },
            'gradient_boosting': {
                'model': GradientBoostingClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.1, 0.2],
                    'max_depth': [3, 5, 7]
                }
            },
            'logistic_regression': {
                'model': LogisticRegression(random_state=42, max_iter=1000),
                'params': {
                    'C': [0.1, 1, 10],
                    'penalty': ['l1', 'l2']
                }
            },
            'svm': {
                'model': SVC(random_state=42, probability=True),
                'params': {
                    'C': [0.1, 1, 10],
                    'kernel': ['rbf', 'linear']
                }
            },
            'neural_network': {
                'model': MLPClassifier(random_state=42, max_iter=500),
                'params': {
                    'hidden_layer_sizes': [(50,), (100,), (50, 25)],
                    'alpha': [0.0001, 0.001, 0.01]
                }
            }
        }
        
        logger.info(f"定义了 {len(self.models)} 个模型")
    
    def train_model(self, model_name, X_train, y_train, X_val, y_val):
        """训练单个模型"""
        logger.info(f"开始训练模型: {model_name}")
        
        model_info = self.models[model_name]
        model = model_info['model']
        params = model_info['params']
        
        # 使用网格搜索进行超参数调优
        grid_search = GridSearchCV(
            model, params, cv=3, scoring='accuracy', n_jobs=-1, verbose=1
        )
        grid_search.fit(X_train, y_train)
        
        # 获取最佳模型
        best_model = grid_search.best_estimator_
        
        # 在验证集上评估
        val_score = best_model.score(X_val, y_val)
        
        logger.info(f"{model_name} 最佳参数: {grid_search.best_params_}")
        logger.info(f"{model_name} 验证集准确率: {val_score:.4f}")
        
        return best_model, val_score, grid_search.best_params_
    
    def train_all_models(self, X_train, y_train, X_val, y_val):
        """训练所有模型"""
        logger.info("开始训练所有模型...")
        
        results = {}
        
        for model_name in self.models.keys():
            try:
                model, score, params = self.train_model(
                    model_name, X_train, y_train, X_val, y_val
                )
                results[model_name] = {
                    'model': model,
                    'score': score,
                    'params': params
                }
                
                # 更新最佳模型
                if score > self.best_score:
                    self.best_score = score
                    self.best_model = model
                    
            except Exception as e:
                logger.error(f"训练模型 {model_name} 时出错: {str(e)}")
                continue
        
        logger.info(f"所有模型训练完成，最佳模型准确率: {self.best_score:.4f}")
        return results
    
    def evaluate_model(self, model, X_test, y_test, model_name="model"):
        """评估模型性能"""
        logger.info(f"评估模型: {model_name}")
        
        # 预测
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        # 计算指标
        accuracy = accuracy_score(y_test, y_pred)
        
        # 分类报告
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # 混淆矩阵
        cm = confusion_matrix(y_test, y_pred)
        
        logger.info(f"测试集准确率: {accuracy:.4f}")
        logger.info(f"分类报告:\n{classification_report(y_test, y_pred)}")
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
    
    def save_model(self, model, model_name, scaler=None, label_encoder=None):
        """保存模型"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存模型
        model_path = os.path.join(self.models_dir, f"{model_name}_{timestamp}.pkl")
        joblib.dump(model, model_path)
        
        # 保存预处理器
        if scaler:
            scaler_path = os.path.join(self.models_dir, f"scaler_{timestamp}.pkl")
            joblib.dump(scaler, scaler_path)
        
        if label_encoder:
            encoder_path = os.path.join(self.models_dir, f"label_encoder_{timestamp}.pkl")
            joblib.dump(label_encoder, encoder_path)
        
        logger.info(f"模型已保存: {model_path}")
        return model_path
    
    def plot_results(self, results, save_path=None):
        """绘制训练结果"""
        # 准备数据
        model_names = list(results.keys())
        scores = [results[name]['score'] for name in model_names]
        
        # 创建图表
        plt.figure(figsize=(12, 8))
        
        # 准确率对比
        plt.subplot(2, 2, 1)
        bars = plt.bar(model_names, scores)
        plt.title('模型准确率对比')
        plt.ylabel('准确率')
        plt.xticks(rotation=45)
        
        # 为柱状图添加数值标签
        for bar, score in zip(bars, scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom')
        
        # 模型参数对比
        plt.subplot(2, 2, 2)
        param_counts = [len(results[name]['params']) for name in model_names]
        plt.bar(model_names, param_counts)
        plt.title('模型参数数量')
        plt.ylabel('参数数量')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"结果图表已保存: {save_path}")
        
        plt.show()
    
    def load_model(self, model_path):
        """加载模型"""
        model = joblib.load(model_path)
        logger.info(f"模型已加载: {model_path}")
        return model

def main():
    """主函数 - 用于测试模型训练"""
    trainer = IntrusionDetectionTrainer()
    trainer.define_models()
    
    print("模型训练模块已创建")
    print("请先运行数据处理模块，然后使用以下代码训练模型:")
    print("""
    # 示例用法
    from data_processor import BNaTDataProcessor
    
    # 加载和处理数据
    processor = BNaTDataProcessor()
    data = processor.load_data('path_to_BNaT_dataset.csv')
    data = processor.preprocess_data(data)
    X, features = processor.extract_features(data)
    y = processor.prepare_labels(data)
    X_train, X_val, X_test, y_train, y_val, y_test = processor.split_data(X, y)
    
    # 训练模型
    trainer = IntrusionDetectionTrainer()
    trainer.define_models()
    results = trainer.train_all_models(X_train, y_train, X_val, y_val)
    
    # 评估最佳模型
    best_model_name = max(results.keys(), key=lambda k: results[k]['score'])
    best_model = results[best_model_name]['model']
    evaluation = trainer.evaluate_model(best_model, X_test, y_test, best_model_name)
    
    # 保存模型
    trainer.save_model(best_model, best_model_name, processor.scaler, processor.label_encoder)
    """)

if __name__ == "__main__":
    main() 