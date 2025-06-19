"""
BNaT 数据集处理模块
用于加载、预处理和特征工程
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import os
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BNaTDataProcessor:
    """BNaT 数据集处理器"""
    
    def __init__(self, data_path="ml/data"):
        self.data_path = data_path
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def load_data(self, file_path):
        """加载数据集"""
        try:
            logger.info(f"正在加载数据: {file_path}")
            if file_path.endswith('.csv'):
                data = pd.read_csv(file_path)
            elif file_path.endswith('.json'):
                data = pd.read_json(file_path)
            else:
                raise ValueError("不支持的文件格式，请使用 CSV 或 JSON")
            
            logger.info(f"数据加载成功，形状: {data.shape}")
            return data
        except Exception as e:
            logger.error(f"数据加载失败: {str(e)}")
            raise
    
    def preprocess_data(self, data):
        """数据预处理"""
        logger.info("开始数据预处理...")
        
        # 检查缺失值
        missing_values = data.isnull().sum()
        if missing_values.sum() > 0:
            logger.warning(f"发现缺失值: {missing_values[missing_values > 0]}")
            # 对于数值列，使用中位数填充
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].median())
            
            # 对于分类列，使用众数填充
            categorical_columns = data.select_dtypes(include=['object']).columns
            for col in categorical_columns:
                data[col] = data[col].fillna(data[col].mode()[0])
        
        # 处理重复值
        duplicates = data.duplicated().sum()
        if duplicates > 0:
            logger.warning(f"发现 {duplicates} 个重复值，正在删除...")
            data = data.drop_duplicates()
        
        logger.info("数据预处理完成")
        return data
    
    def extract_features(self, data):
        """特征提取和工程"""
        logger.info("开始特征提取...")
        
        # 假设数据包含网络流量特征
        # 这里需要根据实际的 BNaT 数据集结构进行调整
        
        # 基本统计特征
        numeric_features = data.select_dtypes(include=[np.number]).columns.tolist()
        
        # 移除目标变量（如果存在）
        if 'label' in numeric_features:
            numeric_features.remove('label')
        if 'attack_type' in numeric_features:
            numeric_features.remove('attack_type')
        
        # 创建特征矩阵
        X = data[numeric_features]
        
        # 标准化特征
        X_scaled = self.scaler.fit_transform(X)
        
        logger.info(f"特征提取完成，特征数量: {X_scaled.shape[1]}")
        return X_scaled, numeric_features
    
    def prepare_labels(self, data):
        """准备标签数据"""
        logger.info("准备标签数据...")
        
        # 根据 BNaT 数据集的实际结构调整
        if 'label' in data.columns:
            y = data['label']
        elif 'attack_type' in data.columns:
            y = data['attack_type']
        else:
            raise ValueError("未找到标签列，请检查数据集结构")
        
        # 编码标签
        y_encoded = self.label_encoder.fit_transform(y)
        
        logger.info(f"标签准备完成，类别数量: {len(self.label_encoder.classes_)}")
        return y_encoded
    
    def split_data(self, X, y, test_size=0.2, val_size=0.2):
        """分割数据集"""
        logger.info("分割数据集...")
        
        # 首先分割出测试集
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # 从剩余数据中分割出验证集
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size, random_state=42, stratify=y_temp
        )
        
        logger.info(f"数据集分割完成:")
        logger.info(f"  训练集: {X_train.shape[0]} 样本")
        logger.info(f"  验证集: {X_val.shape[0]} 样本")
        logger.info(f"  测试集: {X_test.shape[0]} 样本")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def save_processed_data(self, data_dict, output_path):
        """保存处理后的数据"""
        import pickle
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'wb') as f:
            pickle.dump(data_dict, f)
        
        logger.info(f"处理后的数据已保存到: {output_path}")
    
    def load_processed_data(self, input_path):
        """加载处理后的数据"""
        import pickle
        
        with open(input_path, 'rb') as f:
            data_dict = pickle.load(f)
        
        logger.info(f"已加载处理后的数据: {input_path}")
        return data_dict

def main():
    """主函数 - 用于测试数据处理"""
    processor = BNaTDataProcessor()
    
    # 这里需要根据实际的 BNaT 数据集文件路径进行调整
    # data_file = "ml/data/BNaT_dataset.csv"  # 请替换为实际路径
    
    # 示例用法（需要实际数据文件）
    # data = processor.load_data(data_file)
    # data = processor.preprocess_data(data)
    # X, features = processor.extract_features(data)
    # y = processor.prepare_labels(data)
    # X_train, X_val, X_test, y_train, y_val, y_test = processor.split_data(X, y)
    
    print("数据处理模块已创建，请根据实际的 BNaT 数据集调整代码")

if __name__ == "__main__":
    main() 