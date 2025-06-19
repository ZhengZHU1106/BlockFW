"""
BNaT 数据集专用处理器
专门处理 BNaT (Blockchain Network Attack Traffic) 数据集
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import os
import logging
from typing import Tuple, Dict, List

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BNaTSpecificProcessor:
    """BNaT 数据集专用处理器"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        # BNaT 数据集的特征列名（根据实际数据调整）
        self.feature_columns = [
            'duration', 'protocol_type', 'service', 'src_bytes', 'dst_host_srv_count',
            'flag', 'src_bytes_1', 'dst_bytes_1', 'land', 'wrong_fragment',
            'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised',
            'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
            'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
            'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
            'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
            'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
            'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
            'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
            'dst_host_rerror_rate', 'dst_host_srv_rerror_rate'
        ]
        
    def load_bnat_data(self, file_path: str) -> pd.DataFrame:
        """
        加载 BNaT 数据集
        
        Args:
            file_path: 数据文件路径
            
        Returns:
            加载的数据框
        """
        try:
            logger.info(f"正在加载 BNaT 数据: {file_path}")
            
            # 读取 CSV 文件，假设没有列名
            data = pd.read_csv(file_path, header=None)
            
            # 根据 BNaT 数据集的实际结构设置列名
            if len(data.columns) == 22:  # 21个特征 + 1个标签
                # 设置特征列名
                feature_cols = [f'feature_{i}' for i in range(21)]
                label_col = 'label'
                
                data.columns = feature_cols + [label_col]
                
                logger.info(f"数据加载成功，形状: {data.shape}")
                logger.info(f"特征数量: {len(feature_cols)}")
                logger.info(f"标签列: {label_col}")
                
                return data
            else:
                raise ValueError(f"数据列数不匹配，期望22列，实际{len(data.columns)}列")
                
        except Exception as e:
            logger.error(f"加载数据失败: {str(e)}")
            raise
    
    def merge_datasets(self, file_paths: List[str]) -> pd.DataFrame:
        """
        合并多个 BNaT 数据集文件
        
        Args:
            file_paths: 数据文件路径列表
            
        Returns:
            合并后的数据框
        """
        logger.info(f"开始合并 {len(file_paths)} 个数据集")
        
        merged_data = []
        for i, file_path in enumerate(file_paths):
            logger.info(f"处理文件 {i+1}/{len(file_paths)}: {file_path}")
            data = self.load_bnat_data(file_path)
            merged_data.append(data)
        
        # 合并所有数据
        final_data = pd.concat(merged_data, ignore_index=True)
        
        logger.info(f"数据合并完成，总记录数: {len(final_data)}")
        
        return final_data
    
    def analyze_labels(self, data: pd.DataFrame) -> Dict:
        """
        分析标签分布
        
        Args:
            data: 数据框
            
        Returns:
            标签分析结果
        """
        label_counts = data['label'].value_counts()
        label_percentages = data['label'].value_counts(normalize=True) * 100
        
        analysis = {
            'total_samples': len(data),
            'unique_labels': len(label_counts),
            'label_counts': label_counts.to_dict(),
            'label_percentages': label_percentages.to_dict(),
            'class_imbalance': label_counts.max() / label_counts.min()
        }
        
        logger.info("标签分析结果:")
        logger.info(f"  总样本数: {analysis['total_samples']}")
        logger.info(f"  唯一标签数: {analysis['unique_labels']}")
        logger.info(f"  类别不平衡比例: {analysis['class_imbalance']:.2f}")
        
        for label, count in label_counts.items():
            percentage = label_percentages[label]
            logger.info(f"  {label}: {count} ({percentage:.1f}%)")
        
        return analysis
    
    def preprocess_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        预处理特征
        
        Args:
            data: 原始数据框
            
        Returns:
            预处理后的数据框
        """
        logger.info("开始特征预处理...")
        
        # 分离特征和标签
        features = data.drop('label', axis=1)
        labels = data['label']
        
        # 处理分类特征（如果有的话）
        categorical_features = []
        numerical_features = []
        
        for col in features.columns:
            if features[col].dtype == 'object':
                categorical_features.append(col)
            else:
                numerical_features.append(col)
        
        logger.info(f"分类特征: {len(categorical_features)} 个")
        logger.info(f"数值特征: {len(numerical_features)} 个")
        
        # 对分类特征进行编码
        if categorical_features:
            logger.info("编码分类特征...")
            for col in categorical_features:
                features[col] = pd.Categorical(features[col]).codes
        
        # 处理缺失值
        missing_values = features.isnull().sum()
        if missing_values.sum() > 0:
            logger.warning(f"发现缺失值: {missing_values[missing_values > 0]}")
            # 用中位数填充数值特征
            features = features.fillna(features.median())
        
        # 重新组合特征和标签
        processed_data = pd.concat([features, labels], axis=1)
        
        logger.info("特征预处理完成")
        return processed_data
    
    def prepare_data(self, data: pd.DataFrame, test_size: float = 0.2, val_size: float = 0.2) -> Tuple:
        """
        准备训练、验证和测试数据
        
        Args:
            data: 预处理后的数据框
            test_size: 测试集比例
            val_size: 验证集比例
            
        Returns:
            训练、验证、测试数据元组
        """
        logger.info("准备训练、验证和测试数据...")
        
        # 分离特征和标签
        X = data.drop('label', axis=1)
        y = data['label']
        
        # 编码标签
        y_encoded = self.label_encoder.fit_transform(y)
        
        logger.info(f"标签编码完成，类别: {list(self.label_encoder.classes_)}")
        
        # 分割数据集
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
        )
        
        # 从剩余数据中分割出验证集
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size, random_state=42, stratify=y_temp
        )
        
        # 标准化特征
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        logger.info("数据集准备完成:")
        logger.info(f"  训练集: {X_train_scaled.shape[0]} 样本")
        logger.info(f"  验证集: {X_val_scaled.shape[0]} 样本")
        logger.info(f"  测试集: {X_test_scaled.shape[0]} 样本")
        logger.info(f"  特征数量: {X_train_scaled.shape[1]}")
        
        return (X_train_scaled, X_val_scaled, X_test_scaled, 
                y_train, y_val, y_test, X.columns.tolist())
    
    def save_processed_data(self, data_tuple: Tuple, output_path: str):
        """
        保存处理后的数据
        
        Args:
            data_tuple: 处理后的数据元组
            output_path: 输出文件路径
        """
        import pickle
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        data_dict = {
            'X_train': data_tuple[0],
            'X_val': data_tuple[1],
            'X_test': data_tuple[2],
            'y_train': data_tuple[3],
            'y_val': data_tuple[4],
            'y_test': data_tuple[5],
            'feature_names': data_tuple[6],
            'scaler': self.scaler,
            'label_encoder': self.label_encoder
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(data_dict, f)
        
        logger.info(f"处理后的数据已保存: {output_path}")

def main():
    """主函数 - 演示如何使用 BNaT 处理器"""
    processor = BNaTSpecificProcessor()
    
    # 示例：处理 Merged_dataset
    print("=== 处理 Merged_dataset ===")
    
    # 合并三个工作节点的数据
    merged_files = [
        'data/BNaT-master/Merged_dataset/w1.csv',
        'data/BNaT-master/Merged_dataset/w2.csv',
        'data/BNaT-master/Merged_dataset/w3.csv'
    ]
    
    # 检查文件是否存在
    existing_files = [f for f in merged_files if os.path.exists(f)]
    
    if existing_files:
        print(f"找到 {len(existing_files)} 个数据文件")
        
        # 加载和合并数据
        merged_data = processor.merge_datasets(existing_files)
        
        # 分析标签分布
        analysis = processor.analyze_labels(merged_data)
        
        # 预处理特征
        processed_data = processor.preprocess_features(merged_data)
        
        # 准备训练数据
        data_tuple = processor.prepare_data(processed_data)
        
        # 保存处理后的数据
        processor.save_processed_data(data_tuple, 'models/bnat_merged_processed.pkl')
        
        print("Merged_dataset 处理完成！")
    else:
        print("未找到 Merged_dataset 文件")
    
    # 示例：处理 DL 数据集
    print("\n=== 处理 DL 数据集 ===")
    
    dl_train_file = 'data/BNaT-master/Worker_1_+_2/DL/DL_2_train.csv'
    dl_test_file = 'data/BNaT-master/Worker_1_+_2/DL/DL_2_test.csv'
    
    if os.path.exists(dl_train_file) and os.path.exists(dl_test_file):
        print("找到 DL 数据集文件")
        
        # 加载训练数据
        train_data = processor.load_bnat_data(dl_train_file)
        test_data = processor.load_bnat_data(dl_test_file)
        
        # 分析标签分布
        print("训练集标签分布:")
        train_analysis = processor.analyze_labels(train_data)
        
        print("测试集标签分布:")
        test_analysis = processor.analyze_labels(test_data)
        
        # 预处理训练数据
        processed_train = processor.preprocess_features(train_data)
        processed_test = processor.preprocess_features(test_data)
        
        # 准备数据（使用预定义的训练/测试分割）
        X_train = processed_train.drop('label', axis=1)
        y_train = processed_train['label']
        X_test = processed_test.drop('label', axis=1)
        y_test = processed_test['label']
        
        # 编码标签
        y_train_encoded = processor.label_encoder.fit_transform(y_train)
        y_test_encoded = processor.label_encoder.transform(y_test)
        
        # 标准化特征
        X_train_scaled = processor.scaler.fit_transform(X_train)
        X_test_scaled = processor.scaler.transform(X_test)
        
        # 保存处理后的数据
        dl_data_tuple = (X_train_scaled, None, X_test_scaled, 
                        y_train_encoded, None, y_test_encoded, X_train.columns.tolist())
        processor.save_processed_data(dl_data_tuple, 'models/bnat_dl_processed.pkl')
        
        print("DL 数据集处理完成！")
    else:
        print("未找到 DL 数据集文件")

if __name__ == "__main__":
    main() 