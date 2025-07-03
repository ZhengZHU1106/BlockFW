"""
BNaT数据预处理器
将原始21个特征转换为模型需要的28个特征
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Union, Tuple
import logging

logger = logging.getLogger(__name__)

class BNaTPreprocessor:
    """BNaT数据预处理器，处理原始21个特征到28个one-hot编码特征的转换"""
    
    def __init__(self):
        # BNaT数据集的特征名称
        self.feature_names = [
            "duration", "protocol_type", "service", "src_bytes", "dst_bytes", "flag",
            "count", "srv_count", "serror_rate", "same_srv_rate", "diff_srv_rate",
            "srv_serror_rate", "srv_diff_host_rate", "dst_host_count",
            "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
            "dst_host_same_src_port_rate", "dst_host_serror_rate",
            "dst_host_srv_diff_host_rate", "dst_host_srv_serror_rate"
        ]
        
        # 分类特征索引
        self.categorical_indices = [1, 2, 5]  # protocol_type, service, flag
        
        # 数值特征索引（排除分类特征）
        self.numerical_indices = [i for i in range(21) if i not in self.categorical_indices]
        
        # 根据实际训练配置，28维 = 18个数值特征 + 10个分类特征
        # 分类特征必须恰好编码为10维
        self.encoding_map = {
            'protocol_type': ['tcp', 'udp'],      # 2个类别 -> 1个特征 (drop_first)
            'service': ['http', 'other', 'ssh'],  # 3个类别 -> 2个特征 (drop_first)  
            'flag': ['OTH', 'SF', 'S0', 'S1', 'S2', 'S3', 'REJ', 'RSTO']  # 8个类别 -> 7个特征 (drop_first)
        }
        # 总计: 1 + 2 + 7 = 10个分类特征
        
        # 编码后的特征名（用于调试）
        self.encoded_feature_names = self._generate_encoded_feature_names()
        
    def _generate_encoded_feature_names(self) -> List[str]:
        """生成one-hot编码后的特征名称"""
        names = []
        
        # 添加数值特征名（按照scaler的顺序）
        numerical_feature_names = [
            "duration", "src_bytes", "dst_bytes", "count", "srv_count", 
            "serror_rate", "same_srv_rate", "diff_srv_rate", "srv_serror_rate", 
            "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count", 
            "dst_host_same_srv_rate", "dst_host_diff_srv_rate", 
            "dst_host_same_src_port_rate", "dst_host_serror_rate", 
            "dst_host_srv_diff_host_rate", "dst_host_srv_serror_rate"
        ]
        names.extend(numerical_feature_names)
        
        # 添加one-hot编码的特征名 (drop_first=True)
        for feat_name, categories in self.encoding_map.items():
            for cat in categories[1:]:  # 跳过第一个类别
                names.append(f"{feat_name}_{cat}")
        
        return names
    
    def preprocess(self, features: Union[List, np.ndarray, Dict]) -> np.ndarray:
        """
        预处理BNaT特征
        
        Args:
            features: 21个原始特征，可以是列表、数组或字典
            
        Returns:
            28个one-hot编码后的特征数组
        """
        # 转换输入格式
        if isinstance(features, dict):
            # 如果是字典，按照特征名顺序提取值
            feature_values = [features.get(name, 0) for name in self.feature_names]
        else:
            feature_values = list(features)
        
        # 验证特征数量
        if len(feature_values) != 21:
            raise ValueError(f"需要21个特征，但收到{len(feature_values)}个")
        
        # 准备编码后的特征
        encoded_features = []
        
        # 1. 添加18个数值特征（按照scaler训练时的顺序）
        # 移除分类特征，只保留数值特征
        numerical_values = []
        for i, value in enumerate(feature_values):
            if i not in self.categorical_indices:  # 跳过分类特征
                numerical_values.append(float(value))
        
        # 确保我们有18个数值特征
        if len(numerical_values) != 18:
            raise ValueError(f"预期18个数值特征，但得到{len(numerical_values)}个")
        
        encoded_features.extend(numerical_values)
        
        # 2. One-hot编码分类特征（总共10维）
        # protocol_type (index 1) -> 1维
        protocol = str(feature_values[1]).lower()
        if protocol in self.encoding_map['protocol_type']:
            for prot in self.encoding_map['protocol_type'][1:]:  # drop_first
                encoded_features.append(1.0 if protocol == prot else 0.0)
        else:
            # 未知协议，所有位都是0（相当于第一个类别）
            for prot in self.encoding_map['protocol_type'][1:]:
                encoded_features.append(0.0)
        
        # service (index 2) -> 2维
        service = str(feature_values[2]).lower()
        if service in self.encoding_map['service']:
            for srv in self.encoding_map['service'][1:]:  # drop_first
                encoded_features.append(1.0 if service == srv else 0.0)
        else:
            # 未知服务，映射到'other'
            for srv in self.encoding_map['service'][1:]:
                encoded_features.append(1.0 if srv == 'other' else 0.0)
        
        # flag (index 5) -> 7维
        flag = str(feature_values[5]).upper()
        if flag in self.encoding_map['flag']:
            for flg in self.encoding_map['flag'][1:]:  # drop_first
                encoded_features.append(1.0 if flag == flg else 0.0)
        else:
            # 未知标志，所有位都是0（相当于第一个类别）
            for flg in self.encoding_map['flag'][1:]:
                encoded_features.append(0.0)
        
        result = np.array(encoded_features)
        
        # 验证最终维度
        if len(result) != 28:
            raise ValueError(f"预期28维特征，但生成了{len(result)}维")
        
        return result
    
    def preprocess_with_names(self, features: Dict[str, Union[str, float]]) -> np.ndarray:
        """
        使用特征名称的预处理方法
        
        Args:
            features: 特征名到特征值的字典
            
        Returns:
            28个one-hot编码后的特征数组
        """
        # 转换为列表格式
        feature_list = [features.get(name, 0) for name in self.feature_names]
        return self.preprocess(feature_list)
    
    def get_feature_info(self) -> Dict[str, any]:
        """获取特征信息"""
        return {
            "original_features": 21,
            "encoded_features": 28,
            "numerical_features": 18,
            "categorical_features": 10,
            "feature_names": self.feature_names,
            "encoded_feature_names": self.encoded_feature_names,
            "categorical_mapping": {
                "protocol_type": f"{self.encoding_map['protocol_type']} -> {len(self.encoding_map['protocol_type'])-1}维",
                "service": f"{self.encoding_map['service']} -> {len(self.encoding_map['service'])-1}维",
                "flag": f"{self.encoding_map['flag']} -> {len(self.encoding_map['flag'])-1}维"
            }
        }


# 示例用法和测试
if __name__ == "__main__":
    preprocessor = BNaTPreprocessor()
    
    # 测试数据
    test_features = [
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
    
    try:
        encoded = preprocessor.preprocess(test_features)
        print(f"✅ 预处理成功")
        print(f"   原始特征数: 21")
        print(f"   编码后特征数: {len(encoded)}")
        print(f"   前18个数值特征: {encoded[:18]}")
        print(f"   后10个分类特征: {encoded[18:]}")
        print(f"   特征分布: 18数值 + 10分类 = 28总计")
    except Exception as e:
        print(f"❌ 预处理失败: {e}")