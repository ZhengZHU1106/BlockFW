"""
AI检测服务模块 - 支持21维BNaT特征
自动转换为28维特征进行检测
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
import logging
import joblib

# 导入BNaT预处理器
from services.bnat_preprocessor import BNaTPreprocessor

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ml_path = os.path.join(project_root, 'ml')
if ml_path not in sys.path:
    sys.path.insert(0, ml_path)

class BlockNetworkDetectionService:
    """
    基于实际训练的XGBoost模型的网络入侵检测服务
    支持21维BNaT特征输入，自动转换为28维特征
    """
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.model_loaded = False
        self.detection_history = []
        
        # 初始化BNaT预处理器
        try:
            self.preprocessor = BNaTPreprocessor()
            self.has_preprocessor = True
            logger.info("✅ BNaT预处理器初始化成功")
        except Exception as e:
            logger.error(f"❌ BNaT预处理器初始化失败: {e}")
            self.preprocessor = None
            self.has_preprocessor = False
        
        # 模型文件路径
        self.model_dir = os.path.join(project_root, 'ml', 'ml', 'models')
        self.model_files = {
            'model': 'xgboost_gpu_20250702_073122.pkl',
            'scaler': 'scaler_20250702_073122.pkl',
            'encoder': 'label_encoder_20250702_073122.pkl'
        }
        
        # 模型信息
        self.model_info = {
            'model_type': 'XGBoost GPU',
            'training_date': '2025-07-02',
            'input_features': 21,  # API接受的特征数
            'model_features': 28,  # 模型需要的特征数
            'features_count': 21,  # 向后兼容：用户输入的特征数
            'labels': ['BP', 'DoS', 'FoT', 'MitM', 'Normal'],
            'description': '基于BNaT数据集训练的区块链网络入侵检测模型（支持21维特征输入）'
        }
        
        # 特征名称（经过One-Hot编码后的28个特征）
        self.feature_names = [
            "duration", "src_bytes", "dst_bytes", 
            "count", "srv_count", "serror_rate", "same_srv_rate", "diff_srv_rate",
            "srv_serror_rate", "srv_diff_host_rate", "dst_host_count",
            "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
            "dst_host_same_src_port_rate", "dst_host_serror_rate",
            "dst_host_srv_diff_host_rate", "dst_host_srv_serror_rate",
            # One-Hot编码后的分类特征
            "protocol_type_tcp", "protocol_type_udp", 
            "service_dns", "service_ftp", "service_http", "service_ssh", "service_telnet",
            "flag_REJ", "flag_RSTO", "flag_S0"
        ]
        
        # 尝试加载模型
        self.load_model()
    
    def load_model(self) -> bool:
        """加载XGBoost模型及预处理器"""
        try:
            model_path = os.path.join(self.model_dir, self.model_files['model'])
            scaler_path = os.path.join(self.model_dir, self.model_files['scaler'])
            encoder_path = os.path.join(self.model_dir, self.model_files['encoder'])
            
            # 检查文件是否存在
            missing_files = []
            for name, path in [('XGBoost模型', model_path), ('标准化器', scaler_path), ('标签编码器', encoder_path)]:
                if not os.path.exists(path):
                    missing_files.append(f"{name}: {path}")
            
            if missing_files:
                logger.error("缺少必要的模型文件:")
                for missing in missing_files:
                    logger.error(f"  - {missing}")
                logger.info("使用模拟模式运行...")
                return False
            
            # 加载模型和预处理器
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            self.label_encoder = joblib.load(encoder_path)
            
            self.model_loaded = True
            logger.info("✅ XGBoost模型加载成功")
            logger.info(f"   - 模型类型: {self.model_info['model_type']}")
            logger.info(f"   - 训练日期: {self.model_info['training_date']}")
            logger.info(f"   - 输入特征数: {self.model_info['input_features']}")
            logger.info(f"   - 模型特征数: {self.model_info['model_features']}")
            logger.info(f"   - 标签类别: {self.model_info['labels']}")
            return True
            
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            logger.info("使用模拟模式运行...")
            self.model_loaded = False
            return False
    
    def detect(self, features: Union[List[float], np.ndarray]) -> Dict[str, Any]:
        """
        执行单次网络入侵检测
        
        Args:
            features: 21维BNaT特征向量
            
        Returns:
            检测结果字典
        """
        timestamp = datetime.now().isoformat()
        
        # 验证输入特征数量
        if len(features) != 21:
            return {
                'error': f'特征维度错误，需要21个BNaT特征，但收到{len(features)}个',
                'timestamp': timestamp,
                'success': False
            }
        
        # 检查预处理器是否可用
        if not self.has_preprocessor:
            return {
                'error': 'BNaT预处理器不可用',
                'timestamp': timestamp,
                'success': False
            }
        
        # 检查模型是否加载
        if not self.model_loaded:
            return self._simulate_detection(features, timestamp)
        
        try:
            # 预处理：将21维BNaT特征转换为28维
            feature_array_28d = self.preprocessor.preprocess(features)
            
            # 数据预处理：只标准化前18个数值特征
            numerical_features = feature_array_28d[:18]  # 前18个是数值特征
            categorical_features = feature_array_28d[18:]  # 后10个是One-Hot编码的分类特征
            
            # 标准化数值特征
            numerical_features_scaled = self.scaler.transform(numerical_features.reshape(1, -1))[0]
            
            # 重新组合特征
            final_features = np.concatenate([numerical_features_scaled, categorical_features])
            
            # 执行预测
            prediction = self.model.predict([final_features])[0]
            prediction_proba = self.model.predict_proba([final_features])[0]
            
            # 获取预测标签
            predicted_label = self.label_encoder.inverse_transform([prediction])[0]
            
            # 计算置信度（最高概率）
            confidence = float(np.max(prediction_proba))
            
            # 构建概率字典
            probabilities = {}
            for i, label in enumerate(self.label_encoder.classes_):
                probabilities[label] = float(prediction_proba[i])
            
            # 判断是否为攻击
            is_attack = predicted_label != 'Normal'
            
            result = {
                'timestamp': timestamp,
                'is_attack': is_attack,
                'predicted_label': predicted_label,
                'confidence': confidence,
                'probabilities': probabilities,
                'features_used': 21,  # 用户输入的特征数
                'success': True
            }
            
            # 添加到历史记录
            self.detection_history.append(result)
            
            # 保持历史记录不超过1000条
            if len(self.detection_history) > 1000:
                self.detection_history = self.detection_history[-1000:]
            
            return result
            
        except Exception as e:
            logger.error(f"检测失败: {e}")
            return {
                'error': str(e),
                'timestamp': timestamp,
                'success': False
            }
    
    def _simulate_detection(self, features: Union[List[float], np.ndarray], timestamp: str) -> Dict[str, Any]:
        """模拟检测（当模型未加载时使用）"""
        np.random.seed(int(datetime.now().timestamp()) % 1000)
        
        # 模拟预测结果
        labels = self.model_info['labels']
        predicted_label = np.random.choice(labels, p=[0.1, 0.1, 0.1, 0.1, 0.6])  # Normal概率更高
        confidence = float(np.random.uniform(0.7, 0.99))
        
        # 模拟概率分布
        probabilities = {}
        remaining_prob = 1.0 - confidence
        for label in labels:
            if label == predicted_label:
                probabilities[label] = confidence
            else:
                probabilities[label] = remaining_prob / (len(labels) - 1)
        
        is_attack = predicted_label != 'Normal'
        
        result = {
            'timestamp': timestamp,
            'is_attack': is_attack,
            'predicted_label': predicted_label,
            'confidence': confidence,
            'probabilities': probabilities,
            'features_used': 21,
            'success': True,
            'simulated': True  # 标记为模拟结果
        }
        
        # 添加到历史记录
        self.detection_history.append(result)
        
        return result
    
    def batch_detect(self, features_list: List[Union[List[float], np.ndarray]]) -> List[Dict[str, Any]]:
        """批量检测"""
        results = []
        for features in features_list:
            result = self.detect(features)
            results.append(result)
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取检测统计信息"""
        if not self.detection_history:
            return {
                'total_detections': 0,
                'attack_count': 0,
                'normal_count': 0,
                'attack_rate': 0.0,
                'attack_types': {},
                'model_status': 'loaded' if self.model_loaded else 'simulated',
                'preprocessor_status': 'available' if self.has_preprocessor else 'unavailable'
            }
        
        # 统计检测结果
        total_detections = len(self.detection_history)
        attack_count = sum(1 for r in self.detection_history if r.get('is_attack', False))
        normal_count = total_detections - attack_count
        attack_rate = attack_count / total_detections if total_detections > 0 else 0.0
        
        # 统计攻击类型
        attack_types = {}
        for result in self.detection_history:
            if result.get('is_attack', False):
                label = result.get('predicted_label', 'Unknown')
                attack_types[label] = attack_types.get(label, 0) + 1
        
        return {
            'total_detections': total_detections,
            'attack_count': attack_count,
            'normal_count': normal_count,
            'attack_rate': attack_rate,
            'attack_types': attack_types,
            'model_status': 'loaded' if self.model_loaded else 'simulated',
            'preprocessor_status': 'available' if self.has_preprocessor else 'unavailable'
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取AI模型信息"""
        info = {
            'loaded': self.model_loaded,
            'info': self.model_info.copy(),
            'preprocessor_available': self.has_preprocessor
        }
        
        if self.has_preprocessor:
            info['feature_info'] = self.preprocessor.get_feature_info()
        
        return info
    
    def clear_history(self):
        """清除检测历史"""
        self.detection_history = []
        logger.info("检测历史已清除")
    
    def get_recent_attacks(self, limit: int = 10) -> List[Dict[str, Any]]:
        """获取最近的攻击记录"""
        attacks = [r for r in self.detection_history if r.get('is_attack', False)]
        return attacks[-limit:] if attacks else []
    
    def generate_sample_features(self) -> List[float]:
        """生成示例BNaT特征"""
        # 生成21个随机特征作为示例
        features = []
        
        # 数值特征（18个）
        features.extend([
            np.random.uniform(0, 100),      # duration
            np.random.choice(['tcp', 'udp']),  # protocol_type
            np.random.choice(['http', 'ssh', 'other']),  # service
            np.random.randint(0, 10000),    # src_bytes
            np.random.randint(0, 10000),    # dst_bytes
            np.random.choice(['OTH', 'SF', 'S0']),  # flag
        ] + [np.random.uniform(0, 1) for _ in range(15)])  # 其他15个数值特征
        
        return features


# 创建全局检测服务实例
ai_detection_manager = BlockNetworkDetectionService()