"""
入侵检测推理服务模块
提供实时检测和批量检测功能
"""

import numpy as np
import pandas as pd
import joblib
import os
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import json

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IntrusionDetectionService:
    """入侵检测推理服务"""
    
    def __init__(self, model_path: str, scaler_path: str = None, encoder_path: str = None):
        """
        初始化检测服务
        
        Args:
            model_path: 模型文件路径
            scaler_path: 标准化器文件路径
            encoder_path: 标签编码器文件路径
        """
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.encoder_path = encoder_path
        
        # 加载模型和预处理器
        self.model = None
        self.scaler = None
        self.label_encoder = None
        
        self.load_components()
        
    def load_components(self):
        """加载模型和预处理器"""
        try:
            # 加载模型
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
                logger.info(f"模型已加载: {self.model_path}")
            else:
                raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
            
            # 加载标准化器
            if self.scaler_path and os.path.exists(self.scaler_path):
                self.scaler = joblib.load(self.scaler_path)
                logger.info(f"标准化器已加载: {self.scaler_path}")
            
            # 加载标签编码器
            if self.encoder_path and os.path.exists(self.encoder_path):
                self.label_encoder = joblib.load(self.encoder_path)
                logger.info(f"标签编码器已加载: {self.encoder_path}")
                
        except Exception as e:
            logger.error(f"加载组件时出错: {str(e)}")
            raise
    
    def preprocess_input(self, data: np.ndarray) -> np.ndarray:
        """
        预处理输入数据
        
        Args:
            data: 输入数据
            
        Returns:
            预处理后的数据
        """
        if self.scaler is not None:
            data = self.scaler.transform(data)
        
        return data
    
    def detect_single(self, features: np.ndarray) -> Dict:
        """
        单次检测
        
        Args:
            features: 特征向量
            
        Returns:
            检测结果字典
        """
        try:
            # 预处理
            features_processed = self.preprocess_input(features.reshape(1, -1))
            
            # 预测
            prediction = self.model.predict(features_processed)[0]
            probabilities = self.model.predict_proba(features_processed)[0]
            
            # 解码标签
            if self.label_encoder is not None:
                predicted_label = self.label_encoder.inverse_transform([prediction])[0]
                all_labels = self.label_encoder.classes_
            else:
                predicted_label = str(prediction)
                all_labels = [str(i) for i in range(len(probabilities))]
            
            # 构建结果
            result = {
                'timestamp': datetime.now().isoformat(),
                'is_attack': bool(prediction),
                'predicted_label': predicted_label,
                'confidence': float(probabilities.max()),
                'probabilities': {
                    label: float(prob) for label, prob in zip(all_labels, probabilities)
                },
                'features_used': features.shape[0]
            }
            
            return result
            
        except Exception as e:
            logger.error(f"单次检测时出错: {str(e)}")
            raise
    
    def detect_batch(self, features_list: List[np.ndarray]) -> List[Dict]:
        """
        批量检测
        
        Args:
            features_list: 特征向量列表
            
        Returns:
            检测结果列表
        """
        try:
            results = []
            
            for i, features in enumerate(features_list):
                try:
                    result = self.detect_single(features)
                    result['sample_id'] = i
                    results.append(result)
                except Exception as e:
                    logger.warning(f"处理样本 {i} 时出错: {str(e)}")
                    results.append({
                        'sample_id': i,
                        'error': str(e),
                        'timestamp': datetime.now().isoformat()
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"批量检测时出错: {str(e)}")
            raise
    
    def detect_from_dataframe(self, df: pd.DataFrame) -> List[Dict]:
        """
        从DataFrame进行检测
        
        Args:
            df: 包含特征的DataFrame
            
        Returns:
            检测结果列表
        """
        try:
            # 转换为numpy数组
            features_array = df.values
            
            # 批量检测
            results = self.detect_batch([features_array[i] for i in range(features_array.shape[0])])
            
            return results
            
        except Exception as e:
            logger.error(f"从DataFrame检测时出错: {str(e)}")
            raise
    
    def get_detection_summary(self, results: List[Dict]) -> Dict:
        """
        获取检测结果摘要
        
        Args:
            results: 检测结果列表
            
        Returns:
            摘要统计
        """
        try:
            if not results:
                return {'error': '没有检测结果'}
            
            # 统计攻击数量
            attack_count = sum(1 for r in results if r.get('is_attack', False))
            total_count = len(results)
            
            # 统计标签分布
            label_counts = {}
            for result in results:
                if 'predicted_label' in result:
                    label = result['predicted_label']
                    label_counts[label] = label_counts.get(label, 0) + 1
            
            # 计算平均置信度
            confidences = [r.get('confidence', 0) for r in results if 'confidence' in r]
            avg_confidence = np.mean(confidences) if confidences else 0
            
            summary = {
                'total_samples': total_count,
                'attack_count': attack_count,
                'normal_count': total_count - attack_count,
                'attack_rate': attack_count / total_count if total_count > 0 else 0,
                'average_confidence': avg_confidence,
                'label_distribution': label_counts,
                'detection_time': datetime.now().isoformat()
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"生成摘要时出错: {str(e)}")
            raise
    
    def save_results(self, results: List[Dict], output_path: str):
        """
        保存检测结果
        
        Args:
            results: 检测结果列表
            output_path: 输出文件路径
        """
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            logger.info(f"检测结果已保存: {output_path}")
            
        except Exception as e:
            logger.error(f"保存结果时出错: {str(e)}")
            raise
    
    def get_model_info(self) -> Dict:
        """
        获取模型信息
        
        Returns:
            模型信息字典
        """
        try:
            info = {
                'model_path': self.model_path,
                'model_type': type(self.model).__name__,
                'scaler_loaded': self.scaler is not None,
                'encoder_loaded': self.label_encoder is not None,
                'available_classes': list(self.label_encoder.classes_) if self.label_encoder else None,
                'last_updated': datetime.fromtimestamp(os.path.getmtime(self.model_path)).isoformat()
            }
            
            return info
            
        except Exception as e:
            logger.error(f"获取模型信息时出错: {str(e)}")
            raise

class RealTimeDetector:
    """实时检测器"""
    
    def __init__(self, detection_service: IntrusionDetectionService, threshold: float = 0.5):
        """
        初始化实时检测器
        
        Args:
            detection_service: 检测服务实例
            threshold: 检测阈值
        """
        self.detection_service = detection_service
        self.threshold = threshold
        self.detection_history = []
        
    def process_stream(self, feature_stream: List[np.ndarray]) -> List[Dict]:
        """
        处理特征流
        
        Args:
            feature_stream: 特征流
            
        Returns:
            检测结果流
        """
        results = []
        
        for features in feature_stream:
            try:
                result = self.detection_service.detect_single(features)
                
                # 应用阈值
                if result['confidence'] >= self.threshold:
                    result['alert'] = True
                else:
                    result['alert'] = False
                
                # 添加到历史记录
                self.detection_history.append(result)
                
                # 保持历史记录在合理范围内
                if len(self.detection_history) > 1000:
                    self.detection_history = self.detection_history[-1000:]
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"处理流数据时出错: {str(e)}")
                continue
        
        return results
    
    def get_recent_alerts(self, minutes: int = 10) -> List[Dict]:
        """
        获取最近的告警
        
        Args:
            minutes: 时间范围（分钟）
            
        Returns:
            告警列表
        """
        try:
            cutoff_time = datetime.now().timestamp() - (minutes * 60)
            
            alerts = []
            for result in self.detection_history:
                if result.get('alert', False):
                    # 这里需要根据实际的时间戳格式进行调整
                    alerts.append(result)
            
            return alerts
            
        except Exception as e:
            logger.error(f"获取告警时出错: {str(e)}")
            return []

def main():
    """主函数 - 用于测试推理服务"""
    print("推理服务模块已创建")
    print("请先训练模型，然后使用以下代码进行检测:")
    print("""
    # 示例用法
    from inference_service import IntrusionDetectionService
    
    # 初始化检测服务
    service = IntrusionDetectionService(
        model_path='ml/models/best_model.pkl',
        scaler_path='ml/models/scaler.pkl',
        encoder_path='ml/models/label_encoder.pkl'
    )
    
    # 单次检测
    features = np.random.random(10)  # 示例特征
    result = service.detect_single(features)
    print(f"检测结果: {result}")
    
    # 批量检测
    features_list = [np.random.random(10) for _ in range(5)]
    results = service.detect_batch(features_list)
    
    # 获取摘要
    summary = service.get_detection_summary(results)
    print(f"检测摘要: {summary}")
    """)

if __name__ == "__main__":
    main() 