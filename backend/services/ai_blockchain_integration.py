"""
AI检测与区块链集成服务
实现AI检测结果自动触发智能合约操作
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import hashlib
import json

from services.ai_detection_service_v2 import ai_detection_manager
from services.blockchain_service import blockchain_service

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AIBlockchainIntegration:
    """AI检测与区块链集成服务类"""
    
    def __init__(self):
        # 自动联动配置
        self.auto_block_enabled = True
        self.auto_pattern_recording = True
        self.attack_threshold = 0.8  # 置信度阈值
        self.attack_count_threshold = 3  # 同一IP攻击次数阈值
        
        # 攻击统计
        self.attack_history = {}  # IP -> 攻击记录
        
        # 联动统计
        self.integration_stats = {
            "total_detections": 0,
            "blockchain_triggers": 0,
            "auto_blocks": 0,
            "pattern_records": 0,
            "errors": 0
        }
        
        logger.info("✅ AI-区块链集成服务初始化完成")
    
    def enhanced_detect(self, features: List[float], source_ip: str = None, 
                       destination_port: int = None, auto_action: bool = True) -> Dict[str, Any]:
        """
        增强的检测功能，集成区块链自动联动
        
        Args:
            features: 21维BNaT特征
            source_ip: 源IP地址
            destination_port: 目标端口
            auto_action: 是否启用自动联动
            
        Returns:
            包含检测结果和区块链操作结果的字典
        """
        result = {
            "detection": None,
            "blockchain_actions": [],
            "auto_actions_taken": False,
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            # 1. 执行AI检测
            detection_result = ai_detection_manager.detect(features)
            result["detection"] = detection_result
            self.integration_stats["total_detections"] += 1
            
            if not detection_result.get("success", False):
                return result
            
            # 2. 检查是否为攻击
            is_attack = detection_result.get("is_attack", False)
            confidence = detection_result.get("confidence", 0.0)
            predicted_label = detection_result.get("predicted_label", "Unknown")
            
            if is_attack and confidence >= self.attack_threshold and auto_action:
                logger.info(f"🚨 检测到攻击: {predicted_label}, 置信度: {confidence:.2%}")
                
                # 3. 记录攻击历史
                if source_ip:
                    self._record_attack_history(source_ip, predicted_label, confidence, destination_port)
                
                # 4. 执行区块链自动联动
                blockchain_actions = self._trigger_blockchain_actions(
                    predicted_label, confidence, source_ip, destination_port, detection_result
                )
                result["blockchain_actions"] = blockchain_actions
                
                if blockchain_actions:
                    result["auto_actions_taken"] = True
                    self.integration_stats["blockchain_triggers"] += 1
            
            return result
            
        except Exception as e:
            logger.error(f"增强检测失败: {e}")
            self.integration_stats["errors"] += 1
            result["error"] = str(e)
            return result
    
    def _record_attack_history(self, source_ip: str, attack_type: str, 
                              confidence: float, destination_port: int = None):
        """记录攻击历史"""
        if source_ip not in self.attack_history:
            self.attack_history[source_ip] = []
        
        attack_record = {
            "timestamp": datetime.now().isoformat(),
            "attack_type": attack_type,
            "confidence": confidence,
            "destination_port": destination_port
        }
        
        self.attack_history[source_ip].append(attack_record)
        
        # 保持历史记录在合理范围内（每个IP最多保留50条记录）
        if len(self.attack_history[source_ip]) > 50:
            self.attack_history[source_ip] = self.attack_history[source_ip][-50:]
    
    def _trigger_blockchain_actions(self, attack_type: str, confidence: float,
                                   source_ip: str = None, destination_port: int = None,
                                   detection_result: Dict = None) -> List[Dict[str, Any]]:
        """触发区块链自动联动操作"""
        blockchain_actions = []
        
        try:
            # 1. 记录攻击模式到区块链
            if self.auto_pattern_recording and blockchain_service.contract:
                pattern_action = self._record_attack_pattern(attack_type, confidence, detection_result)
                if pattern_action:
                    blockchain_actions.append(pattern_action)
            
            # 2. 自动封锁端口
            if (self.auto_block_enabled and destination_port and 
                blockchain_service.contract and self._should_auto_block(source_ip, destination_port)):
                
                block_action = self._auto_block_port(destination_port, attack_type, source_ip)
                if block_action:
                    blockchain_actions.append(block_action)
            
        except Exception as e:
            logger.error(f"区块链联动操作失败: {e}")
            blockchain_actions.append({
                "action": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
        
        return blockchain_actions
    
    def _record_attack_pattern(self, attack_type: str, confidence: float, 
                              detection_result: Dict) -> Optional[Dict[str, Any]]:
        """记录攻击模式到区块链"""
        try:
            # 生成攻击模式哈希
            pattern_data = {
                "attack_type": attack_type,
                "confidence": confidence,
                "timestamp": datetime.now().isoformat()
            }
            pattern_json = json.dumps(pattern_data, sort_keys=True)
            pattern_hash = int(hashlib.sha256(pattern_json.encode()).hexdigest()[:8], 16)
            
            # 记录到区块链
            result = blockchain_service.add_attack_pattern(
                pattern_hash, 
                f"AI检测: {attack_type} (置信度: {confidence:.2%})"
            )
            
            if result.get("success", False):
                self.integration_stats["pattern_records"] += 1
                logger.info(f"✅ 攻击模式已记录到区块链: {pattern_hash}")
                
                return {
                    "action": "record_attack_pattern",
                    "success": True,
                    "pattern_hash": pattern_hash,
                    "attack_type": attack_type,
                    "confidence": confidence,
                    "transaction_hash": result.get("transaction_hash"),
                    "timestamp": result.get("timestamp")
                }
            else:
                logger.error(f"❌ 攻击模式记录失败: {result.get('error')}")
                return {
                    "action": "record_attack_pattern",
                    "success": False,
                    "error": result.get("error"),
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"记录攻击模式失败: {e}")
            return {
                "action": "record_attack_pattern",
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _should_auto_block(self, source_ip: str, destination_port: int) -> bool:
        """判断是否应该自动封锁端口"""
        if not source_ip:
            return False
        
        # 检查该IP的攻击次数
        if source_ip in self.attack_history:
            attack_count = len(self.attack_history[source_ip])
            if attack_count >= self.attack_count_threshold:
                logger.info(f"🔒 IP {source_ip} 攻击次数达到阈值 ({attack_count}), 触发自动封锁")
                return True
        
        # 检查端口是否为高危端口
        high_risk_ports = [22, 23, 3389, 1433, 3306, 5432]  # SSH, Telnet, RDP, SQL Server, MySQL, PostgreSQL
        if destination_port in high_risk_ports:
            logger.info(f"🔒 端口 {destination_port} 为高危端口，触发自动封锁")
            return True
        
        return False
    
    def _auto_block_port(self, port: int, attack_type: str, source_ip: str = None) -> Optional[Dict[str, Any]]:
        """自动封锁端口"""
        try:
            # 检查端口是否已被封锁
            blocked_ports = blockchain_service.get_blocked_ports()
            if port in blocked_ports:
                return {
                    "action": "auto_block_port",
                    "success": False,
                    "error": f"端口 {port} 已被封锁",
                    "port": port,
                    "timestamp": datetime.now().isoformat()
                }
            
            # 执行自动封锁
            result = blockchain_service.auto_block_port(port)
            
            if result.get("success", False):
                self.integration_stats["auto_blocks"] += 1
                logger.info(f"🔒 端口 {port} 已自动封锁 (攻击类型: {attack_type})")
                
                return {
                    "action": "auto_block_port",
                    "success": True,
                    "port": port,
                    "attack_type": attack_type,
                    "source_ip": source_ip,
                    "transaction_hash": result.get("transaction_hash"),
                    "timestamp": result.get("timestamp")
                }
            else:
                logger.error(f"❌ 端口 {port} 自动封锁失败: {result.get('error')}")
                return {
                    "action": "auto_block_port",
                    "success": False,
                    "error": result.get("error"),
                    "port": port,
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"自动封锁端口失败: {e}")
            return {
                "action": "auto_block_port",
                "success": False,
                "error": str(e),
                "port": port,
                "timestamp": datetime.now().isoformat()
            }
    
    def get_integration_statistics(self) -> Dict[str, Any]:
        """获取集成服务统计信息"""
        return {
            "integration_stats": self.integration_stats.copy(),
            "attack_history_count": len(self.attack_history),
            "config": {
                "auto_block_enabled": self.auto_block_enabled,
                "auto_pattern_recording": self.auto_pattern_recording,
                "attack_threshold": self.attack_threshold,
                "attack_count_threshold": self.attack_count_threshold
            },
            "blockchain_status": {
                "connected": blockchain_service.w3 is not None and blockchain_service.w3.is_connected(),
                "contract_loaded": blockchain_service.contract is not None
            },
            "timestamp": datetime.now().isoformat()
        }
    
    def get_attack_history(self, source_ip: str = None, limit: int = 50) -> Dict[str, Any]:
        """获取攻击历史记录"""
        if source_ip:
            # 返回特定IP的攻击历史
            history = self.attack_history.get(source_ip, [])
            return {
                "source_ip": source_ip,
                "attack_count": len(history),
                "attacks": history[-limit:] if history else []
            }
        else:
            # 返回所有IP的攻击摘要
            summary = {}
            for ip, attacks in self.attack_history.items():
                summary[ip] = {
                    "attack_count": len(attacks),
                    "latest_attack": attacks[-1] if attacks else None,
                    "attack_types": list(set(attack["attack_type"] for attack in attacks))
                }
            
            return {
                "total_ips": len(summary),
                "attack_summary": summary
            }
    
    def configure_auto_actions(self, auto_block: bool = None, auto_pattern: bool = None,
                              attack_threshold: float = None, count_threshold: int = None) -> Dict[str, Any]:
        """配置自动联动参数"""
        if auto_block is not None:
            self.auto_block_enabled = auto_block
        
        if auto_pattern is not None:
            self.auto_pattern_recording = auto_pattern
        
        if attack_threshold is not None:
            if 0.0 <= attack_threshold <= 1.0:
                self.attack_threshold = attack_threshold
            else:
                raise ValueError("攻击阈值必须在0.0到1.0之间")
        
        if count_threshold is not None:
            if count_threshold > 0:
                self.attack_count_threshold = count_threshold
            else:
                raise ValueError("攻击次数阈值必须大于0")
        
        logger.info("⚙️ 自动联动配置已更新")
        
        return {
            "success": True,
            "message": "自动联动配置已更新",
            "current_config": {
                "auto_block_enabled": self.auto_block_enabled,
                "auto_pattern_recording": self.auto_pattern_recording,
                "attack_threshold": self.attack_threshold,
                "attack_count_threshold": self.attack_count_threshold
            }
        }
    
    def clear_attack_history(self, source_ip: str = None):
        """清除攻击历史记录"""
        if source_ip:
            if source_ip in self.attack_history:
                del self.attack_history[source_ip]
                logger.info(f"🗑️ 已清除IP {source_ip} 的攻击历史")
        else:
            self.attack_history.clear()
            logger.info("🗑️ 已清除所有攻击历史")


# 创建全局AI-区块链集成服务实例
ai_blockchain_integration = AIBlockchainIntegration()