"""
API数据模型定义
用于请求和响应的数据结构
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any, Union
from datetime import datetime
from enum import Enum

class DetectionStatus(str, Enum):
    """检测状态枚举"""
    NORMAL = "normal"
    SUSPICIOUS = "suspicious"
    ATTACK = "attack"

class BlockchainStatus(str, Enum):
    """区块链操作状态枚举"""
    SUCCESS = "success"
    PENDING = "pending"
    FAILED = "failed"

# === AI检测相关模型 ===

class DetectionRequest(BaseModel):
    """AI检测请求模型（21维BNaT特征）"""
    features: List[float] = Field(..., description="网络流量特征向量（21维BNaT特征）", min_items=21, max_items=21)
    source_ip: Optional[str] = Field(None, description="源IP地址")
    destination_port: Optional[int] = Field(None, description="目标端口")
    protocol: Optional[str] = Field(None, description="协议类型")
    metadata: Optional[Dict[str, Any]] = Field(None, description="额外元数据")

class DetectionResult(BaseModel):
    """AI检测结果模型"""
    timestamp: str = Field(..., description="检测时间戳")
    is_attack: bool = Field(..., description="是否为攻击")
    predicted_label: str = Field(..., description="预测标签")
    confidence: float = Field(..., description="置信度", ge=0.0, le=1.0)
    probabilities: Optional[Dict[str, float]] = Field(None, description="各类别概率")
    features_used: int = Field(..., description="使用的特征数量")
    source_ip: Optional[str] = None
    destination_port: Optional[int] = None
    status: Optional[DetectionStatus] = DetectionStatus.NORMAL
    metadata: Optional[Dict[str, Any]] = None

class BatchDetectionRequest(BaseModel):
    """批量检测请求模型"""
    features_list: List[List[float]] = Field(..., description="批量特征向量列表（每个包含21个特征）")
    batch_id: Optional[str] = Field(None, description="批次ID")

class BatchDetectionResult(BaseModel):
    """批量检测结果模型"""
    batch_id: str = Field(..., description="批次ID")
    total_samples: int = Field(..., description="总样本数")
    attack_count: int = Field(..., description="攻击数量")
    normal_count: int = Field(..., description="正常数量")
    results: List[DetectionResult] = Field(..., description="详细结果列表")
    summary: Dict[str, Any] = Field(..., description="汇总统计")

# === 区块链相关模型 ===

class BlockPortRequest(BaseModel):
    """端口封锁请求模型"""
    port: int = Field(..., description="要封锁的端口", ge=1, le=65535)
    reason: str = Field(..., description="封锁原因")
    auto_block: bool = Field(False, description="是否为AI自动封锁")

class BlockPortResult(BaseModel):
    """端口封锁结果模型"""
    transaction_hash: str = Field(..., description="交易哈希")
    port: int = Field(..., description="封锁的端口")
    status: BlockchainStatus = Field(..., description="操作状态")
    gas_used: Optional[int] = Field(None, description="使用的Gas")
    timestamp: datetime = Field(default_factory=datetime.now)

class AttackPatternRequest(BaseModel):
    """攻击模式请求模型"""
    pattern: int = Field(..., description="攻击模式编码")
    description: str = Field(..., description="攻击模式描述")
    severity: int = Field(..., description="严重程度", ge=1, le=10)

class AttackPatternResult(BaseModel):
    """攻击模式结果模型"""
    transaction_hash: str = Field(..., description="交易哈希")
    pattern: int = Field(..., description="攻击模式编码")
    status: BlockchainStatus = Field(..., description="操作状态")
    timestamp: datetime = Field(default_factory=datetime.now)

class VoteRequest(BaseModel):
    """多签名投票请求模型"""
    port: int = Field(..., description="要投票封锁的端口", ge=1, le=65535)
    voter_address: str = Field(..., description="投票人地址")

class VoteResult(BaseModel):
    """投票结果模型"""
    transaction_hash: str = Field(..., description="交易哈希")
    port: int = Field(..., description="投票的端口")
    voter: str = Field(..., description="投票人地址")
    current_votes: int = Field(..., description="当前票数")
    required_votes: int = Field(..., description="所需票数")
    is_blocked: bool = Field(..., description="是否已被封锁")
    status: BlockchainStatus = Field(..., description="操作状态")
    timestamp: datetime = Field(default_factory=datetime.now)

# === 系统状态相关模型 ===

class SystemStatus(BaseModel):
    """系统状态模型"""
    api_status: str = Field(..., description="API状态")
    ai_model_status: str = Field(..., description="AI模型状态")
    blockchain_status: str = Field(..., description="区块链状态")
    database_status: str = Field(..., description="数据库状态")
    last_detection_time: Optional[datetime] = Field(None, description="最后检测时间")
    total_detections: int = Field(0, description="总检测次数")
    total_blocks: int = Field(0, description="总封锁次数")

class ContractInfo(BaseModel):
    """智能合约信息模型"""
    address: str = Field(..., description="合约地址")
    network: str = Field(..., description="网络名称")
    signers: List[str] = Field(..., description="签名人列表")
    min_signatures: int = Field(..., description="最小签名数")
    detection_threshold: int = Field(..., description="检测阈值")
    blocked_ports_count: int = Field(..., description="已封锁端口数量")
    attack_patterns_count: int = Field(..., description="攻击模式数量")

# === 通用响应模型 ===

class APIResponse(BaseModel):
    """通用API响应模型"""
    success: bool = Field(..., description="操作是否成功")
    message: str = Field(..., description="响应消息")
    data: Optional[Any] = Field(None, description="响应数据")
    timestamp: datetime = Field(default_factory=datetime.now)
    error_code: Optional[str] = Field(None, description="错误代码")

class ErrorResponse(BaseModel):
    """错误响应模型"""
    success: bool = Field(False, description="操作失败")
    error: str = Field(..., description="错误信息")
    error_code: str = Field(..., description="错误代码")
    timestamp: datetime = Field(default_factory=datetime.now)
    details: Optional[Dict[str, Any]] = Field(None, description="错误详情") 