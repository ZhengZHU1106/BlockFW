"""
API数据模型定义
用于请求和响应的数据结构
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any, Union
from datetime import datetime
from enum import Enum
from pydantic import validator

class DetectionStatus(str, Enum):
    """检测状态枚举"""
    NORMAL = "normal"
    SUSPICIOUS = "suspicious"
    ATTACK = "attack"
    SIMULATED = "simulated"

class BlockchainStatus(str, Enum):
    """区块链操作状态枚举"""
    SUCCESS = "success"
    PENDING = "pending"
    FAILED = "failed"

# === AI检测相关模型 ===

class DetectionRequest(BaseModel):
    """单次检测请求模型"""
    features: List[Union[str, float, int]] = Field(
        ...,
        min_items=21,
        max_items=21,
        description="21维BNaT网络流量特征，支持字符串和数值混合",
        example=[0, 'tcp', 'http', 408, 0, 'OTH', 14, 13, 0, 0.64, 0.36, 0, 0.31, 14, 13, 0.64, 0.36, 0.21, 0, 0.31, 0]
    )
    source_ip: Optional[str] = Field(None, description="源IP地址")
    destination_port: Optional[int] = Field(None, description="目标端口")
    protocol: Optional[str] = Field(None, description="协议类型")
    metadata: Optional[Dict[str, Any]] = Field(None, description="可选的元数据")

    @validator('features')
    def validate_features_length(cls, v):
        if len(v) != 21:
            raise ValueError('特征列表必须包含21个元素')
        return v

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
    features_list: List[List[Union[str, float, int]]] = Field(
        ...,
        description="多个21维BNaT特征向量的列表"
    )
    batch_id: Optional[str] = Field(None, description="批次ID")
    metadata: Optional[Dict[str, Any]] = Field(None, description="可选的元数据")

class BatchDetectionResult(BaseModel):
    """批量检测结果模型"""
    batch_id: str = Field(..., description="批次ID")
    total_samples: int = Field(..., description="总样本数")
    attack_count: int = Field(..., description="攻击数量")
    normal_count: int = Field(..., description="正常数量")
    results: List[DetectionResult] = Field(..., description="详细结果列表")
    summary: Dict[str, Any] = Field(..., description="汇总统计")
    success: bool

class EnhancedDetectionRequest(BaseModel):
    """增强检测请求模型"""
    features: List[Union[str, float, int]] = Field(
        ...,
        description="21维BNaT网络流量特征",
        example=[
            0, 'tcp', 'ssh', 5, 0, 'S0', 100, 100, 1.0, 0, 0, 
            1.0, 0, 255, 255, 1, 0, 0.05, 1, 0, 1
        ]
    )
    source_ip: Optional[str] = Field("192.168.1.100", description="源IP地址")
    destination_port: Optional[int] = Field(22, description="目标端口")
    auto_action: bool = Field(True, description="是否启用自动联动")

    @validator('features')
    def validate_features_length(cls, v):
        if len(v) != 21:
            raise ValueError('特征列表必须包含21个元素')
        return v

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
    """检测阈值设置请求模型"""
    threshold: int = Field(..., ge=0)

# === 通用响应模型 ===

class APIResponse(BaseModel):
    """通用API响应模型"""
    success: bool = True
    message: Optional[str] = None
    data: Optional[Any] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    error_code: Optional[int] = None

class ErrorResponse(BaseModel):
    """通用错误响应模型"""
    success: bool = False
    error: str
    details: Optional[Any] = None 