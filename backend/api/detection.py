"""
AI检测相关的API路由
"""

from fastapi import APIRouter, HTTPException, Request
from typing import List, Dict, Any
from datetime import datetime

from models.schemas import (
    DetectionRequest, 
    DetectionResult, 
    BatchDetectionRequest,
    APIResponse,
    ErrorResponse
)
from services.ai_detection_service_v2 import ai_detection_manager
from services.ai_blockchain_integration import ai_blockchain_integration

# 创建路由器
router = APIRouter(
    prefix="/api/detection",
    tags=["AI Detection"],
    responses={
        404: {"description": "Not found"},
        500: {"description": "Internal server error"}
    }
)

@router.post("/single", response_model=DetectionResult)
async def detect_single(request: Request):
    """
    执行单次AI检测
    
    - **features**: 21维BNaT特征（支持混合类型）
    - **metadata**: 可选的元数据
    """
    try:
        body = await request.json()
        features = body.get("features")
        metadata = body.get("metadata")

        if not features or not isinstance(features, list) or len(features) != 21:
            raise HTTPException(status_code=400, detail="特征列表必须是21个元素的有效列表")

        # 执行检测
        result = ai_detection_manager.detect(features)
        
        if not result.get('success', False):
            raise HTTPException(
                status_code=400,
                detail=result.get('error', '检测失败')
            )
        
        # 构建响应
        return DetectionResult(
            timestamp=result['timestamp'],
            is_attack=result['is_attack'],
            predicted_label=result['predicted_label'],
            confidence=result['confidence'],
            features_used=result.get('features_used', 21),
            metadata=metadata
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"检测服务错误: {str(e)}"
        )

@router.post("/batch", response_model=List[DetectionResult])
async def detect_batch(request: BatchDetectionRequest):
    """
    批量执行AI检测
    
    - **features_list**: 多个特征向量的列表
    """
    try:
        results = []
        for idx, features in enumerate(request.features_list):
            result = ai_detection_manager.detect(features)
            
            if result.get('success', False):
                results.append(DetectionResult(
                    timestamp=result['timestamp'],
                    is_attack=result['is_attack'],
                    predicted_label=result['predicted_label'],
                    confidence=result['confidence'],
                    features_used=result.get('features_used', 21),
                    metadata={"batch_index": idx}
                ))
        
        return results
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"批量检测错误: {str(e)}"
        )

@router.post("/enhanced", response_model=APIResponse)
async def enhanced_detection(request: Request):
    """
    增强的AI检测，集成区块链联动
    
    - **features**: 21维BNaT特征
    - **source_ip**: 源IP
    - **destination_port**: 目标端口
    """
    try:
        body = await request.json()
        features = body.get("features")
        source_ip = body.get("source_ip")
        destination_port = body.get("destination_port")

        if not features or not isinstance(features, list) or len(features) != 21:
            raise HTTPException(status_code=400, detail="特征列表必须是21个元素的有效列表")

        # 执行增强检测
        result = ai_blockchain_integration.enhanced_detect(
            features=features,
            source_ip=source_ip,
            destination_port=destination_port
        )
        
        return APIResponse(
            success=True,
            data=result,
            message="增强检测完成"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"增强检测服务错误: {str(e)}"
        )

@router.get("/statistics")
async def get_statistics():
    """获取检测统计信息"""
    try:
        stats = ai_detection_manager.get_statistics()
        return APIResponse(
            success=True,
            data=stats,
            message="统计信息获取成功"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"获取统计信息失败: {str(e)}"
        )

@router.get("/model-info")
async def get_model_info():
    """获取AI模型信息"""
    try:
        info = ai_detection_manager.get_model_info()
        return APIResponse(
            success=True,
            data=info,
            message="模型信息获取成功"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"获取模型信息失败: {str(e)}"
        )

@router.get("/recent-attacks")
async def get_recent_attacks(limit: int = 10):
    """获取最近的攻击记录"""
    try:
        attacks = ai_detection_manager.get_recent_attacks(limit)
        return APIResponse(
            success=True,
            data={
                "attacks": attacks,
                "count": len(attacks)
            },
            message="攻击记录获取成功"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"获取攻击记录失败: {str(e)}"
        )

@router.post("/clear-history")
async def clear_history():
    """清除检测历史"""
    try:
        ai_detection_manager.clear_history()
        return APIResponse(
            success=True,
            data=None,
            message="检测历史已清除"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"清除历史失败: {str(e)}"
        )

@router.post("/test")
async def test_detection():
    """测试检测功能（使用随机特征）"""
    try:
        import numpy as np
        # 生成21个随机特征（BNaT格式），模拟混合类型
        random_features = np.random.rand(18).tolist()  # 18 numerical
        random_features.insert(1, 'tcp')
        random_features.insert(2, 'http')
        random_features.insert(5, 'SF')

        result = ai_detection_manager.detect(random_features)
        
        if not result.get('success', False):
            raise HTTPException(
                status_code=400,
                detail=result.get('error', '测试检测失败')
            )
        
        return APIResponse(
            success=True,
            data=result,
            message="测试检测完成"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"测试错误: {str(e)}"
        )

@router.get("/feature-info")
async def get_feature_info():
    """获取BNaT特征信息和格式说明"""
    try:
        # 获取预处理器的特征信息
        if hasattr(ai_detection_manager, 'preprocessor') and ai_detection_manager.preprocessor:
            feature_info = ai_detection_manager.preprocessor.get_feature_info()
        else:
            feature_info = {
                'total_features': 21,
                'categorical_features': ['protocol_type', 'service', 'flag'],
                'note': '预处理器不可用，请使用数值特征'
            }
        
        # 添加使用示例
        feature_info['usage_example'] = {
            'normal_connection': [
                0.0, 'tcp', 'http', 181.0, 5450.0, 'SF', 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0
            ]
        }
        
        feature_info['feature_descriptions'] = [
            'duration: 连接持续时间', 'protocol_type: 协议类型 (tcp/udp/icmp)', 'service: 服务类型 (http/ftp/ssh等)',
            'src_bytes: 源字节数', 'dst_bytes: 目标字节数', 'flag: 连接标志 (SF/S0/REJ等)', 'land: 是否land攻击 (0/1)',
            'wrong_fragment: 错误片段数', 'urgent: 紧急数据包数', 'hot: 热点指标数', 'num_failed_logins: 失败登录次数',
            'logged_in: 是否成功登录 (0/1)', 'num_compromised: 被入侵指标数', 'root_shell: 是否获得root shell (0/1)',
            'su_attempted: 是否尝试su (0/1)', 'num_root: root访问次数', 'num_file_creations: 文件创建次数',
            'num_shells: shell获取次数', 'num_access_files: 访问控制文件次数', 'num_outbound_cmds: 出站命令次数',
            'is_host_login: 是否主机登录 (0/1)'
        ]
        
        return APIResponse(
            success=True,
            data=feature_info,
            message="特征信息获取成功"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"获取特征信息失败: {str(e)}"
        )

@router.get("/integration/stats")
async def get_integration_statistics():
    """获取AI-区块链集成统计信息"""
    try:
        stats = ai_blockchain_integration.get_integration_statistics()
        return APIResponse(
            success=True,
            data=stats,
            message="集成统计信息获取成功"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"获取集成统计信息失败: {str(e)}"
        )

@router.get("/attack-history")
async def get_attack_history(source_ip: str = None, limit: int = 50):
    """获取指定IP的攻击历史"""
    try:
        history = ai_blockchain_integration.get_attack_history(source_ip=source_ip, limit=limit)
        return APIResponse(
            success=True,
            data=history,
            message="攻击历史获取成功"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"获取攻击历史失败: {str(e)}"
        )

@router.post("/integration/configure")
async def configure_integration(
    auto_block: bool = None,
    auto_pattern: bool = None, 
    attack_threshold: float = None,
    count_threshold: int = None
):
    """配置AI-区块链集成参数"""
    try:
        config = ai_blockchain_integration.configure_auto_actions(
            auto_block=auto_block,
            auto_pattern=auto_pattern,
            attack_threshold=attack_threshold,
            count_threshold=count_threshold
        )
        return APIResponse(
            success=True,
            data=config,
            message="集成配置更新成功"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"配置失败: {str(e)}"
        )

@router.post("/attack-history/clear")
async def clear_attack_history(source_ip: str = None):
    """清除指定IP或所有攻击历史"""
    try:
        ai_blockchain_integration.clear_attack_history(source_ip=source_ip)
        message = f"IP {source_ip} 的攻击历史已清除" if source_ip else "所有攻击历史已清除"
        return APIResponse(
            success=True,
            message=message
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"清除历史失败: {str(e)}"
        )