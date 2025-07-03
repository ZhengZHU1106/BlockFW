"""
AI检测相关的API路由
"""

from fastapi import APIRouter, HTTPException
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
async def detect_single(request: DetectionRequest):
    """
    执行单次AI检测
    
    - **features**: 28个网络流量特征（已预处理）
    - **metadata**: 可选的元数据
    """
    try:
        # 执行检测
        result = ai_detection_manager.detect(request.features)
        
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
            features_used=result.get('features_used', 28),
            metadata=request.metadata
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
                    features_used=result.get('features_used', 28),
                    metadata={"batch_index": idx}
                ))
        
        return results
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"批量检测错误: {str(e)}"
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
        # 生成21个随机特征（BNaT格式）
        random_features = np.random.random(21).tolist()
        
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
        if ai_detection_manager.has_preprocessor:
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
                0.0,      # duration
                'tcp',    # protocol_type
                'http',   # service
                181.0,    # src_bytes
                5450.0,   # dst_bytes
                'SF',     # flag
                0,        # land
                0,        # wrong_fragment
                0,        # urgent
                0,        # hot
                0,        # num_failed_logins
                1,        # logged_in
                0,        # num_compromised
                0,        # root_shell
                0,        # su_attempted
                0,        # num_root
                0,        # num_file_creations
                0,        # num_shells
                0,        # num_access_files
                0,        # num_outbound_cmds
                0         # is_host_login
            ]
        }
        
        feature_info['feature_descriptions'] = [
            'duration: 连接持续时间',
            'protocol_type: 协议类型 (tcp/udp/icmp)',
            'service: 服务类型 (http/ftp/ssh等)',
            'src_bytes: 源字节数',
            'dst_bytes: 目标字节数',
            'flag: 连接标志 (SF/S0/REJ等)',
            'land: 是否land攻击 (0/1)',
            'wrong_fragment: 错误片段数',
            'urgent: 紧急数据包数',
            'hot: 热点指标数',
            'num_failed_logins: 失败登录次数',
            'logged_in: 是否成功登录 (0/1)',
            'num_compromised: 被入侵指标数',
            'root_shell: 是否获得root shell (0/1)',
            'su_attempted: 是否尝试su (0/1)',
            'num_root: root访问次数',
            'num_file_creations: 文件创建次数',
            'num_shells: shell获取次数',
            'num_access_files: 访问控制文件次数',
            'num_outbound_cmds: 出站命令次数',
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