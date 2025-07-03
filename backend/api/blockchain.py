"""
区块链相关的API路由
"""

from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any
from datetime import datetime

from models.schemas import (
    BlockPortRequest, 
    BlockPortResult,
    AttackPatternRequest,
    AttackPatternResult,
    VoteRequest,
    VoteResult,
    ContractInfo,
    APIResponse,
    ErrorResponse,
    BlockchainStatus
)
from services.blockchain_service import blockchain_service

# 创建路由器
router = APIRouter(
    prefix="/api/blockchain",
    tags=["Blockchain"],
    responses={
        404: {"description": "Not found"},
        500: {"description": "Internal server error"}
    }
)

@router.get("/status")
async def get_blockchain_status():
    """获取区块链连接状态"""
    try:
        if not blockchain_service.w3 or not blockchain_service.w3.is_connected():
            return APIResponse(
                success=False,
                message="区块链网络未连接",
                data={"status": "disconnected"}
            )
        
        contract_info = blockchain_service.get_contract_info()
        
        return APIResponse(
            success=True,
            message="区块链状态正常",
            data={
                "status": "connected",
                "network": blockchain_service.rpc_url,
                "accounts_count": len(blockchain_service.accounts),
                "contract_loaded": blockchain_service.contract is not None,
                "contract_info": contract_info
            }
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"获取区块链状态失败: {str(e)}"
        )

@router.post("/deploy")
async def deploy_contract():
    """部署智能合约"""
    try:
        success = blockchain_service.deploy_contract()
        
        if success:
            contract_info = blockchain_service.get_contract_info()
            return APIResponse(
                success=True,
                message="智能合约部署成功",
                data={
                    "contract_address": blockchain_service.contract_address,
                    "deployer": blockchain_service.w3.eth.default_account,
                    "contract_info": contract_info
                }
            )
        else:
            raise HTTPException(
                status_code=500,
                detail="智能合约部署失败"
            )
            
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"部署合约时发生错误: {str(e)}"
        )

@router.get("/contract/info")
async def get_contract_info():
    """获取智能合约信息"""
    try:
        if not blockchain_service.contract:
            return APIResponse(
                success=False,
                message="智能合约未加载，请先部署合约",
                data=None
            )
        
        contract_info = blockchain_service.get_contract_info()
        
        if "error" in contract_info:
            raise HTTPException(
                status_code=500,
                detail=contract_info["error"]
            )
        
        return APIResponse(
            success=True,
            message="合约信息获取成功",
            data=contract_info
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"获取合约信息失败: {str(e)}"
        )

@router.post("/port/vote", response_model=VoteResult)
async def vote_block_port(request: VoteRequest):
    """对端口封锁进行投票"""
    try:
        if not blockchain_service.contract:
            raise HTTPException(
                status_code=400,
                detail="智能合约未加载，请先部署合约"
            )
        
        # 查找投票人账户索引
        signer_index = -1
        for i, account in enumerate(blockchain_service.accounts):
            if account.lower() == request.voter_address.lower():
                signer_index = i
                break
        
        if signer_index == -1:
            raise HTTPException(
                status_code=400,
                detail=f"投票人地址不在可用账户列表中: {request.voter_address}"
            )
        
        result = blockchain_service.add_port_vote(request.port, signer_index)
        
        if not result.get("success", False):
            raise HTTPException(
                status_code=400,
                detail=result.get("error", "投票失败")
            )
        
        return VoteResult(
            transaction_hash=result["transaction_hash"],
            port=result["port"],
            voter=result["voter"],
            current_votes=result["vote_count"],
            required_votes=result["min_signatures"],
            is_blocked=result["is_blocked"],
            status=BlockchainStatus.SUCCESS
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"投票失败: {str(e)}"
        )

@router.post("/port/auto-block")
async def auto_block_port(request: BlockPortRequest):
    """AI自动封锁端口"""
    try:
        if not blockchain_service.contract:
            raise HTTPException(
                status_code=400,
                detail="智能合约未加载，请先部署合约"
            )
        
        result = blockchain_service.auto_block_port(request.port)
        
        if not result.get("success", False):
            raise HTTPException(
                status_code=400,
                detail=result.get("error", "自动封锁失败")
            )
        
        return BlockPortResult(
            transaction_hash=result["transaction_hash"],
            port=result["port"],
            status=BlockchainStatus.SUCCESS,
            gas_used=result.get("gas_used")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"自动封锁失败: {str(e)}"
        )

@router.post("/attack-pattern/add")
async def add_attack_pattern(request: AttackPatternRequest):
    """添加攻击模式到区块链"""
    try:
        if not blockchain_service.contract:
            raise HTTPException(
                status_code=400,
                detail="智能合约未加载，请先部署合约"
            )
        
        result = blockchain_service.add_attack_pattern(
            request.pattern, 
            request.description
        )
        
        if not result.get("success", False):
            raise HTTPException(
                status_code=400,
                detail=result.get("error", "添加攻击模式失败")
            )
        
        return AttackPatternResult(
            transaction_hash=result["transaction_hash"],
            pattern=result["pattern"],
            status=BlockchainStatus.SUCCESS
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"添加攻击模式失败: {str(e)}"
        )

@router.get("/ports/blocked")
async def get_blocked_ports():
    """获取所有被封锁的端口"""
    try:
        if not blockchain_service.contract:
            return APIResponse(
                success=False,
                message="智能合约未加载",
                data={"blocked_ports": []}
            )
        
        blocked_ports = blockchain_service.get_blocked_ports()
        
        return APIResponse(
            success=True,
            message=f"获取到 {len(blocked_ports)} 个被封锁的端口",
            data={"blocked_ports": blocked_ports}
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"获取封锁端口失败: {str(e)}"
        )

@router.get("/attack-patterns")
async def get_attack_patterns():
    """获取所有攻击模式"""
    try:
        if not blockchain_service.contract:
            return APIResponse(
                success=False,
                message="智能合约未加载",
                data={"attack_patterns": []}
            )
        
        attack_patterns = blockchain_service.get_attack_patterns()
        
        return APIResponse(
            success=True,
            message=f"获取到 {len(attack_patterns)} 个攻击模式",
            data={"attack_patterns": attack_patterns}
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"获取攻击模式失败: {str(e)}"
        )

@router.post("/threshold/set")
async def set_detection_threshold(threshold: int):
    """设置检测阈值"""
    try:
        if not blockchain_service.contract:
            raise HTTPException(
                status_code=400,
                detail="智能合约未加载，请先部署合约"
            )
        
        if threshold <= 0:
            raise HTTPException(
                status_code=400,
                detail="检测阈值必须大于0"
            )
        
        result = blockchain_service.set_detection_threshold(threshold)
        
        if not result.get("success", False):
            raise HTTPException(
                status_code=400,
                detail=result.get("error", "设置阈值失败")
            )
        
        return APIResponse(
            success=True,
            message="检测阈值设置成功",
            data={
                "transaction_hash": result["transaction_hash"],
                "new_threshold": result["new_threshold"],
                "gas_used": result.get("gas_used")
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"设置阈值失败: {str(e)}"
        )

@router.post("/multi-sig/setup")
async def setup_multi_signature(signers: List[str], min_signatures: int):
    """设置多签名配置"""
    try:
        if not blockchain_service.contract:
            raise HTTPException(
                status_code=400,
                detail="智能合约未加载，请先部署合约"
            )
        
        if not signers:
            raise HTTPException(
                status_code=400,
                detail="签名人列表不能为空"
            )
        
        if min_signatures <= 0 or min_signatures > len(signers):
            raise HTTPException(
                status_code=400,
                detail="最小签名数必须在1到签名人数量之间"
            )
        
        result = blockchain_service.setup_multi_sig(signers, min_signatures)
        
        if not result.get("success", False):
            raise HTTPException(
                status_code=400,
                detail=result.get("error", "多签名设置失败")
            )
        
        return APIResponse(
            success=True,
            message="多签名配置设置成功",
            data={
                "transaction_hash": result["transaction_hash"],
                "signers": result["signers"],
                "min_signatures": result["min_signatures"],
                "gas_used": result.get("gas_used")
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"多签名设置失败: {str(e)}"
        )

@router.post("/test")
async def test_contract_features():
    """测试智能合约所有功能"""
    try:
        if not blockchain_service.contract:
            # 尝试部署合约
            deploy_success = blockchain_service.deploy_contract()
            if not deploy_success:
                raise HTTPException(
                    status_code=400,
                    detail="智能合约未部署且自动部署失败"
                )
        
        # 执行合约功能测试
        test_results = blockchain_service.test_contract_features()
        
        return APIResponse(
            success=True,
            message="合约功能测试完成",
            data=test_results
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"合约测试失败: {str(e)}"
        )

@router.get("/accounts")
async def get_available_accounts():
    """获取可用的区块链账户"""
    try:
        if not blockchain_service.w3:
            raise HTTPException(
                status_code=400,
                detail="区块链网络未连接"
            )
        
        accounts_info = []
        for i, account in enumerate(blockchain_service.accounts):
            try:
                balance = blockchain_service.w3.eth.get_balance(account)
                balance_eth = blockchain_service.w3.from_wei(balance, 'ether')
                
                accounts_info.append({
                    "index": i,
                    "address": account,
                    "balance_wei": balance,
                    "balance_eth": float(balance_eth),
                    "is_default": account == blockchain_service.w3.eth.default_account
                })
            except Exception as e:
                accounts_info.append({
                    "index": i,
                    "address": account,
                    "error": str(e)
                })
        
        return APIResponse(
            success=True,
            message=f"获取到 {len(accounts_info)} 个账户",
            data={"accounts": accounts_info}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"获取账户失败: {str(e)}"
        )