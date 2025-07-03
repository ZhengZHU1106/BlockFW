"""
区块链交互服务模块
整合智能合约交互、部署、测试等功能
"""

import json
import os
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from web3 import Web3
from web3.contract import Contract
from web3.exceptions import ContractLogicError, TransactionNotFound
import time

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BlockchainService:
    """区块链交互服务类"""
    
    def __init__(self, rpc_url: str = "http://127.0.0.1:7545"):
        """
        初始化区块链服务
        
        Args:
            rpc_url: Ganache RPC URL
        """
        self.rpc_url = rpc_url
        self.w3 = None
        self.contract = None
        self.contract_address = None
        self.contract_abi = None
        self.accounts = []
        
        # 合约信息文件路径
        self.contract_info_file = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
            'scripts', 'contract_info.json'
        )
        
        # 初始化连接
        self.connect()
        
    def connect(self) -> bool:
        """连接到区块链网络"""
        try:
            self.w3 = Web3(Web3.HTTPProvider(self.rpc_url))
            
            # 检查连接
            if not self.w3.is_connected():
                logger.error(f"无法连接到区块链网络: {self.rpc_url}")
                return False
            
            # 获取账户列表
            self.accounts = self.w3.eth.accounts
            if not self.accounts:
                logger.error("未找到可用账户")
                return False
            
            # 设置默认账户
            self.w3.eth.default_account = self.accounts[0]
            
            logger.info(f"✅ 成功连接到区块链网络: {self.rpc_url}")
            logger.info(f"✅ 找到 {len(self.accounts)} 个账户")
            logger.info(f"✅ 默认账户: {self.w3.eth.default_account}")
            
            # 尝试加载现有合约
            self.load_contract()
            
            return True
            
        except Exception as e:
            logger.error(f"连接区块链失败: {e}")
            return False
    
    def load_contract(self) -> bool:
        """加载现有的智能合约"""
        try:
            if not os.path.exists(self.contract_info_file):
                logger.warning("合约信息文件不存在，需要先部署合约")
                return False
            
            with open(self.contract_info_file, 'r') as f:
                contract_info = json.load(f)
            
            self.contract_address = contract_info['address']
            self.contract_abi = contract_info['abi']
            
            # 创建合约实例
            self.contract = self.w3.eth.contract(
                address=self.contract_address,
                abi=self.contract_abi
            )
            
            logger.info(f"✅ 成功加载智能合约: {self.contract_address}")
            return True
            
        except Exception as e:
            logger.error(f"加载合约失败: {e}")
            return False
    
    def deploy_contract(self) -> bool:
        """部署智能合约"""
        try:
            # 导入编译工具
            from solcx import compile_standard, install_solc
            
            # 安装Solidity编译器
            install_solc('0.8.0')
            
            # 读取合约源码
            contract_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                'contracts', 'FirewallRules.sol'
            )
            
            if not os.path.exists(contract_path):
                logger.error(f"合约文件不存在: {contract_path}")
                return False
            
            with open(contract_path, 'r', encoding='utf-8') as f:
                source_code = f.read()
            
            # 编译合约
            logger.info("🔧 正在编译智能合约...")
            compiled = compile_standard({
                "language": "Solidity",
                "sources": {"FirewallRules.sol": {"content": source_code}},
                "settings": {
                    "outputSelection": {
                        "*": {"*": ["abi", "metadata", "evm.bytecode"]}
                    }
                }
            }, solc_version="0.8.0")
            
            # 提取ABI和字节码
            contract_interface = compiled["contracts"]["FirewallRules.sol"]["FirewallRules"]
            abi = contract_interface["abi"]
            bytecode = contract_interface["evm"]["bytecode"]["object"]
            
            # 部署合约
            logger.info("🚀 正在部署智能合约...")
            FirewallRules = self.w3.eth.contract(abi=abi, bytecode=bytecode)
            tx_hash = FirewallRules.constructor().transact()
            tx_receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
            
            self.contract_address = tx_receipt.contractAddress
            self.contract_abi = abi
            self.contract = self.w3.eth.contract(
                address=self.contract_address,
                abi=self.contract_abi
            )
            
            # 保存合约信息
            contract_info = {
                "address": self.contract_address,
                "abi": abi,
                "deployed_at": datetime.now().isoformat(),
                "deployer": self.w3.eth.default_account,
                "network": "Ganache Local"
            }
            
            os.makedirs(os.path.dirname(self.contract_info_file), exist_ok=True)
            with open(self.contract_info_file, 'w') as f:
                json.dump(contract_info, f, indent=2)
            
            logger.info(f"✅ 智能合约部署成功: {self.contract_address}")
            logger.info(f"✅ 合约信息已保存: {self.contract_info_file}")
            
            return True
            
        except Exception as e:
            logger.error(f"部署合约失败: {e}")
            return False
    
    def get_contract_info(self) -> Dict[str, Any]:
        """获取合约基本信息"""
        if not self.contract:
            return {"error": "合约未加载"}
        
        try:
            # 获取合约状态
            signers_count = len(self.contract.functions.signers(0).call() if self.accounts else 0)
            min_signatures = self.contract.functions.minSignatures().call()
            detection_threshold = self.contract.functions.detectionThreshold().call()
            blocked_ports_count = self.contract.functions.getLength().call()
            attack_patterns_count = self.contract.functions.getAttackPatternLength().call()
            
            return {
                "address": self.contract_address,
                "network": "Ganache Local",
                "min_signatures": min_signatures,
                "detection_threshold": detection_threshold,
                "blocked_ports_count": blocked_ports_count,
                "attack_patterns_count": attack_patterns_count,
                "available_accounts": len(self.accounts),
                "current_account": self.w3.eth.default_account
            }
            
        except Exception as e:
            logger.error(f"获取合约信息失败: {e}")
            return {"error": str(e)}
    
    def add_port_vote(self, port: int, signer_index: int = 0) -> Dict[str, Any]:
        """对端口封锁进行投票"""
        if not self.contract:
            return {"success": False, "error": "合约未加载"}
        
        try:
            # 设置投票账户
            if signer_index >= len(self.accounts):
                return {"success": False, "error": f"账户索引超出范围: {signer_index}"}
            
            original_account = self.w3.eth.default_account
            self.w3.eth.default_account = self.accounts[signer_index]
            
            # 检查是否已经封锁
            if self.contract.functions.isBlocked(port).call():
                self.w3.eth.default_account = original_account
                return {"success": False, "error": f"端口 {port} 已被封锁"}
            
            # 检查是否已经投票
            has_voted = self.contract.functions.portVotes(port, self.accounts[signer_index]).call()
            if has_voted:
                self.w3.eth.default_account = original_account
                return {"success": False, "error": f"账户 {self.accounts[signer_index]} 已对端口 {port} 投票"}
            
            # 执行投票
            tx_hash = self.contract.functions.addPort(port).transact()
            tx_receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
            
            # 检查当前投票状态
            vote_count = self.contract.functions.portVoteCount(port).call()
            is_blocked = self.contract.functions.isBlocked(port).call()
            min_signatures = self.contract.functions.minSignatures().call()
            
            # 恢复原账户
            self.w3.eth.default_account = original_account
            
            return {
                "success": True,
                "transaction_hash": tx_hash.hex(),
                "port": port,
                "voter": self.accounts[signer_index],
                "vote_count": vote_count,
                "min_signatures": min_signatures,
                "is_blocked": is_blocked,
                "gas_used": tx_receipt.gasUsed
            }
            
        except ContractLogicError as e:
            self.w3.eth.default_account = original_account
            return {"success": False, "error": f"合约逻辑错误: {str(e)}"}
        except Exception as e:
            self.w3.eth.default_account = original_account
            return {"success": False, "error": str(e)}
    
    def auto_block_port(self, port: int) -> Dict[str, Any]:
        """AI自动封锁端口（无需投票）"""
        if not self.contract:
            return {"success": False, "error": "合约未加载"}
        
        try:
            # 检查是否已经封锁
            if self.contract.functions.isBlocked(port).call():
                return {"success": False, "error": f"端口 {port} 已被封锁"}
            
            # 执行自动封锁
            tx_hash = self.contract.functions.autoBlock(port).transact()
            tx_receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
            
            return {
                "success": True,
                "transaction_hash": tx_hash.hex(),
                "port": port,
                "blocked_by": "AI_AUTO",
                "gas_used": tx_receipt.gasUsed,
                "timestamp": datetime.now().isoformat()
            }
            
        except ContractLogicError as e:
            return {"success": False, "error": f"合约逻辑错误: {str(e)}"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def add_attack_pattern(self, pattern: int, description: str = "") -> Dict[str, Any]:
        """添加攻击模式到区块链"""
        if not self.contract:
            return {"success": False, "error": "合约未加载"}
        
        try:
            # 执行添加攻击模式
            tx_hash = self.contract.functions.addAttackPattern(pattern).transact()
            tx_receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
            
            # 获取当前攻击模式数量
            pattern_count = self.contract.functions.getAttackPatternLength().call()
            
            return {
                "success": True,
                "transaction_hash": tx_hash.hex(),
                "pattern": pattern,
                "description": description,
                "pattern_index": pattern_count - 1,
                "gas_used": tx_receipt.gasUsed,
                "timestamp": datetime.now().isoformat()
            }
            
        except ContractLogicError as e:
            return {"success": False, "error": f"合约逻辑错误: {str(e)}"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_blocked_ports(self) -> List[int]:
        """获取所有被封锁的端口"""
        if not self.contract:
            return []
        
        try:
            length = self.contract.functions.getLength().call()
            blocked_ports = []
            
            for i in range(length):
                port = self.contract.functions.getPort(i).call()
                blocked_ports.append(port)
            
            return blocked_ports
            
        except Exception as e:
            logger.error(f"获取封锁端口失败: {e}")
            return []
    
    def get_attack_patterns(self) -> List[int]:
        """获取所有攻击模式"""
        if not self.contract:
            return []
        
        try:
            length = self.contract.functions.getAttackPatternLength().call()
            patterns = []
            
            for i in range(length):
                pattern = self.contract.functions.getAttackPattern(i).call()
                patterns.append(pattern)
            
            return patterns
            
        except Exception as e:
            logger.error(f"获取攻击模式失败: {e}")
            return []
    
    def set_detection_threshold(self, threshold: int) -> Dict[str, Any]:
        """设置检测阈值"""
        if not self.contract:
            return {"success": False, "error": "合约未加载"}
        
        try:
            tx_hash = self.contract.functions.setDetectionThreshold(threshold).transact()
            tx_receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
            
            return {
                "success": True,
                "transaction_hash": tx_hash.hex(),
                "new_threshold": threshold,
                "gas_used": tx_receipt.gasUsed,
                "timestamp": datetime.now().isoformat()
            }
            
        except ContractLogicError as e:
            return {"success": False, "error": f"合约逻辑错误: {str(e)}"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def setup_multi_sig(self, signer_addresses: List[str], min_signatures: int) -> Dict[str, Any]:
        """设置多签名配置"""
        if not self.contract:
            return {"success": False, "error": "合约未加载"}
        
        try:
            # 验证输入
            if len(signer_addresses) == 0:
                return {"success": False, "error": "签名人列表不能为空"}
            
            if min_signatures > len(signer_addresses):
                return {"success": False, "error": "最小签名数不能超过签名人数量"}
            
            # 执行设置
            tx_hash = self.contract.functions.setSigners(signer_addresses, min_signatures).transact()
            tx_receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
            
            return {
                "success": True,
                "transaction_hash": tx_hash.hex(),
                "signers": signer_addresses,
                "min_signatures": min_signatures,
                "gas_used": tx_receipt.gasUsed,
                "timestamp": datetime.now().isoformat()
            }
            
        except ContractLogicError as e:
            return {"success": False, "error": f"合约逻辑错误: {str(e)}"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def test_contract_features(self) -> Dict[str, Any]:
        """自动测试合约所有功能"""
        results = {
            "timestamp": datetime.now().isoformat(),
            "tests": {}
        }
        
        try:
            # 1. 设置多签名
            logger.info("🧪 测试1: 多签名设置")
            signers = self.accounts[:3] if len(self.accounts) >= 3 else self.accounts
            min_sigs = 2 if len(signers) >= 2 else 1
            
            setup_result = self.setup_multi_sig(signers, min_sigs)
            results["tests"]["multi_sig_setup"] = setup_result
            
            if setup_result.get("success"):
                logger.info("✅ 多签名设置成功")
            else:
                logger.error("❌ 多签名设置失败")
            
            # 2. 测试端口投票封锁
            logger.info("🧪 测试2: 端口投票封锁")
            test_port = 10086
            vote_results = []
            
            for i in range(min(min_sigs, len(signers))):
                vote_result = self.add_port_vote(test_port, i)
                vote_results.append(vote_result)
                
            results["tests"]["port_voting"] = vote_results
            
            # 3. 测试攻击模式添加
            logger.info("🧪 测试3: 攻击模式添加")
            pattern_result = self.add_attack_pattern(12345, "测试攻击模式")
            results["tests"]["attack_pattern"] = pattern_result
            
            # 4. 测试阈值设置
            logger.info("🧪 测试4: 检测阈值设置")
            threshold_result = self.set_detection_threshold(20)
            results["tests"]["threshold_setting"] = threshold_result
            
            # 5. 测试自动封锁
            logger.info("🧪 测试5: AI自动封锁")
            auto_block_result = self.auto_block_port(8080)
            results["tests"]["auto_block"] = auto_block_result
            
            # 6. 获取合约状态
            logger.info("🧪 测试6: 合约状态查询")
            contract_info = self.get_contract_info()
            blocked_ports = self.get_blocked_ports()
            attack_patterns = self.get_attack_patterns()
            
            results["tests"]["contract_status"] = {
                "info": contract_info,
                "blocked_ports": blocked_ports,
                "attack_patterns": attack_patterns
            }
            
            logger.info("✅ 合约功能测试完成")
            
        except Exception as e:
            logger.error(f"合约测试失败: {e}")
            results["error"] = str(e)
        
        return results


# 创建全局区块链服务实例
blockchain_service = BlockchainService()