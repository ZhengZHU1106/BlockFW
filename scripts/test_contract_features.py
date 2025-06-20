import json
from web3 import Web3

# 读取合约信息
with open("contract_info.json", "r") as f:
    info = json.load(f)

address = info["address"]
abi = info["abi"]

# 连接 Ganache
w3 = Web3(Web3.HTTPProvider("http://127.0.0.1:7545"))
accounts = w3.eth.accounts

contract = w3.eth.contract(address=address, abi=abi)

print("==== 合约功能自动化验证 ====")

# 1. 多签名人管理 (setSigners) —— 必须最先做
print("\n[1] 多签名人管理 (setSigners)")
new_signers = [accounts[0], accounts[1], accounts[2]]
min_sigs = 2
w3.eth.default_account = accounts[0]
tx_hash = contract.functions.setSigners(new_signers, min_sigs).transact()
w3.eth.wait_for_transaction_receipt(tx_hash)
for acc in new_signers:
    is_signer = contract.functions.isSigner(acc).call()
    print(f"  {acc} 是否为签名人: {is_signer}")
print(f"  最小签名数: {contract.functions.minSignatures().call()}")

# 2. 多签名端口封锁 (addPort)
print("\n[2] 多签名端口封锁 (addPort)")
port = 10086
for i in range(2):
    w3.eth.default_account = accounts[i]
    if not contract.functions.isBlocked(port).call():
        tx_hash = contract.functions.addPort(port).transact()
        w3.eth.wait_for_transaction_receipt(tx_hash)
        print(f"  签名人 {accounts[i]} 投票封锁端口 {port}")
    else:
        print(f"  端口 {port} 已被封锁，跳过投票")

is_blocked = contract.functions.isBlocked(port).call()
print(f"  端口 {port} 是否已封锁: {is_blocked}")

# 3. AI检测结果上链 (addAttackPattern)
print("\n[3] AI检测结果上链 (addAttackPattern)")
pattern = 12345
w3.eth.default_account = accounts[0]
tx_hash = contract.functions.addAttackPattern(pattern).transact()
w3.eth.wait_for_transaction_receipt(tx_hash)
read_pattern = contract.functions.getAttackPattern(0).call()
print(f"  写入攻击模式: {pattern}，读取攻击模式: {read_pattern}")

# 4. 检测阈值设置 (setDetectionThreshold)
print("\n[4] 检测阈值设置 (setDetectionThreshold)")
new_threshold = 20
tx_hash = contract.functions.setDetectionThreshold(new_threshold).transact()
w3.eth.wait_for_transaction_receipt(tx_hash)
current_threshold = contract.functions.detectionThreshold().call()
print(f"  设置阈值: {new_threshold}，当前阈值: {current_threshold}")

# 5. 自动封锁 (autoBlock)
print("\n[5] 自动封锁 (autoBlock)")
auto_port = 8080
w3.eth.default_account = accounts[0]
tx_hash = contract.functions.autoBlock(auto_port).transact()
w3.eth.wait_for_transaction_receipt(tx_hash)
is_auto_blocked = contract.functions.isBlocked(auto_port).call()
print(f"  端口 {auto_port} 是否已自动封锁: {is_auto_blocked}")

print("\n==== 合约功能验证结束 ====") 