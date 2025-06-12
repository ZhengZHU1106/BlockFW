import json
from web3 import Web3

# 读取部署信息
with open("contract_info.json", "r") as f:
    info = json.load(f)

address = info["address"]
abi = info["abi"]

# 连接 Ganache
w3 = Web3(Web3.HTTPProvider("http://127.0.0.1:7545"))
w3.eth.default_account = w3.eth.accounts[0]

contract = w3.eth.contract(address=address, abi=abi)

# 添加一个端口（比如 22）
tx_hash = contract.functions.addPort(22).transact()
w3.eth.wait_for_transaction_receipt(tx_hash)

print("✅ 已添加封锁端口 22 到区块链")
