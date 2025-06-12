import json
import os
from web3 import Web3

# 加载合约地址与 ABI
with open("contract_info.json", "r") as f:
    info = json.load(f)

address = info["address"]
abi = info["abi"]

# 连接 Ganache
w3 = Web3(Web3.HTTPProvider("http://127.0.0.1:7545"))
contract = w3.eth.contract(address=address, abi=abi)

# 获取被封锁的端口列表
length = contract.functions.getLength().call()
blocked_ports = [contract.functions.getPort(i).call() for i in range(length)]

print("🚫 需要封锁的端口列表：", blocked_ports)

# 替代真正的防火墙命令（仅输出模拟行为）
def block_port(port):
    print(f"❗ 模拟封锁端口 {port} —— Windows 无法执行 iptables")


for port in blocked_ports:
    block_port(port)
