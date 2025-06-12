import json
import os
import time
from web3 import Web3

# === 1. 连接合约 ===
with open("contract_info.json", "r") as f:
    info = json.load(f)

address = info["address"]
abi = info["abi"]

w3 = Web3(Web3.HTTPProvider("http://127.0.0.1:7545"))
w3.eth.default_account = w3.eth.accounts[0]
contract = w3.eth.contract(address=address, abi=abi)

# === 2. 配置 ===
TRAFFIC_FILE = "traffic.json"
BLOCK_THRESHOLD = 15
blocked_ports = set()

def read_traffic():
    if os.path.exists(TRAFFIC_FILE):
        with open(TRAFFIC_FILE, "r") as f:
            return json.load(f)
    return {}

# === 3. 检测逻辑 ===
def detect_and_block(traffic):
    for port_str, count in traffic.items():
        port = int(port_str)
        print(f"[监测] 端口 {port} 的访问次数：{count}")
        if count > BLOCK_THRESHOLD and port not in blocked_ports:
            print(f"🚨 检测到异常行为！封锁端口 {port}")
            tx_hash = contract.functions.addPort(port).transact()
            w3.eth.wait_for_transaction_receipt(tx_hash)
            blocked_ports.add(port)

# === 4. 循环执行 ===
if __name__ == "__main__":
    print("🧠 AI 检测器启动中 ...")
    while True:
        traffic = read_traffic()
        detect_and_block(traffic)
        time.sleep(1)
