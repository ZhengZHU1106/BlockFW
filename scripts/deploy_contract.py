from web3 import Web3
from solcx import compile_standard, install_solc
import json
import os

install_solc('0.8.0')

# 读取合约
with open("../contracts/FirewallRules.sol", "r") as file:
    source_code = file.read()

compiled = compile_standard({
    "language": "Solidity",
    "sources": {"FirewallRules.sol": {"content": source_code}},
    "settings": {
        "outputSelection": {
            "*": {"*": ["abi", "metadata", "evm.bytecode"]}
        }
    }
}, solc_version="0.8.0")

abi = compiled["contracts"]["FirewallRules.sol"]["FirewallRules"]["abi"]
bytecode = compiled["contracts"]["FirewallRules.sol"]["FirewallRules"]["evm"]["bytecode"]["object"]

# 连接 Ganache
w3 = Web3(Web3.HTTPProvider("http://127.0.0.1:7545"))
w3.eth.default_account = w3.eth.accounts[0]

FirewallRules = w3.eth.contract(abi=abi, bytecode=bytecode)
tx_hash = FirewallRules.constructor().transact()
tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)

print("✅ 合约地址：", tx_receipt.contractAddress)

# 保存合约地址和 abi
with open("contract_info.json", "w") as f:
    json.dump({
        "address": tx_receipt.contractAddress,
        "abi": abi
    }, f, indent=2)
