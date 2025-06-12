import json
import random
import time
import os

# 模拟攻击目标端口
target_ports = [22, 80, 443]
attack_counts = {port: 0 for port in target_ports}

# 输出文件
TRAFFIC_FILE = "traffic.json"

# 配置
ATTACK_DURATION = 30  # 总持续时间（秒）
FIRE_RATE = 5  # 每秒攻击次数

print("💥 攻击者启动！")

for sec in range(ATTACK_DURATION):
    port = random.choice(target_ports)
    for _ in range(FIRE_RATE):
        attack_counts[port] += 1
        print(f"🔴 攻击：端口 {port} 总访问次数：{attack_counts[port]}")

    # 写入共享文件
    with open(TRAFFIC_FILE, "w") as f:
        json.dump(attack_counts, f)

    time.sleep(1)

print("✅ 攻击结束。")
