#!/usr/bin/env python3
"""
快速检查API配置是否正确
"""

import requests
import numpy as np

BASE_URL = "http://127.0.0.1:8000"

def check_api_running():
    """检查API是否运行"""
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def test_21_features():
    """测试21维特征是否能被接受"""
    features = np.random.random(21).tolist()
    request_data = {"features": features}
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/detection/single",
            json=request_data,
            timeout=10
        )
        
        print(f"状态码: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ 21维特征测试成功")
            print(f"   是否攻击: {result.get('is_attack')}")
            print(f"   特征数: {result.get('features_used')}")
            return True
        else:
            print("❌ 21维特征测试失败")
            print(f"错误详情: {response.text}")
            return False
    except Exception as e:
        print(f"❌ 请求异常: {e}")
        return False

def test_invalid_features():
    """测试无效特征是否被正确拒绝"""
    for size, description in [(20, "20维"), (22, "22维"), (10, "10维")]:
        features = [0.1] * size
        request_data = {"features": features}
        
        try:
            response = requests.post(
                f"{BASE_URL}/api/detection/single",
                json=request_data,
                timeout=5
            )
            
            if response.status_code != 200:
                print(f"✅ {description}特征正确被拒绝")
            else:
                print(f"❌ {description}特征应该被拒绝但被接受了")
        except Exception as e:
            print(f"⚠️ {description}特征测试异常: {e}")

def main():
    print("🔍 快速API配置检查")
    print("=" * 40)
    
    # 检查API是否运行
    if not check_api_running():
        print("❌ API服务器未运行")
        print("请先启动: cd backend && python app.py")
        return
    
    print("✅ API服务器运行中")
    
    # 测试21维特征
    print("\n📊 测试21维特征...")
    if test_21_features():
        print("\n🔒 测试无效特征...")
        test_invalid_features()
    
    print("\n" + "=" * 40)
    print("检查完成！")

if __name__ == "__main__":
    main()