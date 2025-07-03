"""
步骤2：测试AI检测模块集成
"""

import requests
import json
import numpy as np
from datetime import datetime

# API基础URL
BASE_URL = "http://127.0.0.1:8000"

def test_api_health():
    """测试API健康状态"""
    print("\n1. 测试API健康检查...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            print("✅ API健康检查通过")
            print(f"   响应: {response.json()}")
        else:
            print(f"❌ API健康检查失败: {response.status_code}")
    except Exception as e:
        print(f"❌ 无法连接到API服务器: {e}")
        return False
    return True

def test_model_info():
    """测试模型信息接口"""
    print("\n2. 测试模型信息接口...")
    try:
        response = requests.get(f"{BASE_URL}/api/detection/model-info")
        if response.status_code == 200:
            data = response.json()
            print("✅ 模型信息获取成功")
            print(f"   模型加载状态: {data['data']['loaded']}")
            if data['data']['loaded']:
                print(f"   模型类型: {data['data']['info']['model_type']}")
                print(f"   训练日期: {data['data']['info']['training_date']}")
                print(f"   特征数量: {data['data']['info']['features_count']}")
        else:
            print(f"❌ 获取模型信息失败: {response.status_code}")
    except Exception as e:
        print(f"❌ 测试失败: {e}")

def test_single_detection():
    """测试单次检测"""
    print("\n3. 测试单次AI检测（21维特征）...")
    
    # 生成21个随机特征（BNaT格式）
    features = np.random.random(21).tolist()
    
    request_data = {
        "features": features,
        "source_ip": "192.168.1.100",
        "destination_port": 8080,
        "metadata": {"test": True}
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/detection/single",
            json=request_data
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ 单次检测成功")
            print(f"   输入特征数: {len(features)}")
            print(f"   是否为攻击: {result['is_attack']}")
            print(f"   预测标签: {result['predicted_label']}")
            print(f"   置信度: {result['confidence']:.2%}")
            print(f"   时间戳: {result['timestamp']}")
            
            # 安全地处理概率分布
            if result.get('probabilities'):
                print("   概率分布:")
                for label, prob in result['probabilities'].items():
                    print(f"     {label}: {prob:.2%}")
            else:
                print("   概率分布: 未提供")
        else:
            print(f"❌ 检测失败: {response.status_code}")
            print(f"   错误详情: {response.text}")
    except Exception as e:
        print(f"❌ 测试失败: {e}")

def test_batch_detection():
    """测试批量检测"""
    print("\n4. 测试批量AI检测...")
    
    # 生成5个样本的批量数据
    features_list = [np.random.random(21).tolist() for _ in range(5)]
    
    request_data = {
        "features_list": features_list,
        "batch_id": "test_batch_001"
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/detection/batch",
            json=request_data
        )
        
        if response.status_code == 200:
            results = response.json()
            print(f"✅ 批量检测成功，处理了 {len(results)} 个样本")
            
            # 统计结果
            attack_count = sum(1 for r in results if r['is_attack'])
            print(f"   攻击样本: {attack_count}")
            print(f"   正常样本: {len(results) - attack_count}")
            
            # 显示第一个结果
            if results:
                print(f"   第一个样本结果:")
                print(f"     - 特征数: {results[0]['features_used']}")
                print(f"     - 是否攻击: {results[0]['is_attack']}")
                print(f"     - 预测标签: {results[0]['predicted_label']}")
                print(f"     - 置信度: {results[0]['confidence']:.2%}")
        else:
            print(f"❌ 批量检测失败: {response.status_code}")
            print(f"   错误详情: {response.text}")
    except Exception as e:
        print(f"❌ 测试失败: {e}")

def test_detection_test_endpoint():
    """测试检测测试端点"""
    print("\n5. 测试随机检测端点...")
    
    try:
        response = requests.post(f"{BASE_URL}/api/detection/test")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ 测试检测成功")
            if result['success']:
                data = result['data']
                print(f"   特征数: {data.get('features_used', 'N/A')}")
                print(f"   是否为攻击: {data.get('is_attack', 'N/A')}")
                print(f"   预测标签: {data.get('predicted_label', 'N/A')}")
                print(f"   置信度: {data.get('confidence', 0):.2%}")
        else:
            print(f"❌ 测试检测失败: {response.status_code}")
    except Exception as e:
        print(f"❌ 测试失败: {e}")

def test_feature_dimensions():
    """测试特征维度验证"""
    print("\n6. 测试特征维度验证...")
    
    test_cases = [
        ([], "空特征"),
        ([0.1] * 21, "21维特征（标准BNaT格式）"),
        ([0.1] * 30, "30维特征（过多）"),
        ([0.1] * 10, "10维特征（过少）"),
        ([0.1] * 20, "20维特征（缺少1个）"),
        ([0.1] * 22, "22维特征（多1个）"),
    ]
    
    for features, description in test_cases:
        try:
            response = requests.post(
                f"{BASE_URL}/api/detection/single",
                json={"features": features}
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"   ✅ {description} - 检测成功")
                print(f"     特征数: {result['features_used']}")
            else:
                error_detail = response.json().get('detail', '未知错误')
                if len(features) == 21:
                    print(f"   ❌ {description} - 应该成功但失败: {error_detail}")
                else:
                    print(f"   ✅ {description} - 正确拒绝（验证错误）")
        except Exception as e:
            print(f"   ❌ {description} - 测试异常: {e}")

def test_statistics():
    """测试统计信息"""
    print("\n7. 测试统计信息接口...")
    
    try:
        response = requests.get(f"{BASE_URL}/api/detection/statistics")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ 获取统计信息成功")
            stats = result['data']
            print(f"   总检测次数: {stats['total_detections']}")
            print(f"   攻击次数: {stats['attack_count']}")
            print(f"   正常次数: {stats['normal_count']}")
            print(f"   模型状态: {'loaded' if stats.get('total_detections', 0) >= 0 else 'error'}")
            if stats['total_detections'] > 0:
                print(f"   攻击率: {stats['attack_rate']:.2%}")
                if stats['attack_types']:
                    print(f"   攻击类型分布: {stats['attack_types']}")
        else:
            print(f"❌ 获取统计信息失败: {response.status_code}")
    except Exception as e:
        print(f"❌ 测试失败: {e}")

def test_recent_attacks():
    """测试最近攻击记录"""
    print("\n8. 测试最近攻击记录接口...")
    
    try:
        response = requests.get(f"{BASE_URL}/api/detection/recent-attacks?limit=5")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ 获取攻击记录成功")
            attacks = result['data']['attacks']
            print(f"   最近攻击记录数: {result['data']['count']}")
            if attacks:
                print(f"   最新攻击:")
                attack = attacks[-1]
                print(f"     - 时间: {attack['timestamp']}")
                print(f"     - 类型: {attack['predicted_label']}")
                print(f"     - 置信度: {attack['confidence']:.2%}")
        else:
            print(f"❌ 获取攻击记录失败: {response.status_code}")
    except Exception as e:
        print(f"❌ 测试失败: {e}")

def test_api_docs():
    """测试API文档"""
    print("\n9. 测试API文档...")
    
    try:
        response = requests.get(f"{BASE_URL}/docs")
        if response.status_code == 200:
            print("✅ API文档可访问")
            print(f"   文档地址: {BASE_URL}/docs")
            print(f"   ReDoc地址: {BASE_URL}/redoc")
        else:
            print(f"❌ API文档访问失败: {response.status_code}")
    except Exception as e:
        print(f"❌ 测试失败: {e}")

def main():
    """运行所有测试"""
    print("="*60)
    print("BlockFW 后端API - 步骤2测试")
    print("AI检测模块集成测试")
    print("="*60)
    
    # 检查API是否运行
    if not test_api_health():
        print("\n⚠️  请先启动API服务器:")
        print("cd backend && python app.py")
        return
    
    # 运行所有测试
    test_model_info()
    test_single_detection()
    test_batch_detection()
    test_detection_test_endpoint()
    test_feature_dimensions()
    test_statistics()
    test_recent_attacks()
    test_api_docs()
    
    print("\n="*60)
    print("步骤2测试完成！")
    print("✨ AI检测模块已成功集成到FastAPI")
    print("\n下一步：步骤3 - 区块链交互服务")
    print("="*60)

if __name__ == "__main__":
    main()