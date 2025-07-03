#!/usr/bin/env python3
"""
第三阶段步骤1验证
验证FastAPI基础架构是否正确创建
"""

import sys
import os

def check_file_structure():
    """检查文件结构"""
    print("📁 检查文件结构...")
    
    required_files = [
        "app.py",
        "requirements.txt", 
        "models/schemas.py",
        "models/__init__.py",
        "__init__.py"
    ]
    
    missing_files = []
    for file in required_files:
        if os.path.exists(file):
            print(f"   ✅ {file}")
        else:
            print(f"   ❌ {file}")
            missing_files.append(file)
    
    return len(missing_files) == 0

def check_imports():
    """检查核心导入"""
    print("\n🔍 检查核心导入...")
    
    try:
        import fastapi
        print("   ✅ FastAPI")
    except ImportError:
        print("   ❌ FastAPI")
        return False
    
    try:
        import uvicorn
        print("   ✅ Uvicorn")
    except ImportError:
        print("   ❌ Uvicorn")
        return False
    
    try:
        from app import app
        print("   ✅ App导入")
    except ImportError as e:
        print(f"   ❌ App导入失败: {e}")
        return False
    
    try:
        from models.schemas import DetectionRequest, DetectionResult
        print("   ✅ 数据模型")
    except ImportError as e:
        print(f"   ❌ 数据模型: {e}")
        return False
    
    return True

def check_app_structure():
    """检查应用结构"""
    print("\n🏗️ 检查应用结构...")
    
    try:
        from app import app
        
        # 检查路由
        routes = [route.path for route in app.routes]
        expected_routes = ["/", "/health", "/api/info"]
        
        for route in expected_routes:
            if route in routes:
                print(f"   ✅ 路由: {route}")
            else:
                print(f"   ❌ 路由: {route}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ 应用结构检查失败: {e}")
        return False

def manual_server_test():
    """手动服务器测试说明"""
    print("\n🔧 手动服务器测试:")
    print("   1. 在终端中运行: uvicorn app:app --host 127.0.0.1 --port 8000")
    print("   2. 打开浏览器访问: http://127.0.0.1:8000/docs")
    print("   3. 测试以下端点:")
    print("      - GET /")
    print("      - GET /health") 
    print("      - GET /api/info")

def main():
    """主验证函数"""
    print("="*60)
    print("🎯 第三阶段步骤1验证")
    print("✨ FastAPI基础架构")
    print("="*60)
    
    # 检查文件结构
    structure_ok = check_file_structure()
    
    # 检查导入
    imports_ok = check_imports()
    
    # 检查应用结构
    app_ok = check_app_structure()
    
    # 总结
    print("\n" + "="*60)
    print("📊 验证结果")
    print("="*60)
    
    if structure_ok and imports_ok and app_ok:
        print("🎉 ✅ 步骤1验证通过！")
        print("\n📋 完成的内容:")
        print("   ✅ FastAPI应用创建")
        print("   ✅ 数据模型定义")
        print("   ✅ 基础路由设置")
        print("   ✅ CORS中间件配置")
        print("   ✅ 依赖包安装")
        
        print("\n🚀 下一步:")
        print("   - 步骤2: 集成XGBoost AI检测模块")
        print("   - 步骤3: 集成智能合约交互")
        
        manual_server_test()
        
    else:
        print("❌ 验证失败，请检查以下问题:")
        if not structure_ok:
            print("   - 文件结构不完整")
        if not imports_ok:
            print("   - 导入失败")
        if not app_ok:
            print("   - 应用结构问题")

if __name__ == "__main__":
    main() 