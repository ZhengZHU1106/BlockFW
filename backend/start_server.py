#!/usr/bin/env python3
"""
BlockFW 后端服务启动脚本
用于测试和验证第三阶段基础架构
"""

import subprocess
import sys
import os

def check_dependencies():
    """检查依赖是否安装"""
    print("🔍 检查依赖包...")
    
    required_packages = [
        "fastapi",
        "uvicorn", 
        "pydantic"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✅ {package} - 已安装")
        except ImportError:
            print(f"  ❌ {package} - 未安装")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n📦 需要安装以下依赖包:")
        print("cd backend")
        print("pip install -r requirements.txt")
        return False
    
    print("✅ 所有依赖包已安装")
    return True

def test_import():
    """测试导入是否正常"""
    print("\n🧪 测试模块导入...")
    
    try:
        from app import app
        print("  ✅ FastAPI应用导入成功")
        
        try:
            from models.schemas import DetectionRequest, DetectionResult
            print("  ✅ 数据模型导入成功")
        except ImportError as e:
            print(f"  ⚠️ 数据模型导入警告: {e}")
        
        return True
        
    except ImportError as e:
        print(f"  ❌ 应用导入失败: {e}")
        return False

def start_server():
    """启动FastAPI服务器"""
    print("\n🚀 启动FastAPI服务器...")
    print("📚 API文档: http://localhost:8000/docs")
    print("🔍 健康检查: http://localhost:8000/health")
    print("ℹ️ API信息: http://localhost:8000/api/info")
    print("\n按 Ctrl+C 停止服务器\n")
    
    try:
        # 直接运行app.py
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        subprocess.run([sys.executable, "app.py"])
    except KeyboardInterrupt:
        print("\n🛑 服务器已停止")
    except Exception as e:
        print(f"❌ 启动失败: {e}")

if __name__ == "__main__":
    print("="*60)
    print("🎯 BlockFW 第三阶段 - 后端API验证")
    print("="*60)
    
    # 检查依赖
    if not check_dependencies():
        print("\n请先安装依赖后再运行此脚本")
        sys.exit(1)
    
    # 测试导入
    if not test_import():
        print("\n模块导入失败，请检查代码")
        sys.exit(1)
    
    print("\n✅ 基础架构验证通过！")
    
    # 询问是否启动服务器
    response = input("\n是否启动API服务器？(y/n): ").lower().strip()
    if response in ['y', 'yes', '是']:
        start_server()
    else:
        print("🏁 验证完成，未启动服务器") 