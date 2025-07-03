"""
BlockFW 后端 API 服务
第三阶段：后端API开发
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import uvicorn
import os
import sys

# 注释掉复杂的路径添加，可能导致问题
# project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append(project_root)

# 导入路由
from api import detection

# 创建FastAPI应用
app = FastAPI(
    title="BlockFW API",
    description="区块链防火墙 - AI检测与智能合约集成API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中应该设置具体的域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """根路径 - API信息"""
    return {
        "message": "BlockFW API 服务",
        "version": "1.0.0",
        "stage": "第三阶段 - 后端API开发",
        "timestamp": datetime.now().isoformat(),
        "status": "运行中"
    }

@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "api": "running",
            "ai_detection": "ready",
            "blockchain": "ready"
        }
    }

@app.get("/api/info")
async def api_info():
    """API信息接口"""
    return {
        "project": "BlockFW",
        "stage": "第三阶段",
        "components": {
            "ai_detection": "XGBoost模型已准备",
            "smart_contract": "FirewallRules.sol已部署",
            "backend_api": "FastAPI开发中"
        },
        "endpoints": {
            "health": "/health",
            "info": "/api/info",
            "detection": "/api/detection/* (已实现)",
            "blockchain": "/api/blockchain/* (开发中)"
        }
    }

# 包含路由
app.include_router(detection.router)

if __name__ == "__main__":
    print("🚀 启动 BlockFW 后端API服务...")
    print("📚 API文档地址: http://127.0.0.1:8000/docs")
    print("🔍 健康检查: http://127.0.0.1:8000/health")
    uvicorn.run(app, host="127.0.0.1", port=8000) 