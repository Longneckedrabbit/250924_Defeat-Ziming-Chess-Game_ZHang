#!/bin/bash
# 部署脚本

echo "开始部署象棋AI学习系统..."

# 创建必要的目录
mkdir -p data logs ssl static

# 检查Docker和Docker Compose
if ! command -v docker &> /dev/null; then
    echo "Docker未安装，请先安装Docker"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "Docker Compose未安装，请先安装Docker Compose"
    exit 1
fi

# 构建并启动服务
echo "构建Docker镜像..."
docker-compose build

echo "启动服务..."
docker-compose up -d

echo "部署完成！"
echo "请访问: https://longneckedrabbit.dpdns.org"
echo "查看日志: docker-compose logs -f"