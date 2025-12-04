"""
Web部署配置 - 部署到longneckedrabbit.dpdns.org
"""

import os
import subprocess
import sys
from pathlib import Path

def create_requirements_file():
    """创建requirements.txt文件"""
    requirements = """
Flask==2.3.3
numpy==1.24.3
tensorflow==2.13.0
torch==2.0.1
torchvision==0.15.2
gunicorn==21.2.0
waitress==2.1.2
"""
    with open('requirements.txt', 'w', encoding='utf-8') as f:
        f.write(requirements.strip())
    print("requirements.txt 文件已创建")

def create_docker_config():
    """创建Docker配置文件"""
    dockerfile = """
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "app:app"]
"""
    with open('Dockerfile', 'w', encoding='utf-8') as f:
        f.write(dockerfile.strip())
    print("Dockerfile 文件已创建")

def create_docker_compose():
    """创建Docker Compose配置"""
    compose_content = """
version: '3.8'

services:
  chess-ai:
    build: .
    ports:
      - "5000:5000"
    environment:
      - FLASK_ENV=production
      - SECRET_KEY=your-secret-key-here
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    restart: unless-stopped
    container_name: chess-ai-app

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - chess-ai
    restart: unless-stopped
    container_name: chess-ai-nginx
"""
    with open('docker-compose.yml', 'w', encoding='utf-8') as f:
        f.write(compose_content.strip())
    print("docker-compose.yml 文件已创建")

def create_nginx_config():
    """创建Nginx配置"""
    nginx_config = """
events {
    worker_connections 1024;
}

http {
    upstream chess_ai_backend {
        server chess-ai:5000;
    }

    # HTTP重定向到HTTPS
    server {
        listen 80;
        server_name longneckedrabbit.dpdns.org www.longneckedrabbit.dpdns.org;
        return 301 https://$server_name$request_uri;
    }

    # HTTPS配置
    server {
        listen 443 ssl http2;
        server_name longneckedrabbit.dpdns.org www.longneckedrabbit.dpdns.org;

        # SSL证书配置（需要替换为实际证书路径）
        ssl_certificate /etc/nginx/ssl/cert.pem;
        ssl_certificate_key /etc/nginx/ssl/key.pem;

        # SSL安全配置
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers HIGH:!aNULL:!MD5;
        ssl_prefer_server_ciphers on;

        # 静态文件处理
        location /static/ {
            alias /app/static/;
            expires 1y;
            add_header Cache-Control "public, immutable";
        }

        # API请求代理
        location / {
            proxy_pass http://chess_ai_backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;

            # WebSocket支持
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
        }

        # 启用Gzip压缩
        gzip on;
        gzip_types
            text/plain
            text/css
            text/js
            text/xml
            text/javascript
            application/javascript
            application/json
            application/xml+rss;
    }
}
"""
    with open('nginx.conf', 'w', encoding='utf-8') as f:
        f.write(nginx_config.strip())
    print("nginx.conf 文件已创建")

def create_ssl_directory():
    """创建SSL证书目录和临时证书"""
    os.makedirs('ssl', exist_ok=True)

    # 创建自签名证书（仅用于测试，生产环境需要真实证书）
    subprocess.run([
        'openssl', 'req', '-x509', '-newkey', 'rsa:4096', '-keyout', 'ssl/key.pem',
        '-out', 'ssl/cert.pem', '-days', '365', '-nodes',
        '-subj', '/C=CN/ST=Beijing/L=Beijing/O=ChessAI/CN=longneckedrabbit.dpdns.org'
    ], check=True, capture_output=True)
    print("SSL证书目录已创建，临时自签名证书已生成")

def create_production_config():
    """创建生产环境配置"""
    config = """
# 生产环境配置
FLASK_ENV=production
SECRET_KEY=your-ultra-secure-secret-key-here-change-this-in-production
LOG_LEVEL=INFO
DATA_DIR=./data
LOG_DIR=./logs
HOST=0.0.0.0
PORT=5000
"""
    with open('.env.production', 'w', encoding='utf-8') as f:
        f.write(config.strip())
    print(".env.production 文件已创建")

def create_deploy_script():
    """创建部署脚本"""
    deploy_script = """#!/bin/bash
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
"""
    with open('deploy.sh', 'w', encoding='utf-8') as f:
        f.write(deploy_script.strip())

    # 设置执行权限
    os.chmod('deploy.sh', 0o755)
    print("deploy.sh 部署脚本已创建")

def create_systemd_service():
    """创建SystemD服务文件"""
    service_content = f"""[Unit]
Description=Chess AI Learning System
After=network.target

[Service]
Type=exec
User=www-data
Group=www-data
WorkingDirectory={os.getcwd()}
Environment=FLASK_ENV=production
Environment=SECRET_KEY=your-secret-key-here
ExecStart=/usr/local/bin/gunicorn --bind 0.0.0.0:5000 --workers 4 app:app
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
"""
    with open('chess-ai.service', 'w', encoding='utf-8') as f:
        f.write(service_content.strip())
    print("chess-ai.service SystemD服务文件已创建")

def main():
    """主函数"""
    print("=== 象棋AI学习系统 - Web部署配置 ===")
    print("域名: longneckedrabbit.dpdns.org")
    print("")

    # 创建部署文件
    create_requirements_file()
    create_docker_config()
    create_docker_compose()
    create_nginx_config()
    create_ssl_directory()
    create_production_config()
    create_deploy_script()
    create_systemd_service()

    print("")
    print("=== 部署配置完成 ===")
    print("")
    print("部署方式选择:")
    print("1. Docker Compose (推荐)")
    print("2. SystemD + Nginx")
    print("3. 传统Gunicorn")
    print("")
    print("重要提醒:")
    print("- 生产环境请替换SSL证书为Let's Encrypt或购买的真实证书")
    print("- 请修改.env.production中的SECRET_KEY")
    print("- 确保域名DNS指向服务器IP地址")
    print("")
    print("快速启动:")
    print("./deploy.sh")

if __name__ == "__main__":
    main()