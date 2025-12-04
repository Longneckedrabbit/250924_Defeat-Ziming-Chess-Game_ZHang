"""
象棋AI学习系统启动脚本
"""

import os
import sys
import logging
from app import app, logger

def check_dependencies():
    """检查依赖项"""
    required_packages = ['flask', 'numpy']
    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print(f"缺少必要的Python包: {missing_packages}")
        print("请运行: pip install -r requirements.txt")
        sys.exit(1)

def create_directories():
    """创建必要的目录"""
    directories = ['logs', 'data', 'static', 'templates']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"创建目录: {directory}")

def setup_logging():
    """设置日志"""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.FileHandler('logs/chess_ai.log'),
            logging.StreamHandler()
        ]
    )

def main():
    """主函数"""
    print("=== 象棋AI学习系统 ===")
    print("专门学习张子鸣棋路的智能象棋对手")
    print("")

    # 检查依赖
    check_dependencies()

    # 创建目录
    create_directories()

    # 设置日志
    setup_logging()

    # 配置信息
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_ENV') == 'development'

    print(f"启动服务器...")
    print(f"地址: http://{host}:{port}")
    print(f"调试模式: {'开启' if debug else '关闭'}")
    print(f"日志文件: logs/chess_ai.log")
    print("")
    print("功能特性:")
    print("- 完整的中国象棋规则实现")
    print("- AI从每局游戏中学习")
    print("- 深度学习模型分析棋路")
    print("- 详细的日志记录系统")
    print("- 模式识别和预测")
    print("- 学习进度追踪")
    print("")

    try:
        app.run(host=host, port=port, debug=debug)
    except KeyboardInterrupt:
        print("\n服务器已停止")
        sys.exit(0)
    except Exception as e:
        print(f"服务器启动失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()