"""
ApiChatBot 测试配置

从 config.py 加载默认配置，并提供配置检查工具。
"""

import sys
from pathlib import Path

# 添加父目录到路径，以便导入 ApiChatBot.config
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ApiChatBot.config import load_default_config, COMPANIES

# 加载默认配置
_config = load_default_config()

# 导出配置项（供测试使用）
API_KEYS = _config['API_KEYS']
BASE_URLS = _config['BASE_URLS']
INTERFACETYPE = _config['INTERFACETYPE']
MODELS = _config['MODELS']

__all__ = ['COMPANIES', 'INTERFACETYPE', 'MODELS', 'BASE_URLS', 'API_KEYS', 'check_config']

# 验证配置
def check_config():
    """检查配置是否完整，用于测试前验证"""
    missing = []
    for company in COMPANIES:
        if not API_KEYS.get(company):
            missing.append(company)

    if missing:
        print(f"⚠️  警告: 以下提供商缺少API密钥配置: {', '.join(missing)}")
        print(f"   这些提供商的测试将被跳过。")
        print(f"   如需测试这些提供商，请在 .env 文件中配置对应的API密钥。")

    return len(missing) == 0

if __name__ == '__main__':
    # 运行配置检查
    print("ApiChatBot 测试配置检查\n")
    print(f"支持的提供商: {', '.join(COMPANIES)}\n")

    for company in COMPANIES:
        api_key = API_KEYS.get(company, '')
        status = '✓' if api_key else '✗'
        masked_key = f"{api_key[:8]}..." if api_key else "未配置"
        print(f"{status} {company:12s}: {masked_key}")

    print()
    check_config()
