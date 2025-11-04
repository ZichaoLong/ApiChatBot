"""
ApiChatBot 配置模块

提供配置模板和加载工具，支持多配置场景。

核心设计:
    - 不在模块导入时自动加载环境变量
    - 提供工具函数按需加载配置
    - 支持多个模块使用不同配置

环境变量支持:
    - API Keys: OPENAI_API_KEY, GEMINI_API_KEY, ANTHROPIC_API_KEY 等
    - Base URLs: OPENAI_BASE_URL, GOOGLE_BASE_URL, ANTHROPIC_BASE_URL 等
    - Base URL 未配置时使用 DEFAULT_BASE_URLS 中的默认值

使用方式:
    # 方式1: 使用默认配置（从环境变量）
    from ApiChatBot.config import load_default_config
    config = load_default_config()
    chatbot = ChatBot(
        api_key=config['API_KEYS']['openai'],
        base_url=config['BASE_URLS']['openai']
    )

    # 方式2: 使用自定义配置文件
    from ApiChatBot.config import load_config_from_env
    config = load_config_from_env('.env.production')

    # 方式3: 单独获取配置项
    from ApiChatBot.config import get_api_key, get_base_url
    api_key = get_api_key('openai')
    base_url = get_base_url('openai')

    # 方式4: 直接传参（最灵活）
    chatbot = ChatBot(api_key='sk-xxx', base_url='https://...')
"""

import os
from typing import Dict, Optional
from dotenv import load_dotenv

__all__ = [
    'COMPANIES',
    'INTERFACETYPE',
    'MODELS',
    'DEFAULT_BASE_URLS',
    'load_default_config',
    'load_config_from_env',
    'get_api_key',
    'get_base_url',
]

# ============================================================
# 配置模板（常量）
# ============================================================

# 支持的LLM提供商列表
COMPANIES = [
    'openai',
    'google',
    'anthropic',
    'moonshot',
    'aliyun',
    'deepseek',
    'openrouter'
]

# 提供商接口类型映射
INTERFACETYPE = {
    'openai': 'openai',
    'google': 'google',
    'anthropic': 'anthropic',
    'moonshot': 'openai',      # 使用OpenAI兼容接口
    'aliyun': 'openai',         # 使用OpenAI兼容接口
    'deepseek': 'openai',       # 使用OpenAI兼容接口
    'openrouter': 'openai',     # 使用OpenAI兼容接口
}

# 各提供商的建议模型列表（用于测试）
# 注意：这不是全部可用模型，实际使用时可以指定任意该提供商支持的模型
MODELS = {
    'openai': [
        'gpt-4o',
        # 'gemini-2.5-flash',
        # 'claude-sonnet-4-20250514'
    ],
    'google': [
        'gemini-2.5-flash',
        # 'gemini-2.5-pro'
    ],
    'anthropic': [
        'claude-sonnet-4-20250514'
    ],
    'moonshot': [
        'kimi-k2-0711-preview',
        # 'kimi-k2-turbo-preview'
    ],
    'aliyun': [
        'qwen-plus'
    ],
    'deepseek': [
        # 'deepseek-reasoner',
        'deepseek-chat'
    ],
    'openrouter': [
        'openai/gpt-4o',
        # 'google/gemini-2.5-flash',
        # 'anthropic/claude-sonnet-4'
    ],
}

# 默认 API 基础 URL
DEFAULT_BASE_URLS = {
    'openai': 'https://api.openai.com/v1',
    'google': 'https://generativelanguage.googleapis.com',
    'anthropic': 'https://api.anthropic.com/v1/messages',
    'moonshot': 'https://api.moonshot.cn/v1',
    'aliyun': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
    'deepseek': 'https://api.deepseek.com',
    'openrouter': 'https://openrouter.ai/api/v1',
}

# 环境变量名映射
_ENV_VAR_KEYS = {
    'openai': 'OPENAI_API_KEY',
    'google': 'GEMINI_API_KEY',
    'anthropic': 'ANTHROPIC_API_KEY',
    'moonshot': 'MOONSHOT_API_KEY',
    'aliyun': 'ALIYUN_API_KEY',
    'deepseek': 'DEEPSEEK_API_KEY',
    'openrouter': 'OPENROUTER_API_KEY',
}

_ENV_VAR_BASE_URLS = {
    'openai': 'OPENAI_BASE_URL',
    'google': 'GOOGLE_BASE_URL',
    'anthropic': 'ANTHROPIC_BASE_URL',
    'moonshot': 'MOONSHOT_BASE_URL',
    'aliyun': 'ALIYUN_BASE_URL',
    'deepseek': 'DEEPSEEK_BASE_URL',
    'openrouter': 'OPENROUTER_BASE_URL',
}

# ============================================================
# 配置加载函数（按需使用）
# ============================================================

def load_config_from_env(env_file: Optional[str] = None) -> Dict:
    """
    从环境变量或 .env 文件加载配置

    Args:
        env_file: .env 文件路径（可选）
                 None 表示自动向上查找 .env 文件

    Returns:
        配置字典，包含 API_KEYS 和 BASE_URLS
    """
    if env_file:
        load_dotenv(dotenv_path=env_file, override=True)
    else:
        load_dotenv()  # 自动向上查找 .env

    api_keys = {
        company: os.getenv(_ENV_VAR_KEYS[company], '')
        for company in COMPANIES
    }

    base_urls = {
        company: os.getenv(_ENV_VAR_BASE_URLS[company], DEFAULT_BASE_URLS[company])
        for company in COMPANIES
    }

    return {
        'API_KEYS': api_keys,
        'BASE_URLS': base_urls,
        'COMPANIES': COMPANIES,
        'INTERFACETYPE': INTERFACETYPE,
        'MODELS': MODELS,
    }


def load_default_config() -> Dict:
    """
    加载默认配置（从环境变量）

    等同于 load_config_from_env(None)
    """
    return load_config_from_env()


def get_api_key(company: str, env_file: Optional[str] = None) -> str:
    """
    获取指定提供商的 API 密钥

    Args:
        company: 提供商名称（如 'openai', 'google'）
        env_file: 可选的 .env 文件路径

    Returns:
        API 密钥字符串
    """
    if env_file:
        load_dotenv(dotenv_path=env_file, override=True)
    else:
        load_dotenv()

    return os.getenv(_ENV_VAR_KEYS.get(company, ''), '')


def get_base_url(company: str, env_file: Optional[str] = None) -> str:
    """
    获取指定提供商的 base URL

    Args:
        company: 提供商名称（如 'openai', 'google'）
        env_file: 可选的 .env 文件路径

    Returns:
        Base URL 字符串
    """
    if env_file:
        load_dotenv(dotenv_path=env_file, override=True)
    else:
        load_dotenv()

    return os.getenv(
        _ENV_VAR_BASE_URLS.get(company, ''),
        DEFAULT_BASE_URLS.get(company, '')
    )
