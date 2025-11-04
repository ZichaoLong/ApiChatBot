"""
ApiChatBot - 多提供商LLM统一接口封装

本模块提供了统一的聊天机器人API封装，支持多个LLM提供商：
- OpenAI (gpt-4o, etc.)
- Google Gemini (gemini-2.5-flash, etc.)
- Anthropic Claude (claude-sonnet-4, etc.)

支持两种实现方式：
1. SDK模式：使用官方SDK（openai、google.genai、anthropic）
2. HTTP模式：使用原始httpx请求（仅支持OpenAI接口）

主要功能：
- 统一的Chat接口（同步/异步）
- 流式和完整响应支持
- 实时显示和思考过程可见性
- 自定义回调函数
- SSL验证配置（适配代理环境）
"""

from .Client import *
from ._BaseChatBot import *
from ._OpenAI import *
from ._Google import *
from ._Anthropic import *

__all__ = ['ChatBotFunc', 'ResponseDict']


def ChatBotFunc(interfacetype: str, use_sdk: bool = True):
    """
    聊天机器人工厂函数

    根据接口类型和SDK偏好返回对应的聊天机器人类。

    参数:
        interfacetype: 接口类型，可选值：'openai', 'google', 'anthropic'
        use_sdk: 是否使用官方SDK，默认True
                 - True: 使用官方SDK实现（OpenAISDKChatBot等）
                 - False: 使用httpx原始请求（仅支持OpenAI接口）

    返回:
        聊天机器人类（未实例化）

    异常:
        AssertionError: 当use_sdk=False但interfacetype不是'openai'时抛出

    示例:
        >>> # 使用OpenAI SDK
        >>> chatbot_class = ChatBotFunc(interfacetype='openai', use_sdk=True)
        >>> chatbot = chatbot_class(api_key='sk-xxx', base_url='https://api.openai.com/v1')
        >>>
        >>> # 使用httpx模式（适用于OpenAI兼容API）
        >>> chatbot_class = ChatBotFunc(interfacetype='openai', use_sdk=False)
        >>> chatbot = chatbot_class(api_key='xxx', base_url='https://api.example.com/v1')
    """
    if use_sdk and interfacetype == 'openai':
        return OpenAISDKChatBot
    elif use_sdk and interfacetype == 'google':
        return GoogleSDKChatBot
    elif use_sdk and interfacetype == 'anthropic':
        return AnthropicSDKChatBot
    else:
        assert interfacetype == 'openai', "Httpx ChatBot 仅支持 openai interface"
        return OpenAIHttpxChatBot
