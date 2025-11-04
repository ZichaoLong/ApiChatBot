"""
HTTP客户端和SDK客户端工厂模块

本模块提供了创建HTTP客户端和各提供商SDK客户端的工厂函数。
支持同步/异步模式、SSL验证配置、超时设置等。

主要功能:
- HttpxClient: 创建原始httpx客户端（用于直接HTTP请求）
- ApiSDKClient: 创建特定提供商的SDK客户端（OpenAI、Google、Anthropic）
"""

import httpx
import openai
import google
import anthropic
from openai import OpenAI, AsyncOpenAI
from google import genai
from anthropic import Anthropic, AsyncAnthropic

__all__ = ['HttpxClient', 'ApiSDKClient']


def HttpxClient(is_async: bool, is_ssl_verify: bool = False, timeout: int = 300):
    """
    创建httpx HTTP客户端

    参数:
        is_async: 是否创建异步客户端
        is_ssl_verify: 是否启用SSL证书验证，默认False
                       在公司代理环境中通常需要设为False
        timeout: 请求超时时间（秒），默认300秒

    返回:
        httpx.Client 或 httpx.AsyncClient 实例

    示例:
        >>> # 创建同步客户端
        >>> client = HttpxClient(is_async=False, is_ssl_verify=False)
        >>>
        >>> # 创建异步客户端
        >>> async_client = HttpxClient(is_async=True, is_ssl_verify=True, timeout=60)
    """
    client_func = httpx.AsyncClient if is_async else httpx.Client
    return client_func(verify=is_ssl_verify, timeout=timeout)

def ApiSDKClient(
    api_key: str,
    base_url: str,
    *,
    interfacetype: str = 'openai',
    is_async: bool = True,
    is_ssl_verify: bool = False
):
    """
    创建特定提供商的SDK客户端

    根据接口类型创建对应的官方SDK客户端实例。
    支持OpenAI、Google Gemini、Anthropic Claude三种SDK。

    参数:
        api_key: API密钥
        base_url: API基础URL
        interfacetype: 接口类型，可选值：'openai', 'google', 'anthropic'
        is_async: 是否创建异步客户端，默认True
                  注意：Google的genai.Client内部处理异步，此参数对Google无效
        is_ssl_verify: 是否启用SSL证书验证，默认False
                       注意：Google SDK不支持外部配置SSL，需要修改源码（见下方注释）

    返回:
        对应提供商的SDK客户端实例:
        - OpenAI: OpenAI 或 AsyncOpenAI
        - Google: genai.Client（内部处理同步/异步）
        - Anthropic: Anthropic 或 AsyncAnthropic

    异常:
        ValueError: 如果 API key 为空或无效
        其他SDK相关的认证或连接异常

    示例:
        >>> # 创建OpenAI SDK客户端
        >>> client = ApiSDKClient(
        ...     api_key='sk-xxx',
        ...     base_url='https://api.openai.com/v1',
        ...     interfacetype='openai',
        ...     is_async=False
        ... )
        >>>
        >>> # 创建Google Gemini SDK客户端
        >>> client = ApiSDKClient(
        ...     api_key='xxx',
        ...     base_url='https://generativelanguage.googleapis.com',
        ...     interfacetype='google'
        ... )

    Google SDK的SSL验证配置说明:
        Google genai.Client不支持外部传入http_client，内部创建的http_client
        会强制设定ssl verify=True。如需禁用SSL验证，可按以下步骤修改源码：

        Step 1: 找到文件
            import google.genai._api_client as api_client
            print(api_client.__file__)

        Step 2: 修改BaseApiClient类的初始化
            在以下两行代码前：
                self._httpx_client = SyncHttpxClient(**client_args)
                self._async_httpx_client = AsyncHttpxClient(**async_client_args)
            添加：
                client_args['verify'] = async_client_args['verify'] = False

        Step 3: 禁用aiohttp（可选，仅异步模式需要）
            在 try: import aiohttp 下（约第70行）添加：
                has_aiohttp = False
            以强制使用httpx，使Step 2的配置在异步情形也生效
    """
    # 验证 API key
    if not api_key or not api_key.strip():
        raise ValueError(
            f"API key cannot be empty for interfacetype: {interfacetype}. "
            f"Please check your .env file or pass api_key explicitly."
        )

    # 根据接口类型选择对应的客户端类
    match interfacetype:
        case 'openai':
            client_func = AsyncOpenAI if is_async else OpenAI
        case 'google':
            client_func = genai.Client  # Google内部处理同步/异步
        case 'anthropic':
            client_func = AsyncAnthropic if is_async else Anthropic

    # 根据接口类型使用不同的初始化方式
    match interfacetype:
        case 'google':
            # Google SDK不支持外部传入http_client
            # 使用http_options配置base_url
            client = client_func(
                api_key=api_key,
                http_options={'base_url': base_url}
            )
        case _:
            # OpenAI和Anthropic支持外部传入http_client
            # 可以配置SSL验证和超时等参数
            client = client_func(
                api_key=api_key,
                base_url=base_url,
                http_client=HttpxClient(is_async, is_ssl_verify, timeout=300)
            )

    return client

