#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ApiChatBot 综合测试脚本

支持多提供商（OpenAI/Google/Anthropic等）和灵活的测试配置：
- 模式：SDK / Httpx（仅OpenAI接口）
- 执行：同步 / 异步
- 响应：流式 / 完整

运行: python test_providers.py --help
"""

#%%
# 启用运行时类型检查
from typeguard import install_import_hook
install_import_hook('ApiChatBot')

import sys
from pathlib import Path

# 添加父目录到路径，以便导入 ApiChatBot
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import httpx
import asyncio
import random
import openai
import google
import anthropic
from typing import Optional, Union, List, Dict, Any
from openai import OpenAI, AsyncOpenAI
from openai.types.chat import ChatCompletion
from google import genai
from google.genai.types import GenerateContentResponse
from anthropic import Anthropic, AsyncAnthropic
from anthropic.types import Message
from _config import *
import ApiChatBot

#%%
# TODO: 待扩展功能 - Tool calls, JSON输出
# 多轮对话默认配置
DEFAULT_CONVERSATION = [
    "你好，请自我介绍一下。",
    "帮我生成一个Python函数来计算斐波那契数列。",
]
DEFAULT_CONVERSATION = [
    "你好，请自我介绍一下。",
    "请以'我是挖掘机'为题写一首儿歌歌词。",
]
DEFAULT_CONVERSATION = ["你好，你是谁？", "好的。"]

def is_api_key_available(company: str) -> bool:
    """检查提供商的 API key 是否已配置"""
    api_key = API_KEYS.get(company, '')
    return bool(api_key and api_key.strip())


def get_companies_by_interfacetype(interfacetype: str) -> List[str]:
    """根据接口类型获取对应的公司列表"""
    return [company for company, itype in INTERFACETYPE.items() if itype == interfacetype]


def test_conversation(
    company: str,
    model: str,
    is_async: bool,
    is_streamed: bool,
    is_ssl_verify: bool,
    use_sdk: bool,
    conversation: List[str],
    system_instruction: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    执行多轮对话测试

    Returns:
        List[Dict[str, Any]]: 所有轮次的响应列表

    注意: Httpx模式仅支持OpenAI接口类型，其他接口返回空列表
    """
    base_url = BASE_URLS[company]
    api_key = API_KEYS[company]
    interfacetype = INTERFACETYPE[company]

    # Httpx模式仅支持OpenAI接口
    if not use_sdk and interfacetype != 'openai':
        return []

    # 打印测试配置
    print("company:", company, "\nmodel:", model,
          "\ninterfacetype:", interfacetype,
          "\nbase url:", base_url,
          "\nis_async:", is_async,
          "\nis_streamed:", is_streamed,
          "\nuse_sdk:", use_sdk,
          "\nconversation_turns:", len(conversation), end='\n', flush=True)

    # 创建聊天机器人实例
    chatbotfunc = ApiChatBot.ChatBotFunc(interfacetype, use_sdk)
    chatbot = chatbotfunc(
        api_key, base_url,
        is_async=is_async,
        is_ssl_verify=is_ssl_verify,
        realtime_display=True,
        show_thinking=True
    )

    # 执行测试（统一在async函数中处理，避免代码重复）
    responses = asyncio.run(_async_conversation(
        chatbot, model, conversation, is_async, is_streamed, system_instruction
    ))
    return responses


async def _async_conversation(
    chatbot,
    model: str,
    conversation: List[str],
    is_async: bool,
    is_streamed: bool,
    system_instruction: Optional[str]
) -> List[Dict[str, Any]]:
    """统一的多轮对话处理函数：在单一事件循环中完成所有操作"""
    messages = []
    responses = []

    # 处理特殊配置（如通过OpenAI接口访问Gemini时的thinking_config）
    if isinstance(chatbot, ApiChatBot.BaseSDKChatBot) and chatbot.interfacetype == 'openai' and model.startswith('gemini'):
        extra_body = {
            'extra_body': {
                'google': {
                    'thinking_config': {
                        'thinking_budget': -1,
                        'include_thoughts': True
                    }
                }
            }
        }
        kwargs = {'extra_body': extra_body}
    else:
        kwargs = {}

    for turn, user_message in enumerate(conversation, 1):
        print(f"\n{'='*60}")
        print(f"第 {turn} 轮对话")
        print(f"{'='*60}")
        print(f"用户: {user_message}\n")

        # 添加用户消息
        messages.append({'role': 'user', 'content': user_message})

        # 执行请求（根据 is_async 选择方法）
        if is_async:
            response = await chatbot.AsyncChat(
                model=model, messages=messages,
                stream=is_streamed, system_instruction=system_instruction,
                **kwargs
            )
        else:
            # 在 async 函数中可以调用同步方法
            response = chatbot.Chat(
                model=model, messages=messages,
                stream=is_streamed, system_instruction=system_instruction,
                **kwargs
            )

        # 保存响应
        responses.append(response)
        messages.append(response)

    print(f"\n{'='*60}")
    print(f"✓ 多轮对话测试完成 (共 {turn} 轮)")
    print(f"{'='*60}\n")

    # 清理资源
    if is_async:
        await chatbot.aclose()
    else:
        chatbot.close()

    return responses


def test_company(
    company: str,
    is_ssl_verify: bool = False,
    model: Optional[str] = None,
    mode: str = 'both',
    execution: str = 'both',
    response: str = 'both',
    conversation: Optional[List[str]] = None,
    system_instruction: Optional[str] = None
) -> int:
    """
    测试单个公司的配置组合（多轮对话）

    过滤器:
      - model: 指定任意模型名称，或None（使用建议列表）
      - mode: sdk/httpx/both
      - execution: sync/async/both
      - response: stream/complete/both
      - conversation: 对话列表，或None（使用默认对话）
      - system_instruction: 系统指令

    Returns:
      int: 完成的测试配置数量
    """
    # 检查 API key
    if not is_api_key_available(company):
        print(f"\n{'='*60}")
        print(f"⚠️  跳过 {company}: 未配置 API Key")
        print(f"{'='*60}\n")
        return 0

    # 使用默认对话或自定义对话
    if conversation is None:
        conversation = DEFAULT_CONVERSATION

    print(f"\n{'='*60}")
    print(f"开始测试 {company.upper()}")
    print(f"{'='*60}\n")

    interfacetype = INTERFACETYPE[company]

    # 1. 确定要测试的 use_sdk 选项
    if mode == 'sdk':
        use_sdk_options = [True]
    elif mode == 'httpx':
        if interfacetype != 'openai':
            print(f"⚠️  {company} 不支持 Httpx 模式（仅OpenAI接口支持）")
            return 0
        use_sdk_options = [False]
    else:  # mode == 'both'
        if interfacetype == 'openai':
            use_sdk_options = [False, True]  # OpenAI接口支持 SDK 和 Httpx
        else:
            use_sdk_options = [True]  # 其他接口只支持 SDK

    # 2. 确定要测试的模型列表
    if model:
        # 指定模型时，允许使用任意模型（不限于建议列表）
        model_options = [model]
        if model not in MODELS[company]:
            print(f"ℹ️  使用自定义模型 '{model}'（不在建议列表中）")
    else:
        # 未指定模型时，使用建议模型列表
        model_options = MODELS[company]

    # 3. 确定要测试的 is_async 选项
    if execution == 'sync':
        is_async_options = [False]
    elif execution == 'async':
        is_async_options = [True]
    else:  # execution == 'both'
        is_async_options = [True, False]

    # 4. 确定要测试的 is_streamed 选项
    if response == 'stream':
        is_streamed_options = [True]
    elif response == 'complete':
        is_streamed_options = [False]
    else:  # response == 'both'
        is_streamed_options = [True, False]

    # 执行所有组合的测试
    test_count = 0
    for use_sdk in use_sdk_options:
        for test_model in model_options:
            for is_async in is_async_options:
                for is_streamed in is_streamed_options:
                    test_count += 1
                    print(f"\n--- Test {test_count} ---")
                    responses = test_conversation(
                        company, test_model, is_async, is_streamed,
                        is_ssl_verify, use_sdk, conversation, system_instruction
                    )
                    print("✓ 测试成功\n")

    print(f"\n{'='*60}")
    print(f"{company.upper()} 测试完成 (共 {test_count} 个配置)")
    print(f"{'='*60}\n")

    return test_count


def test_interfacetype(
    interfacetype: str,
    is_ssl_verify: bool = False,
    model: Optional[str] = None,
    mode: str = 'both',
    execution: str = 'both',
    response: str = 'both',
    conversation: Optional[List[str]] = None,
    system_instruction: Optional[str] = None
) -> int:
    """
    测试指定接口类型下的所有公司，支持过滤器（多轮对话）

    Returns:
      int: 完成的测试配置总数
    """
    companies = get_companies_by_interfacetype(interfacetype)

    print(f"\n{'#'*60}")
    print(f"开始测试接口类型: {interfacetype.upper()}")
    print(f"包含公司: {', '.join(companies)}")
    print(f"{'#'*60}\n")

    total_count = 0
    for company in companies:
        count = test_company(company, is_ssl_verify, model, mode, execution, response,
                           conversation, system_instruction)
        total_count += count

    print(f"\n{'#'*60}")
    print(f"接口类型 {interfacetype.upper()} 测试完成 (共 {total_count} 个配置)")
    print(f"{'#'*60}\n")

    return total_count


def test_all(
    is_ssl_verify: bool = False,
    model: Optional[str] = None,
    mode: str = 'both',
    execution: str = 'both',
    response: str = 'both',
    conversation: Optional[List[str]] = None,
    system_instruction: Optional[str] = None
) -> int:
    """
    测试所有已配置的提供商，支持过滤器，自动跳过未配置的（多轮对话）

    Returns:
      int: 完成的测试配置总数
    """
    print(f"\n{'#'*60}")
    print(f"开始测试所有提供商")
    print(f"{'#'*60}\n")

    # 先显示配置状态
    print("配置检查:")
    for company in COMPANIES:
        status = '✓' if is_api_key_available(company) else '✗'
        print(f"  {status} {company}")
    print()

    # 按 interfacetype 分组测试
    total_count = 0
    for interfacetype in ['openai', 'google', 'anthropic']:
        count = test_interfacetype(interfacetype, is_ssl_verify, model, mode, execution, response,
                                  conversation, system_instruction)
        total_count += count

    print(f"\n{'#'*60}")
    print(f"所有测试完成 (共 {total_count} 个配置)")
    print(f"{'#'*60}\n")

    return total_count

#%%
# ============================================================
# 命令行执行入口
# ============================================================

if __name__ == '__main__':
    import argparse

    # 动态生成支持的接口类型列表
    supported_interfacetypes = sorted(set(INTERFACETYPE.values()))

    # 构建建议模型列表显示
    models_display = "\n建议模型列表（若使用 --model 则可指定非建议模型）:\n"
    for company in COMPANIES:
        models_list = ', '.join(MODELS.get(company, []))
        models_display += f"  {company:12s}: {models_list}\n"

    # 构建帮助信息
    epilog = f"""
过滤器层级结构:
  ┌─ 测试范围（互斥）───────────────────────┐
  │ • 默认: 测试所有提供商（自动跳过未配置）│
  │ • -i/--interfacetype: 限定到某接口类型  │
  │ • -c/--company: 限定到单个公司          │
  └─────────────────────────────────────────┘
        ↓
  ┌─ 测试维度（可组合）─────────────────────┐
  │ • --model: 指定模型（默认使用建议列表） │
  │ • --mode: sdk/httpx/both（默认both）    │
  │ • --execution: sync/async/both          │
  │ • --response: stream/complete/both      │
  └─────────────────────────────────────────┘

基础用法:
  python test_providers.py                    # 测试所有提供商（默认）
  python test_providers.py -i openai          # 仅测试 OpenAI 接口类型
  python test_providers.py -c google          # 仅测试 Google 公司

组合过滤器:
  python test_providers.py -r stream          # 所有提供商，仅流式响应
  python test_providers.py -i openai --m httpx  # OpenAI 接口，仅 Httpx 模式
  python test_providers.py -c openai --model gpt-4o-mini --m sdk -e async
                                              # OpenAI 公司，指定模型，SDK 模式，异步执行

支持的提供商: {', '.join(COMPANIES)}
{models_display}"""

    parser = argparse.ArgumentParser(
        description='ApiChatBot 综合测试工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=epilog
    )

    # 测试范围选择（互斥组）：默认测试所有，或限定接口类型，或限定单个公司
    scope_group = parser.add_mutually_exclusive_group(required=False)
    scope_group.add_argument(
        '--interfacetype', '-i',
        choices=['openai', 'google', 'anthropic'],
        help='仅测试指定接口类型下的所有公司'
    )
    scope_group.add_argument(
        '--company', '-c',
        choices=COMPANIES,
        help='仅测试单个公司'
    )

    parser.add_argument(
        '--ssl-verify',
        action='store_true',
        help='启用 SSL 证书验证（默认禁用，适用于代理环境）'
    )

    # 过滤参数（适用于所有测试模式）
    parser.add_argument(
        '--model',
        type=str,
        help='指定任意模型名称（不限建议列表，不指定则测试建议模型）'
    )
    parser.add_argument(
        '--mode', '-m',
        choices=['sdk', 'httpx', 'both'],
        default='both',
        help='选择测试模式（默认both，注意httpx仅OpenAI接口支持）'
    )
    parser.add_argument(
        '--execution', '-e',
        choices=['sync', 'async', 'both'],
        default='both',
        help='选择执行方式（默认both）'
    )
    parser.add_argument(
        '--response', '-r',
        choices=['stream', 'complete', 'both'],
        default='both',
        help='选择响应类型（默认both）'
    )

    args = parser.parse_args()

    # 准备通用参数
    is_ssl_verify = args.ssl_verify
    filter_kwargs = {
        'model': args.model,
        'mode': args.mode,
        'execution': args.execution,
        'response': args.response
    }

    # 根据参数执行对应的测试（默认测试所有提供商）
    if args.interfacetype:
        test_interfacetype(args.interfacetype, is_ssl_verify=is_ssl_verify, **filter_kwargs)
    elif args.company:
        test_company(args.company, is_ssl_verify=is_ssl_verify, **filter_kwargs)
    else:
        # 默认测试所有提供商
        test_all(is_ssl_verify=is_ssl_verify, **filter_kwargs)

#%%



