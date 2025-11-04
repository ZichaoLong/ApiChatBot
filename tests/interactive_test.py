#%%
# 启用运行时类型检查
from typeguard import install_import_hook
install_import_hook('ApiChatBot')

import sys
from pathlib import Path
import httpx
import asyncio
import json
from typing import Optional, Dict, Any, List

import ApiChatBot
from ApiChatBot.tests._config import *

#%%
# 多轮对话测试的消息列表
conversation = [
    "你好，请自我介绍一下。",
    "帮我生成一个Python函数来计算斐波那契数列。",
    # "请给出这个函数的时间复杂度分析。",
]
conversation = ['你好', '好的']
system_instruction = "你是一个有用的助手。"

is_ssl_verify = False
show_thinking = True

company = 'openai'
model = MODELS[company][0]
use_sdk = True
is_async = False
is_streamed = True
base_url = BASE_URLS[company]
api_key = API_KEYS[company]
interfacetype = INTERFACETYPE[company]

# 创建聊天机器人实例
chatbotfunc = ApiChatBot.ChatBotFunc(interfacetype, use_sdk)
chatbot = chatbotfunc(
    api_key, base_url,
    is_async=is_async,
    is_ssl_verify=is_ssl_verify,
    realtime_display=True,
    show_thinking=True
)

#%% 多轮对话
async def main():
    """统一的主函数：在单一事件循环中完成所有操作"""
    messages = []
    responses = []

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
                stream=is_streamed, system_instruction=system_instruction)
        else:
            # 在 async 函数中可以调用同步方法
            response = chatbot.Chat(
                model=model, messages=messages,
                stream=is_streamed, system_instruction=system_instruction)

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

# 执行测试（只调用一次 asyncio.run）
asyncio.run(main())

#%%
