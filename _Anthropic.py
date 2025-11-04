"""
Anthropic Claude聊天机器人实现模块

使用Anthropic官方SDK与Claude API进行交互。

特性：
- 支持Claude Sonnet 4等模型
- 支持思考预算（thinking_budget）配置
- 内容块类型包括text和thinking
- 自动配置thinking参数
"""

import anthropic
from anthropic.types import Message
from typing import Iterator, AsyncIterator, Optional, Callable, Union, Coroutine, List, Dict, Any

from . import StreamUtils
from ._BaseChatBot import *

__all__ = ['AnthropicSDKChatBot']


def _message_to_unified_format(response: Message) -> Dict[str, Any]:
    """
    将 Anthropic Message 转换为统一格式

    参数:
        response: Anthropic Message 对象

    返回:
        统一格式字典，参考 BaseChatBot._to_unified_format 文档
    """
    if not response.content:
        return _create_empty_unified_response(
            role='assistant',
            model=str(response.model)
        )

    # 提取文本内容和思考内容
    content_text = ''
    thinking_text = ''

    for block in response.content:
        if block.type == 'text':
            content_text += block.text
        elif block.type == 'thinking':
            thinking_text += block.thinking

    result = {
        'role': 'assistant',
        'content': content_text,
        '_model': str(response.model),
        '_finish_reason': str(response.stop_reason) if response.stop_reason else 'unknown'
    }

    # 添加思考内容
    if thinking_text:
        result['_thinking'] = thinking_text

    # 添加 usage 信息
    if response.usage:
        input_tokens = response.usage.input_tokens or 0
        output_tokens = response.usage.output_tokens or 0
        result['_usage'] = {
            'prompt_tokens': input_tokens,
            'completion_tokens': output_tokens,
            'total_tokens': input_tokens + output_tokens
        }
    else:
        result['_usage'] = {}

    # 添加完整原始数据
    result['_raw_dict'] = response.model_dump(exclude_none=True)

    return result


class AnthropicSDKChatBot(BaseSDKChatBot):
    """
    基于Anthropic官方SDK的Claude聊天机器人

    使用Anthropic官方SDK与Claude API进行交互。

    特性:
        - 支持Claude系列模型（Sonnet 4等）
        - 支持thinking_budget配置（控制思考过程的token数量）
        - 内容块类型：thinking（思考过程）和text（普通文本）
        - 自动配置thinking参数

    注意:
        - 消息格式：[{'role':'user', 'content':'text'}]（类似OpenAI）
        - thinking_budget默认8192 tokens，设为0禁用思考
        - max_tokens默认设为64000

    示例:
        >>> chatbot = AnthropicSDKChatBot(
        ...     api_key='sk-ant-xxx',
        ...     base_url='https://api.anthropic.com',
        ...     is_async=False,
        ...     thinking_budget=4096
        ... )
        >>> response = chatbot.Chat(
        ...     model='claude-sonnet-4-20250514',
        ...     messages=[{'role':'user', 'content':'你好'}]
        ... )
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        is_async: bool,
        is_ssl_verify: bool = False,
        realtime_display: bool = True,
        show_thinking: bool = True,
        thinking_budget: int = 8192
    ):
        """
        初始化Anthropic Claude聊天机器人

        参数:
            thinking_budget: 思考预算（tokens），默认8192
                            设置为0或负数禁用思考
        """
        super().__init__(api_key, base_url, is_async, is_ssl_verify, realtime_display, show_thinking)
        self.thinking_budget = thinking_budget

    @property
    def interfacetype(self):
        return 'anthropic'

    def sa_factory(self):
        """创建Anthropic流式累加器"""
        return StreamUtils.Anthropic.StreamAccumulator()

    def _normalize_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        过滤消息中的元数据字段

        Anthropic 使用标准的 role+content 格式，不需要格式转换，
        但需要过滤掉元数据字段（以_开头的字段），否则会导致 API 验证错误。

        参数:
            messages: 消息列表，可能包含元数据字段

        返回:
            只包含标准字段的消息列表
        """
        if not messages:
            return messages

        cleaned = []
        for msg in messages:
            # 只保留标准字段：role 和 content
            result = {
                'role': msg['role'],
                'content': msg['content']
            }
            cleaned.append(result)

        return cleaned

    def _to_unified_format(self, raw_response: Message) -> Dict[str, Any]:
        """将 Anthropic Message 转换为统一格式"""
        return _message_to_unified_format(raw_response)

    def _handle_complete_response(
        self,
        inputs: Message,
        callback: Optional[Callable[[str, bool, int], None]] = None
    ):
        """
        处理完整的Message响应

        遍历content_block列表，分别处理thinking和text类型的内容块。

        参数:
            inputs: Anthropic的Message对象
            callback: 可选的回调函数

        返回:
            原始的Message对象
        """
        if not inputs or not inputs.content:
            return inputs

        for content_block in inputs.content:
            if content_block.type == 'thinking':
                # 处理思考内容
                if callback:
                    callback(content_block.thinking, True, 0)
                if self.realtime_display and self.show_thinking:
                    print(content_block.thinking, flush=True)
            elif content_block.type == 'text':
                # 处理普通文本
                if callback:
                    callback(content_block.text, False, 0)
                if self.realtime_display:
                    print(content_block.text, flush=True)

        return inputs

    def send_request(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        stream: bool = False,
        system_instruction: Optional[str] = None,
        **kwargs
    ):
        """
        发送 Anthropic Claude 消息创建请求

        【调用链】
        Chat()
            → _normalize_messages() (Anthropic无需转换)
            → send_request() [当前]
                → client.messages.create()
            → _handle_sync()
            → _to_unified_format()

        【提供商特殊处理】
        1. 系统指令：作为独立参数传递
           system=system_instruction

        2. 思考配置：
           thinking = {
               'type': 'enabled',
               'budget_tokens': self.thinking_budget  # 默认 8192
           }

        3. 固定参数：
           max_tokens=64000  # Anthropic 要求必填

        【内部调用】
        self.client.messages.create(
            max_tokens=64000,
            model=model,
            system=system_instruction,  # 独立参数
            messages=messages,
            stream=stream,
            thinking=thinking,  # 思考配置
            **kwargs
        )

        【被调用】
        - BaseChatBot.Chat() - 同步调用
        - BaseChatBot.AsyncChat() - 异步调用

        【参数】
        model: Anthropic 模型名称（'claude-sonnet-4-20250514', 'claude-opus-4' 等）
        messages: Anthropic 消息格式 [{'role':'user', 'content':'text'}]
        stream: 是否流式响应
        system_instruction: 系统指令（可选，作为 system 参数传递）
        **kwargs: 其他 API 参数

        【返回】
        同步流式: Stream[RawMessageStreamEvent]（实现了 Iterator）
        同步完整: Message
        异步流式: AsyncStream[RawMessageStreamEvent]（实现了 AsyncIterator）
        异步完整: Awaitable[Message]（协程）

        【与其他提供商的差异】
        - 消息格式: role + content（与 OpenAI 类似）
        - 系统指令: 作为 system 参数（独立于消息列表）
        - 思考配置: thinking 参数 + budget_tokens
        - 流式响应: 事件驱动模式（RawMessageStreamEvent）
        - 必需参数: max_tokens（默认 64000）
        参考: ARCHITECTURE.md - "各提供商差异对比"
        """
        # 配置thinking参数
        if self.thinking_budget > 0:
            thinking = {'type': 'enabled', 'budget_tokens': self.thinking_budget}
        else:
            thinking = anthropic.NOT_GIVEN

        return self.client.messages.create(
            max_tokens=64000,
            model=model,
            system=system_instruction or anthropic.NOT_GIVEN,
            messages=messages,
            stream=stream,
            thinking=thinking,
            **kwargs
        )

