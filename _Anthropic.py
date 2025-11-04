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
from .Client import *
from ._BaseChatBot import *

__all__ = ['AnthropicSDKChatBot']


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
            api_key: Anthropic API密钥
            base_url: API基础URL
            is_async: 是否使用异步客户端
            is_ssl_verify: 是否验证SSL证书，默认False
            realtime_display: 是否实时显示输出，默认True
            show_thinking: 是否显示思考过程，默认True
            thinking_budget: 思考预算（tokens），默认8192
                            设置为0或负数禁用思考
        """
        super().__init__(realtime_display, show_thinking)
        self.client = ApiSDKClient(
            api_key, base_url,
            interfacetype='anthropic',
            is_async=is_async,
            is_ssl_verify=is_ssl_verify
        )
        self.thinking_budget = thinking_budget

    def sa_factory(self):
        """创建Anthropic流式累加器"""
        return StreamUtils.Anthropic.StreamAccumulator()

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
        发送消息创建请求

        参数:
            model: 模型名称（如 'claude-sonnet-4-20250514'）
            messages: 消息列表，格式：[{'role':'user', 'content':'text'}]
            stream: 是否使用流式响应
            system_instruction: 系统指令（可选）
            **kwargs: 其他API参数

        返回:
            Message对象或流式迭代器
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

