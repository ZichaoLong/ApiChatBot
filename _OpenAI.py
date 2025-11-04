"""
OpenAI聊天机器人实现模块

提供两种OpenAI接口的实现方式：
1. OpenAISDKChatBot: 使用官方OpenAI SDK
2. OpenAIHttpxChatBot: 使用原始httpx请求（适用于OpenAI兼容API）

特性：
- 支持DeepSeek等模型的推理/思考功能（reasoning_content）
- 支持流式和完整响应
- 支持同步和异步模式
- 自动处理系统指令
"""

import openai
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from typing import Iterator, AsyncIterator, Optional, Callable, Union, Coroutine, List, Dict, Any

from . import StreamUtils
from .Client import *
from ._BaseChatBot import *

__all__ = ['OpenAISDKChatBot', 'OpenAIHttpxChatBot']


def _chatcompletion_message_has_reasoning(message):
    """
    检查消息是否包含推理内容

    DeepSeek等模型提供thinking/reasoning功能，通过reasoning_content属性暴露。

    参数:
        message: ChatCompletion中的message对象

    返回:
        bool: 是否包含推理内容
    """
    return hasattr(message, 'reasoning_content') and message.reasoning_content


def _handle_complete_chatcompletion(
    chatbot,
    inputs: ChatCompletion,
    callback: Optional[Callable[[str, bool, int], None]] = None
):
    """
    处理完整的ChatCompletion响应

    执行回调和实时显示逻辑，支持推理内容和普通内容的分别处理。

    参数:
        chatbot: 聊天机器人实例（用于访问realtime_display和show_thinking配置）
        inputs: OpenAI的ChatCompletion响应对象
        callback: 可选的回调函数，签名为 callback(content, is_thinking, index)

    返回:
        原始的ChatCompletion对象

    处理逻辑:
        1. 如果有callback，对所有choice的推理内容和普通内容执行回调
        2. 如果启用realtime_display，打印第一个choice的内容
        3. 根据show_thinking配置决定是否显示推理内容
    """
    if callback:
        for choice in inputs.choices:
            # 处理推理内容（如DeepSeek的思考过程）
            if _chatcompletion_message_has_reasoning(choice.message):
                callback(choice.message.reasoning_content, True, choice.index)
            # 处理普通内容
            if choice.message.content:
                callback(choice.message.content, False, choice.index)

    if chatbot.realtime_display and inputs.choices:
        message = inputs.choices[0].message
        # 显示推理内容
        if chatbot.show_thinking and _chatcompletion_message_has_reasoning(message):
            print(message.reasoning_content, flush=True)
        # 显示普通内容
        if message.content:
            print(message.content, flush=True)

    return inputs


def _insert_openai_system_instruction(system_instruction: str, messages: List[Dict[str, Any]]):
    """
    向消息列表插入系统指令

    OpenAI接口通过在消息列表开头插入role='system'的消息来传递系统指令。
    如果消息列表已有系统指令，会先移除旧的再插入新的。

    参数:
        system_instruction: 系统指令文本
        messages: 原始消息列表

    返回:
        新的消息列表（包含系统指令）

    注意:
        返回新列表，不修改原列表
    """
    messages = messages.copy()
    message = messages[0]
    # 如果已有系统消息，先移除
    if message['role'] == 'system':
        messages.pop(0)
    # 插入新的系统指令
    messages.insert(0, {'role': 'system', 'content': system_instruction})
    return messages


def _chatcompletion_to_unified_format(response: ChatCompletion) -> Dict[str, Any]:
    """
    将 ChatCompletion 转换为统一格式

    参数:
        response: OpenAI ChatCompletion 对象

    返回:
        统一格式字典，参考 BaseChatBot._to_unified_format 文档
    """
    if not response.choices:
        return _create_empty_unified_response(role='assistant', model=response.model)

    choice = response.choices[0]
    message = choice.message

    # 提取核心内容
    result = {
        'role': 'assistant',
        'content': message.content or '',
        '_model': response.model,
        '_finish_reason': choice.finish_reason or 'unknown'
    }

    # 添加推理内容（如果有）
    if _chatcompletion_message_has_reasoning(message):
        result['_thinking'] = message.reasoning_content

    # 添加 usage 信息
    if response.usage:
        result['_usage'] = {
            'prompt_tokens': response.usage.prompt_tokens,
            'completion_tokens': response.usage.completion_tokens,
            'total_tokens': response.usage.total_tokens
        }
    else:
        result['_usage'] = {}

    # 添加完整原始数据
    result['_raw_dict'] = response.model_dump(exclude_none=True)

    return result


class OpenAISDKChatBot(BaseSDKChatBot):
    """
    基于OpenAI官方SDK的聊天机器人

    使用官方openai Python SDK与OpenAI API进行交互。
    支持所有使用OpenAI SDK兼容接口的提供商（如Moonshot、Aliyun、DeepSeek等）。

    特性:
        - 使用OpenAI SDK的类型安全接口
        - 自动处理流式响应累积
        - 支持推理内容显示（DeepSeek reasoning_content）
        - 支持usage统计（通过stream_options）

    示例:
        >>> chatbot = OpenAISDKChatBot(
        ...     api_key='sk-xxx',
        ...     base_url='https://api.openai.com/v1',
        ...     is_async=False,
        ...     is_ssl_verify=False
        ... )
        >>> response = chatbot.Chat(
        ...     model='gpt-4o',
        ...     messages=[{'role': 'user', 'content': '你好'}],
        ...     stream=True
        ... )
    """

    @property
    def interfacetype(self):
        return 'openai'

    def sa_factory(self):
        """创建OpenAI流式累加器"""
        return StreamUtils.OpenAI.StreamAccumulator()

    def _to_unified_format(self, raw_response: ChatCompletion) -> Dict[str, Any]:
        """将 ChatCompletion 转换为统一格式"""
        return _chatcompletion_to_unified_format(raw_response)

    def _handle_complete_response(
        self,
        inputs: ChatCompletion,
        callback: Optional[Callable[[str, bool, int], None]] = None
    ):
        """处理完整的ChatCompletion响应"""
        return _handle_complete_chatcompletion(self, inputs, callback)

    def send_request(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        stream: bool = False,
        system_instruction: Optional[str] = None,
        **kwargs
    ):
        """
        发送 OpenAI 聊天完成请求

        【调用链】
        Chat()
            → _normalize_messages() (OpenAI无需转换)
            → send_request() [当前]
                → client.chat.completions.create()
            → _handle_sync()
            → _to_unified_format()

        【提供商特殊处理】
        1. 系统指令：插入到消息列表开头
           messages.insert(0, {'role': 'system', 'content': system_instruction})

        2. 流式选项：启用 usage 统计
           stream_options = {'include_usage': True}

        【内部调用】
        self.client.chat.completions.create(
            model=model,
            messages=messages,  # 已插入系统指令
            stream=stream,
            stream_options=stream_options,  # 仅流式时
            **kwargs
        )

        【被调用】
        - BaseChatBot.Chat() - 通过模板方法调用
        - BaseChatBot.AsyncChat() - 异步版本

        【参数】
        model: OpenAI 模型名称（'gpt-4o', 'gpt-4o-mini', 'o1-preview' 等）
        messages: OpenAI 消息格式 [{'role':'user', 'content':'text'}]
        stream: 是否流式响应
        system_instruction: 系统指令（可选，会插入到消息列表开头）
        **kwargs: OpenAI API 参数（temperature, max_tokens, tools, 等）

        【返回】
        同步流式: Stream[ChatCompletionChunk]（实现了 Iterator）
        同步完整: ChatCompletion
        异步流式: AsyncStream[ChatCompletionChunk]（实现了 AsyncIterator）
        异步完整: Awaitable[ChatCompletion]（协程）

        【与其他提供商的差异】
        - Google: 系统指令通过 config.system_instruction
        - Anthropic: 系统指令通过 system 参数
        参考: ARCHITECTURE.md - "系统指令处理差异"
        """
        # 流式响应时包含usage统计
        stream_options = {'include_usage': True} if stream else openai.NOT_GIVEN

        # 处理系统指令
        if system_instruction:
            messages = _insert_openai_system_instruction(system_instruction, messages)

        return self.client.chat.completions.create(
            model=model,
            messages=messages,
            stream=stream,
            stream_options=stream_options,
            **kwargs
        )

class OpenAIHttpxChatBot(BaseChatBot):
    """
    基于httpx原始请求的OpenAI聊天机器人

    绕过OpenAI SDK，使用原始httpx请求直接访问OpenAI兼容的API端点。
    适用于某些提供商的SDK不兼容情况，或需要更精细的HTTP控制。

    特性:
        - 使用原始HTTP请求，无SDK依赖
        - 完全兼容OpenAI的 /chat/completions 端点
        - 支持所有实现OpenAI兼容API的提供商
        - 手动处理流式响应解析

    使用场景:
        - OpenAI SDK与某些提供商不兼容时
        - 需要自定义HTTP请求参数时
        - 调试和理解API行为时

    注意:
        仅支持interfacetype='openai'的提供商

    示例:
        >>> chatbot = OpenAIHttpxChatBot(
        ...     api_key='sk-xxx',
        ...     base_url='https://api.moonshot.cn/v1',
        ...     is_async=False
        ... )
        >>> response = chatbot.Chat(
        ...     model='moonshot-v1-8k',
        ...     messages=[{'role': 'user', 'content': '你好'}]
        ... )
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        is_async: bool,
        is_ssl_verify: bool = False,
        realtime_display: bool = True,
        show_thinking: bool = True
    ):
        """
        初始化Httpx聊天机器人

        参数:
            api_key: API密钥
            base_url: API基础URL（会自动添加 /chat/completions 端点）
            is_async: 是否使用异步客户端
            is_ssl_verify: 是否验证SSL证书，默认False
            realtime_display: 是否实时显示输出，默认True
            show_thinking: 是否显示推理内容，默认True

        异常:
            ValueError: 如果 API key 为空或无效
        """
        super().__init__(realtime_display, show_thinking)

        # 验证 API key
        if not api_key or not api_key.strip():
            raise ValueError(
                "API key cannot be empty for OpenAI Httpx client. "
                "Please check your .env file or pass api_key explicitly."
            )

        self.client = HttpxClient(is_async, is_ssl_verify, timeout=300)
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.url = f'{self.base_url}/chat/completions'
        self.headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-type': 'application/json'
        }

    async def aclose(self):
        """
        关闭异步客户端会话，释放资源

        Httpx特殊实现：httpx使用aclose()而非close()
        """
        await self.client.aclose()

    def _to_unified_format(self, raw_response: ChatCompletion) -> Dict[str, Any]:
        """将 ChatCompletion 转换为统一格式"""
        return _chatcompletion_to_unified_format(raw_response)

    def send_request(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        stream: bool = False,
        system_instruction: Optional[str] = None,
        **kwargs
    ):
        """
        发送HTTP POST请求到 /chat/completions 端点

        参数:
            model: 模型名称
            messages: 消息列表
            stream: 是否使用流式响应
            system_instruction: 系统指令（可选）
            **kwargs: 其他API参数

        返回:
            元组 (stream, response)，其中：
            - stream: bool，是否为流式响应
            - response: httpx Response对象
        """
        # 处理系统指令
        if system_instruction:
            messages = _insert_openai_system_instruction(system_instruction, messages)

        # 构建请求数据
        data = {'model': model, 'messages': messages, 'stream': stream, **kwargs}
        if stream:
            data['stream_options'] = {'include_usage': True}

        # 发送请求
        if stream:
            response = self.client.stream('POST', self.url, headers=self.headers, json=data)
        else:
            response = self.client.post(self.url, headers=self.headers, json=data)

        return stream, response

    def _handle_sync(self, inputs, callback: Optional[Callable[[str, bool, int], None]] = None):
        """
        同步处理httpx响应

        根据响应类型（流式或完整）进行不同处理：
        - 流式：使用Httpx2OpenAI工具解析SSE流
        - 完整：解析JSON并转换为OpenAI格式

        参数:
            inputs: 元组 (stream, response)
            callback: 可选的回调函数

        返回:
            OpenAI格式的ChatCompletion对象
        """
        stream = inputs[0]
        if stream:
            # 流式响应处理
            with inputs[1] as response:
                response.raise_for_status()
                # 将httpx流式响应转换为OpenAI格式
                response = StreamUtils.Httpx2OpenAI.ProcessStreamResponse(
                    response.iter_lines(),
                    callback,
                    self.realtime_display,
                    self.show_thinking
                )
                if self.realtime_display:
                    print('\n')  # 流式输出完成后换行
                return response
        else:
            # 完整响应处理
            response = inputs[1]
            response.raise_for_status()
            # 将JSON响应解析为OpenAI ChatCompletion格式
            response = StreamUtils.Httpx2OpenAI.ParseTotalResponse(response.json())
            return _handle_complete_chatcompletion(self, response, callback)

    async def _handle_async(self, inputs, callback: Optional[Callable[[str, bool, int], None]] = None):
        """
        异步处理httpx响应

        异步版本的_handle_sync，处理逻辑相同。

        参数:
            inputs: 元组 (stream, response)
            callback: 可选的回调函数

        返回:
            OpenAI格式的ChatCompletion对象
        """
        stream = inputs[0]
        if stream:
            # 异步流式响应处理
            async with inputs[1] as response:
                response.raise_for_status()
                # 将httpx异步流式响应转换为OpenAI格式
                response = await StreamUtils.Httpx2OpenAI.AsyncProcessStreamResponse(
                    response.aiter_lines(),
                    callback,
                    self.realtime_display,
                    self.show_thinking
                )
                if self.realtime_display:
                    print('\n')  # 流式输出完成后换行
                return response
        else:
            # 异步完整响应处理（等待响应后复用同步逻辑）
            response = await inputs[1]
            return self._handle_sync((False, response), callback)




