"""
聊天机器人抽象基类模块

本模块定义了所有聊天机器人实现的基础接口和通用处理逻辑。
提供了统一的对话接口，支持同步/异步、流式/完整响应等多种模式。
"""

from openai.types.chat import ChatCompletion, ChatCompletionChunk
from google.genai.types import GenerateContentResponse
from anthropic.types import Message, RawMessageStreamEvent
from typing import Iterator, AsyncIterator, Optional, Callable, Union, Coroutine, Awaitable, List, Dict, Any

__all__ = ['BaseChatBot', 'BaseSDKChatBot']


class BaseChatBot:
    """
    聊天机器人抽象基类

    定义了所有聊天机器人实现必须遵循的基础接口。
    支持同步/异步对话、实时显示、思考过程可见性等功能。

    属性:
        realtime_display: 是否实时显示输出内容
        show_thinking: 是否显示模型的思考过程（适用于支持推理的模型）

    注意:
        这是一个抽象基类，不应直接实例化。
        子类必须实现所有抽象方法。
    """

    def __init__(self, realtime_display: bool = True, show_thinking: bool = True):
        """
        初始化聊天机器人基类

        参数:
            realtime_display: 是否实时显示输出内容，默认True
            show_thinking: 是否显示思考过程，默认True（适用于DeepSeek等支持推理的模型）
        """
        self.realtime_display = realtime_display
        self.show_thinking = show_thinking

    def _handle_sync(self, inputs, callback: Optional[Callable[[str, bool, int], None]] = None):
        """
        同步处理响应（抽象方法）

        参数:
            inputs: API响应数据（流式迭代器或完整响应对象）
            callback: 可选的回调函数，签名为 callback(content, is_thinking, index)

        返回:
            处理后的完整响应对象

        异常:
            NotImplementedError: 子类必须实现此方法
        """
        raise NotImplementedError

    async def _handle_async(self, inputs, callback: Optional[Callable[[str, bool, int], None]] = None):
        """
        异步处理响应（抽象方法）

        参数:
            inputs: API响应数据（异步流式迭代器或完整响应对象）
            callback: 可选的回调函数，签名为 callback(content, is_thinking, index)

        返回:
            处理后的完整响应对象

        异常:
            NotImplementedError: 子类必须实现此方法
        """
        raise NotImplementedError

    def send_request(self, model: str, messages: List[Dict[str, Any]], stream: bool = False, **kwargs):
        """
        发送API请求（抽象方法）

        参数:
            model: 模型名称
            messages: 消息列表
            stream: 是否使用流式响应
            **kwargs: 其他API参数

        返回:
            API响应对象或迭代器

        异常:
            NotImplementedError: 子类必须实现此方法
        """
        raise NotImplementedError

    def Chat(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        stream: bool = False,
        system_instruction: Optional[str] = None,
        callback: Optional[Callable[[str, bool, int], None]] = None,
        **kwargs
    ):
        """
        同步对话方法

        发送消息并等待响应。支持流式和完整响应模式。

        参数:
            model: 模型名称（如 'gpt-4o', 'gemini-2.5-flash'）
            messages: 消息列表，格式因接口类型而异：
                     - OpenAI/Anthropic: [{'role':'user', 'content':'text'}]
                     - Google: [{'role':'user', 'parts':[{'text':'text'}]}]
            stream: 是否使用流式响应，默认False
            system_instruction: 系统指令（可选）
            callback: 自定义回调函数，签名为 callback(content, is_thinking, index)
            **kwargs: 其他API参数

        返回:
            完整的响应对象（格式因提供商而异）

        示例:
            >>> response = chatbot.Chat(
            ...     model='gpt-4o',
            ...     messages=[{'role': 'user', 'content': '你好'}],
            ...     stream=True
            ... )
        """
        response = self.send_request(model, messages, stream, system_instruction, **kwargs)
        return self._handle_sync(response, callback)

    async def AsyncChat(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        stream: bool = False,
        system_instruction: Optional[str] = None,
        callback: Optional[Callable[[str, bool, int], None]] = None,
        **kwargs
    ):
        """
        异步对话方法

        异步发送消息并等待响应。支持流式和完整响应模式。

        参数:
            model: 模型名称
            messages: 消息列表
            stream: 是否使用流式响应，默认False
            system_instruction: 系统指令（可选）
            callback: 自定义回调函数
            **kwargs: 其他API参数

        返回:
            完整的响应对象（格式因提供商而异）

        示例:
            >>> response = await chatbot.AsyncChat(
            ...     model='gpt-4o',
            ...     messages=[{'role': 'user', 'content': '你好'}],
            ...     stream=True
            ... )
        """
        response = self.send_request(model, messages, stream, system_instruction, **kwargs)
        return await self._handle_async(response, callback)

    def close(self):
        """
        关闭同步客户端会话，释放资源

        应在不再使用聊天机器人时调用，以释放网络连接等资源。
        """
        self.client.close()

    async def aclose(self):
        """
        关闭异步客户端会话，释放资源

        异步版本的close方法，应在异步环境中使用。
        """
        await self.client.close()

    def __enter__(self):
        """
        同步上下文管理器入口

        支持使用 with 语句自动管理资源：
            >>> with chatbot:
            ...     response = chatbot.Chat(...)
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        同步上下文管理器退出

        自动调用 close() 释放资源。
        """
        self.close()
        return False

    async def __aenter__(self):
        """
        异步上下文管理器入口

        支持使用 async with 语句自动管理资源：
            >>> async with chatbot:
            ...     response = await chatbot.AsyncChat(...)
        """
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """
        异步上下文管理器退出

        自动调用 aclose() 释放资源。
        """
        await self.aclose()
        return False


class BaseSDKChatBot(BaseChatBot):
    """
    基于SDK的聊天机器人抽象基类

    扩展了BaseChatBot，添加了流式响应累加器（StreamAccumulator）支持。
    适用于使用官方SDK的实现（OpenAI SDK、Google genai、Anthropic SDK等）。

    流式累加器负责：
    1. 增量处理流式响应数据块
    2. 实时显示和回调处理
    3. 将流式数据累积为完整响应对象

    子类需要实现：
        - sa_factory(): 创建对应提供商的StreamAccumulator实例
        - _handle_complete_response(): 处理完整响应的显示和回调
    """

    def sa_factory(self):
        """
        创建流式累加器工厂方法（抽象方法）

        返回:
            对应提供商的StreamAccumulator实例

        异常:
            NotImplementedError: 子类必须实现此方法
        """
        raise NotImplementedError

    def _handle_complete_response(
        self,
        inputs: Union[ChatCompletion, GenerateContentResponse, Message],
        callback: Optional[Callable[[str, bool, int], None]] = None
    ):
        """
        处理完整响应（抽象方法）

        对非流式的完整响应进行处理，包括：
        - 执行用户自定义的回调函数
        - 根据realtime_display设置显示内容
        - 根据show_thinking设置显示思考过程

        参数:
            inputs: 完整的API响应对象
            callback: 可选的回调函数

        返回:
            处理后的完整响应对象

        异常:
            NotImplementedError: 子类必须实现此方法
        """
        raise NotImplementedError

    def _handle_sync(
        self,
        inputs: Union[
            Iterator[ChatCompletionChunk],          # OpenAI流式
            Iterator[GenerateContentResponse],      # Google流式
            Iterator[RawMessageStreamEvent],        # Anthropic流式
            ChatCompletion,                         # OpenAI完整
            GenerateContentResponse,                # Google完整
            Message                                 # Anthropic完整
        ],
        callback: Optional[Callable[[str, bool, int], None]] = None
    ):
        """
        同步处理响应

        自动判断是流式还是完整响应，并调用相应的处理逻辑：
        - 流式响应：使用StreamAccumulator逐块处理
        - 完整响应：直接调用_handle_complete_response

        参数:
            inputs: API响应，可能的类型：
                   流式: Iterator[ChatCompletionChunk] | Iterator[GenerateContentResponse] | Iterator[RawMessageStreamEvent]
                   完整: ChatCompletion | GenerateContentResponse | Message
            callback: 可选的回调函数

        返回:
            完整的响应对象
        """
        if isinstance(inputs, Iterator):
            # 流式响应处理
            sa = self.sa_factory()
            for chunk in inputs:
                sa.add_chunk(chunk, callback, self.realtime_display, self.show_thinking)
            if self.realtime_display:
                print('\n')  # 流式输出完成后换行
            return sa.to_complete_response()
        else:
            # 完整响应处理
            assert isinstance(inputs, ChatCompletion) or \
                isinstance(inputs, GenerateContentResponse) or \
                isinstance(inputs, Message)
            return self._handle_complete_response(inputs, callback)

    async def _handle_async(
        self,
        inputs: Union[
            AsyncIterator[ChatCompletionChunk],        # OpenAI异步流式（直接返回，无需await）
            AsyncIterator[GenerateContentResponse],    # Google异步流式（直接返回，无需await）
            AsyncIterator[RawMessageStreamEvent],      # Anthropic异步流式（直接返回，无需await）
            Awaitable[ChatCompletion],                 # OpenAI完整（协程，需要await）
            Awaitable[GenerateContentResponse],        # Google完整（协程，需要await）
            Awaitable[Message]                         # Anthropic完整（协程，需要await）
        ],
        callback: Optional[Callable[[str, bool, int], None]] = None
    ):
        """
        异步处理响应

        异步版本的_handle_sync，处理逻辑相同。
        首先判断输入类型：如果是AsyncIterator则直接使用，否则先await协程。

        参数:
            inputs: API响应，可能的类型：
                   - 异步流式: AsyncIterator（直接返回，无需await）
                   - 完整响应: Awaitable（协程，需要await）
            callback: 可选的回调函数

        返回:
            完整的响应对象

        注意:
            不同SDK的异步行为不同：
            - OpenAI流式: 直接返回AsyncStream，不需要await
            - OpenAI非流式: 返回协程，需要await
        """
        # 区分 await 前后的类型，使用不同变量
        # resolved_inputs: await 后的实际类型
        resolved_inputs: Union[
            AsyncIterator[ChatCompletionChunk],
            AsyncIterator[GenerateContentResponse],
            AsyncIterator[RawMessageStreamEvent],
            ChatCompletion,
            GenerateContentResponse,
            Message
        ]

        if isinstance(inputs, AsyncIterator):
            # 流式响应，直接使用
            resolved_inputs = inputs
        else:
            # 完整响应，需要 await
            resolved_inputs = await inputs

        if isinstance(resolved_inputs, AsyncIterator):
            # 异步流式响应处理
            sa = self.sa_factory()
            async for chunk in resolved_inputs:
                sa.add_chunk(chunk, callback, self.realtime_display, self.show_thinking)
            if self.realtime_display:
                print('\n')  # 流式输出完成后换行
            return sa.to_complete_response()
        else:
            # 完整响应处理（复用同步逻辑）
            return self._handle_sync(resolved_inputs, callback)

