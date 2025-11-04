"""
聊天机器人抽象基类模块

本模块定义了所有聊天机器人实现的基础接口和通用处理逻辑。
提供了统一的对话接口，支持同步/异步、流式/完整响应等多种模式。
"""

import asyncio
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from google.genai.types import GenerateContentResponse
from anthropic.types import Message, RawMessageStreamEvent
from typing import Iterator, AsyncIterator, Optional, Callable, Union, Coroutine, Awaitable, List, Dict, Any
from .Client import ApiSDKClient
from .response_dict import ResponseDict,_create_empty_unified_response

__all__ = ['BaseChatBot', 'BaseSDKChatBot', '_create_empty_unified_response', 'ResponseDict']

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
        """
        self.realtime_display = realtime_display
        self.show_thinking = show_thinking

    def _to_unified_format(self, raw_response: Union[ChatCompletion, GenerateContentResponse, Message]) -> Dict[str, Any]:
        """
        将原始SDK响应转换为统一格式（抽象方法）

        统一格式设计：
        {
            'role': 'assistant',           # 或 'model' (Google)
            'content': '回答内容',
            '_thinking': '思考内容',       # 可选，如果有思考过程
            '_usage': {                    # token使用统计
                'prompt_tokens': int,
                'completion_tokens': int,
                'total_tokens': int
            },
            '_model': 'gpt-4o',           # 使用的模型名
            '_finish_reason': 'stop',     # 停止原因
            '_raw_dict': {...}            # 完整的原始SDK响应（通过model_dump获取）
        }

        注意：
            - 下划线前缀的字段为元信息
            - 无下划线字段可直接作为message使用
            - _raw_dict包含所有提供商特有字段（通过SDK的model_dump方法自动获取）
            - 子类必须实现此方法

        参数:
            raw_response: 原始SDK响应对象

        返回:
            统一格式的字典

        异常:
            NotImplementedError: 子类必须实现此方法
        """
        raise NotImplementedError

    def _normalize_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        消息格式标准化钩子（子类可重写）

        将用户提供的标准格式（OpenAI格式）消息转换为当前提供商需要的格式。
        默认实现不进行转换，直接返回原消息列表。

        【标准格式】
        [{'role': 'user', 'content': '文本'}]

        【提供商差异】
        - OpenAI: 与标准格式相同，不需要重写
        - Google: 需格式转换（content→parts）并过滤元数据，见 `_Google.py::_convert_to_google_format()`
        - Anthropic: 需过滤元数据字段（以_开头的字段），见 `_Anthropic.py::_normalize_messages()`

        参数:
            messages: 用户提供的消息列表（标准 OpenAI 格式）

        返回:
            转换后的消息列表（适配当前提供商格式）

        注意:
            - 此方法应支持向后兼容，如果传入的已是目标格式，直接返回
            - 子类重写时应保持幂等性（多次调用结果相同）
        """
        return messages

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
        raw_response: bool = False,
        **kwargs
    ):
        """
        同步对话方法 - 用户入口

        发送消息并等待响应。支持流式和完整响应模式。

        【调用链】
        用户代码
            ↓
        Chat() [当前]
            ↓
        1. _normalize_messages() - 消息格式标准化
            ↓
        2. send_request() - 子类实现，发送请求
            ↓
        3. _handle_sync() - 处理响应
            ↓
        4. _to_unified_format() - 格式转换（如 raw_response=False）
            ↓
        返回统一格式字典或原始SDK对象

        【内部调用】
        1. messages = self._normalize_messages(messages)
           - 将标准格式转换为提供商所需格式
           - OpenAI: 无需转换，直接返回
           - Google/Anthropic: 重写，格式转换（Google only）及过滤元数据字段（以_开头的字段）

        2. response = self.send_request(model, messages, stream, system_instruction, **kwargs)
           - 子类实现，调用 SDK 或 HTTP 客户端发送请求
           - 返回: Iterator[Chunk] (流式) 或 CompleteResponse (完整)

        3. raw = self._handle_sync(response, callback)
           - 判断响应类型并选择处理策略
           - 流式: 使用 StreamAccumulator 逐块累加
           - 完整: 调用 _handle_complete_response

        4. return raw if raw_response else self._to_unified_format(raw)
           - raw_response=False: 转换为统一格式字典
           - raw_response=True: 返回原始SDK对象

        【参数】
        model: 模型名称（如 'gpt-4o', 'gemini-2.5-flash'）
        messages: 消息列表，统一使用标准格式（OpenAI格式）：
                 [{'role':'user', 'content':'text'}]
                 内部会自动转换为各提供商所需格式
        stream: 是否使用流式响应，默认 False
        system_instruction: 系统指令（可选）
        callback: 自定义回调，签名 callback(content: str, is_thinking: bool, index: int)
        raw_response: 是否返回原始SDK响应对象，默认 False
                     False: 返回统一格式（详见 _to_unified_format）
                     True: 返回原始SDK对象
        **kwargs: 其他 API 参数（temperature, max_tokens 等）

        【返回】
        默认: 统一格式字典（详见 _to_unified_format 文档）
              包含核心字段(role, content)、标准元数据(_thinking, _usage, _model, _finish_reason)
              和完整原始数据(_raw_dict)
        raw_response=True: 原始SDK对象（ChatCompletion/GenerateContentResponse/Message）

        【示例】
        >>> response = chatbot.Chat(model='gpt-4o', messages=[...])
        >>> print(response['content'])
        >>> messages.append(response)  # 可直接追加
        >>> print(response['_usage'])  # 访问token统计
        >>> print(response['_raw_dict'])  # 访问完整原始数据

        【参考】
        详细架构: ApiChatBot/ARCHITECTURE.md
        """
        messages = self._normalize_messages(messages)
        response = self.send_request(model, messages, stream, system_instruction, **kwargs)
        raw = self._handle_sync(response, callback)
        if raw_response:
            return raw
        else:
            unified = self._to_unified_format(raw)
            return unified if isinstance(unified, ResponseDict) else ResponseDict(unified)

    async def AsyncChat(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        stream: bool = False,
        system_instruction: Optional[str] = None,
        callback: Optional[Callable[[str, bool, int], None]] = None,
        raw_response: bool = False,
        **kwargs
    ):
        """
        异步对话方法 - 用户异步入口

        异步发送消息并等待响应。支持流式和完整响应模式。

        【调用链】
        用户代码 (async)
            ↓
        await AsyncChat() [当前]
            ↓
        1. _normalize_messages() - 消息格式标准化
            ↓
        2. send_request() - 子类实现，发送请求
            ↓
        3. await _handle_async() - 异步处理响应
            ↓
        4. _to_unified_format() - 格式转换（如 raw_response=False）
            ↓
        返回统一格式字典或原始SDK对象

        【内部调用】
        1. messages = self._normalize_messages(messages)
           - 将标准格式转换为提供商所需格式（同步版本）

        2. response = self.send_request(model, messages, stream, system_instruction, **kwargs)
           - 子类实现，调用 SDK 客户端发送请求
           - 返回: AsyncIterator[Chunk] (流式) 或 Awaitable[CompleteResponse] (完整)

        3. raw = await self._handle_async(response, callback)
           - 判断响应类型（AsyncIterator 或 Awaitable）
           - 流式: async for chunk 逐块累加
           - 完整: await response 后处理

        4. return raw if raw_response else self._to_unified_format(raw)
           - 根据 raw_response 参数决定返回格式

        【参数】
        model: 模型名称（如 'gpt-4o', 'claude-sonnet-4'）
        messages: 消息列表，统一使用标准格式（OpenAI格式）：
                 [{'role':'user', 'content':'text'}]
                 内部会自动转换为各提供商所需格式
        stream: 是否使用流式响应，默认 False
        system_instruction: 系统指令（可选）
        callback: 自定义回调函数
        raw_response: 是否返回原始SDK响应对象，默认 False
        **kwargs: 其他 API 参数

        【返回】
        与同步版本相同（详见 Chat 方法）
        包含核心字段、标准元数据和完整原始数据(_raw_dict)

        【示例】
        >>> response = await chatbot.AsyncChat(model='gpt-4o', messages=[...])
        >>> print(response['content'])
        >>> messages.append(response)
        >>> print(response['_raw_dict'])  # 访问完整原始数据

        【注意】
        必须在异步环境中使用，初始化时需设置 is_async=True
        """
        messages = self._normalize_messages(messages)
        response = self.send_request(model, messages, stream, system_instruction, **kwargs)
        raw = await self._handle_async(response, callback)
        if raw_response:
            return raw
        else:
            unified = self._to_unified_format(raw)
            return unified if isinstance(unified, ResponseDict) else ResponseDict(unified)

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
        初始化 BaseSDKChatBot

        参数:
            api_key: API密钥
            base_url: API基础URL
            is_async: 是否使用异步客户端
            is_ssl_verify: 是否验证SSL证书，默认False
            realtime_display: 是否实时显示输出，默认True
            show_thinking: 是否显示思考过程，默认True
        """
        super().__init__(realtime_display, show_thinking)
        self.api_key = api_key
        self.base_url = base_url
        self.is_async = is_async
        self.is_ssl_verify = is_ssl_verify
        self.client = None
        self.reset_client() # 初始化 client

    @property
    def interfacetype(self):
        raise NotImplementedError
    def reset_client(self):
        self.client = ApiSDKClient(
            self.api_key, self.base_url,
            interfacetype=self.interfacetype,
            is_async=self.is_async,
            is_ssl_verify=self.is_ssl_verify
        )

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
        同步处理响应 - 流式/完整响应分发器

        根据 inputs 类型自动选择处理策略：流式累加或完整显示。

        【调用链】
        Chat()
            → _normalize_messages()
            → send_request()
            → _handle_sync() [当前]
                ↓
        ┌───────────────┴───────────────┐
        ↓                               ↓
    【流式响应】                      【完整响应】
    isinstance(inputs, Iterator)      其他类型
        ↓                               ↓
    sa = sa_factory()               _handle_complete_response()
        ↓                               ↓
    for chunk in inputs:              执行回调 + 显示
      sa.add_chunk()                        ↓
        ↓                           返回原响应对象
    sa.to_complete_response()
        ↓
    返回完整响应对象
        ↓
    → _to_unified_format() (在 Chat() 中)

        【内部调用】
        流式分支:
          1. sa = self.sa_factory() - 创建流式累加器
          2. for chunk in inputs:
               sa.add_chunk(chunk, callback, ...) - 逐块累加、回调、显示
          3. sa.to_complete_response() - 合并所有 chunk

        完整分支:
          1. self._handle_complete_response(inputs, callback) - 处理完整响应

        【被调用】
        - BaseChatBot.Chat() - 同步对话入口
        - BaseSDKChatBot._handle_async() - 异步处理中复用同步逻辑

        【参数】
        inputs: API 响应，类型由 send_request() 返回值决定
               流式: Iterator[ChatCompletionChunk] (OpenAI)
                    Iterator[GenerateContentResponse] (Google)
                    Iterator[RawMessageStreamEvent] (Anthropic)
               完整: ChatCompletion (OpenAI)
                    GenerateContentResponse (Google)
                    Message (Anthropic)
        callback: 可选回调，签名 callback(content: str, is_thinking: bool, index: int)

        【返回】
        完整响应对象（与 inputs 的完整类型相同）

        【实现位置】
        - BaseSDKChatBot: 本方法实现（统一处理）
        - OpenAIHttpxChatBot: 重写（特殊处理 httpx 响应元组）
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
        异步处理响应 - 异步版本的流式/完整分发器

        与 _handle_sync() 逻辑相同，但需先判断并 await 协程。

        【调用链】
        AsyncChat()
            → _normalize_messages()
            → send_request()
            → await _handle_async() [当前]
                ↓
        【类型判断与 await】
        isinstance(inputs, AsyncIterator)?
        ↓                               ↓
       YES (流式)                      NO (完整)
  resolved_inputs = inputs        resolved_inputs = await inputs
        ↓                               ↓
  isinstance(resolved_inputs, AsyncIterator)?
        ↓                               ↓
  【异步流式响应】                     【完整响应】
  sa = sa_factory()                _handle_sync(resolved_inputs)
        ↓                               ↓
  async for chunk in resolved_inputs:   复用同步逻辑
    sa.add_chunk()                        ↓
        ↓                           返回响应对象
  sa.to_complete_response()
        ↓
  返回完整响应对象
        ↓
    → _to_unified_format() (在 AsyncChat() 中)

        【内部调用】
        1. 类型判断和 await:
           - isinstance(inputs, AsyncIterator): 流式，直接使用
           - else: 完整响应，await inputs 等待协程

        2. 流式分支（resolved_inputs 是 AsyncIterator）:
           a. sa = self.sa_factory()
           b. async for chunk in resolved_inputs:
                sa.add_chunk(chunk, callback, ...)
           c. return sa.to_complete_response()

        3. 完整分支（resolved_inputs 不是 AsyncIterator）:
           - 复用同步逻辑: self._handle_sync(resolved_inputs, callback)

        【被调用】
        - BaseChatBot.AsyncChat() - 异步对话入口

        【参数】
        inputs: API 响应，类型由 send_request() 返回值和 is_async 决定
               异步流式: AsyncIterator[ChatCompletionChunk] (OpenAI)
                        AsyncIterator[GenerateContentResponse] (Google)
                        AsyncIterator[RawMessageStreamEvent] (Anthropic)
               完整响应: Awaitable[ChatCompletion] (OpenAI，协程)
                        Awaitable[GenerateContentResponse] (Google，协程)
                        Awaitable[Message] (Anthropic，协程)
        callback: 可选回调函数

        【返回】
        完整响应对象（与同步版本类型相同）

        【注意】
        SDK 异步行为差异：
        - 流式: 直接返回 AsyncIterator，无需 await
        - 完整: 返回协程 (Awaitable)，必须 await
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

