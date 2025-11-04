"""
Google Gemini聊天机器人实现模块

使用Google官方genai SDK与Gemini API进行交互。

特性：
- 支持Gemini 2.5 Flash、Pro等模型
- 支持思考预算（thinking_budget）配置
- Google特有的消息格式（role + parts）
- 自动配置thinking_config

消息格式说明：
    Google使用不同于OpenAI的消息格式：
    [{'role': 'user', 'parts': [{'text': 'your message'}]}]
"""

from google import genai
from google.genai.types import GenerateContentResponse
from typing import Iterator, AsyncIterator, Optional, Callable, Union, Coroutine, List, Dict, Any

from . import StreamUtils
from ._BaseChatBot import *

__all__ = ['GoogleSDKChatBot']


def _generatecontent_response_to_unified_format(response: GenerateContentResponse) -> Dict[str, Any]:
    """
    将 GenerateContentResponse 转换为统一格式

    参数:
        response: Google GenerateContentResponse 对象

    返回:
        统一格式字典，参考 BaseChatBot._to_unified_format 文档
    """
    if not response.candidates:
        return _create_empty_unified_response(
            role='model',
            model=response.model_version or 'unknown'
        )

    candidate = response.candidates[0]

    # 提取文本内容和思考内容
    content_text = ''
    thinking_text = ''

    if candidate.content and candidate.content.parts:
        for part in candidate.content.parts:
            if part.text:
                if part.thought:  # 思考内容
                    thinking_text += part.text
                else:  # 普通内容
                    content_text += part.text

    result = {
        'role': 'model',  # Google uses 'model' instead of 'assistant'
        'content': content_text,
        '_model': response.model_version or 'unknown',
        '_finish_reason': str(candidate.finish_reason) if candidate.finish_reason else 'unknown'
    }

    # 添加思考内容
    if thinking_text:
        result['_thinking'] = thinking_text

    # 添加 usage 信息
    if response.usage_metadata:
        result['_usage'] = {
            'prompt_tokens': response.usage_metadata.prompt_token_count or 0,
            'completion_tokens': response.usage_metadata.candidates_token_count or 0,
            'total_tokens': response.usage_metadata.total_token_count or 0
        }
    else:
        result['_usage'] = {}

    # 添加完整原始数据
    result['_raw_dict'] = response.model_dump(exclude_none=True)

    return result


def _convert_to_google_format(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    将标准OpenAI格式消息转换为Google格式

    标准格式: [{'role': 'user', 'content': '文本'}]
    Google格式: [{'role': 'user', 'parts': [{'text': '文本'}]}]

    参数:
        messages: 标准格式或Google格式的消息列表

    返回:
        Google格式的消息列表

    注意:
        - 支持向后兼容：如果已经是Google格式，直接返回
        - 保持幂等性：多次调用结果相同
        - 保留元数据字段：转换时保留所有以_开头的元数据字段
    """
    if not messages:
        return messages

    converted = []
    for msg in messages:
        # 检测是否已经是Google格式
        if 'parts' in msg:
            # 已经是Google格式，过滤掉元数据字段
            result = {
                'role': msg['role'],
                'parts': msg['parts']
            }
            converted.append(result)
        elif 'content' in msg:
            # 标准OpenAI格式，转换为Google格式
            result = {
                'role': msg['role'],
                'parts': [{'text': msg['content']}]
            }
            # 注意：不保留元数据字段（以_开头的字段）
            # 这些字段是输出时的元数据，不应该作为输入传递给SDK
            # 否则会导致Google SDK的Pydantic验证错误
            converted.append(result)
        else:
            # 其他格式，直接保留
            converted.append(msg)

    return converted


class GoogleSDKChatBot(BaseSDKChatBot):
    """
    基于Google genai SDK的Gemini聊天机器人

    使用Google官方genai Python SDK与Gemini API进行交互。

    特性:
        - 支持Gemini系列模型（2.5 Flash、2.5 Pro等）
        - 支持thinking_budget配置（控制思考过程的token数量）
        - Google特有的消息格式（role + parts结构）
        - genai.Client内部处理同步/异步切换

    注意:
        - 消息格式：[{'role':'user', 'parts':[{'text':'message'}]}]
        - Google SDK不支持外部http_client配置（SSL验证需修改源码）
        - thinking_budget默认-1（由模型自动决定）
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        is_async: bool,
        is_ssl_verify: bool = False,
        realtime_display: bool = True,
        show_thinking: bool = True,
        thinking_budget: int = -1
    ):
        """
        初始化Google Gemini聊天机器人

        参数:
            thinking_budget: 思考预算（tokens），默认-1（自动）
                            设置为0禁用思考，正数指定token上限
        """
        super().__init__(api_key, base_url, is_async, is_ssl_verify, realtime_display, show_thinking)
        self.thinking_budget = thinking_budget

    @property
    def interfacetype(self):
        return 'google'

    async def aclose(self):
        """
        关闭异步客户端会话，释放资源

        Google特殊实现：使用client.aio.aclose()而非client.close()
        """
        await self.client.aio.aclose()

    def _normalize_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        重写消息格式标准化方法

        将标准OpenAI格式转换为Google所需的parts格式。
        """
        return _convert_to_google_format(messages)

    def _to_unified_format(self, raw_response: GenerateContentResponse) -> Dict[str, Any]:
        """将 GenerateContentResponse 转换为统一格式"""
        return _generatecontent_response_to_unified_format(raw_response)

    def sa_factory(self):
        """创建Google流式累加器"""
        return StreamUtils.Google.StreamAccumulator()

    def _handle_complete_response(
        self,
        inputs: GenerateContentResponse,
        callback: Optional[Callable[[str, bool, int], None]] = None
    ):
        """
        处理完整的GenerateContentResponse响应

        Google SDK特点：chunk与complete response类型相同，
        因此可以复用StreamAccumulator的add_chunk方法处理完整响应。

        参数:
            inputs: Google的GenerateContentResponse对象
            callback: 可选的回调函数

        返回:
            原始的GenerateContentResponse对象
        """
        sa = self.sa_factory()
        sa.add_chunk(inputs, callback, self.realtime_display, self.show_thinking)
        if self.realtime_display:
            # 补上换行符（add_chunk结束符为空）
            print(flush=True)
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
        发送 Google Gemini 内容生成请求

        【调用链】
        Chat()
            → _normalize_messages() (转换为 parts 格式)
            → send_request() [当前]
                → client.models.generate_content() 或 generate_content_stream()
            → _handle_sync()
            → _to_unified_format()

        【提供商特殊处理】
        1. 同步/异步选择：
           models = client.aio.models (异步) 或 client.models (同步)

        2. 系统指令：通过配置对象传递
           config.system_instruction = system_instruction

        3. 思考配置：
           config.thinking_config = ThinkingConfig(
               include_thoughts=True,
               thinking_budget=self.thinking_budget  # -1=自动，0=禁用
           )

        【内部调用】
        1. 选择接口：
           models = self.client.aio.models if self.is_async else self.client.models

        2. 配置思考：
           config.thinking_config = ThinkingConfig(...)

        3. 设置系统指令：
           if system_instruction:
               config.system_instruction = system_instruction

        4. 调用生成：
           models.generate_content() 或 generate_content_stream()

        【被调用】
        - BaseChatBot.Chat() - 同步调用
        - BaseChatBot.AsyncChat() - 异步调用（SDK 内部处理）

        【参数】
        model: Google 模型名称（'gemini-2.5-flash', 'gemini-2.5-pro' 等）
        messages: Google 消息格式 [{'role':'user', 'parts':[{'text':'message'}]}]
                 注意：助手角色名为 'model' 而非 'assistant'
        stream: 是否流式响应
        system_instruction: 系统指令（可选，通过 config 传递）
        **kwargs: 其他 API 参数，可包含自定义 'config'

        【返回】
        同步流式: Iterator[GenerateContentResponse]
        同步完整: GenerateContentResponse
        异步流式: AsyncIterator[GenerateContentResponse]
        异步完整: Awaitable[GenerateContentResponse]（协程）

        【与其他提供商的差异】
        - 消息格式: role + parts（而非 role + content）
        - 系统指令: 通过 config.system_instruction
        - 思考配置: thinking_config + thinking_budget
        - 客户端: 单一客户端，通过 client.models / client.aio.models 切换
        参考: ARCHITECTURE.md - "各提供商差异对比"
        """
        # 根据is_async选择同步或异步models
        models = self.client.aio.models if self.is_async else self.client.models
        generate_func = models.generate_content_stream if stream else models.generate_content

        # 处理配置对象
        if 'config' not in kwargs:
            config = genai.types.GenerateContentConfig()
        else:
            config = kwargs['config']
            kwargs = kwargs.copy()
            kwargs.pop('config')

        # 配置thinking_config
        if not hasattr(config, 'thinking_config') or config.thinking_config is None:
            config.thinking_config = genai.types.ThinkingConfig(
                include_thoughts=True,
                thinking_budget=self.thinking_budget
            )

        # 设置系统指令
        if system_instruction:
            config.system_instruction = system_instruction

        # 确保thinking_budget已设置
        if not config.thinking_config.thinking_budget:
            config.thinking_config.thinking_budget = self.thinking_budget

        return generate_func(model=model, contents=messages, config=config)


