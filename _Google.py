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
from .Client import *
from ._BaseChatBot import *

__all__ = ['GoogleSDKChatBot']


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
            api_key: Google API密钥
            base_url: API基础URL
            is_async: 是否使用异步模式（genai.Client内部处理）
            is_ssl_verify: 是否验证SSL（注意：Google SDK不支持外部配置）
            realtime_display: 是否实时显示输出，默认True
            show_thinking: 是否显示思考过程，默认True
            thinking_budget: 思考预算（tokens），默认-1（自动）
                            设置为0禁用思考，正数指定token上限
        """
        super().__init__(realtime_display, show_thinking)
        self.client = ApiSDKClient(
            api_key, base_url,
            interfacetype='google',
            is_async=is_async,
            is_ssl_verify=is_ssl_verify
        )
        self.is_async = is_async
        self.thinking_budget = thinking_budget

    async def aclose(self):
        """
        关闭异步客户端会话，释放资源

        Google特殊实现：使用client.aio.aclose()而非client.close()
        """
        await self.client.aio.aclose()

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
        发送内容生成请求

        参数:
            model: 模型名称（如 'gemini-2.5-flash'）
            messages: 消息列表，格式：[{'role':'user', 'parts':[{'text':'message'}]}]
            stream: 是否使用流式响应
            system_instruction: 系统指令（可选）
            **kwargs: 其他API参数，可包含'config'配置

        返回:
            GenerateContentResponse对象或流式迭代器
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


