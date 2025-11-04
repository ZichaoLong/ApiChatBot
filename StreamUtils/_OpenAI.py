"""
OpenAI流式响应处理模块

提供OpenAI流式ChatCompletion响应的累加和转换功能。
将流式的ChatCompletionChunk累积并转换为完整的ChatCompletion对象。

主要功能:
- StreamAccumulator: 累加流式chunk并转换为完整响应
- 支持推理内容（reasoning_content）处理
- 实时显示和回调支持
"""

from dataclasses import dataclass, field
from typing import Optional, Callable, List
from openai.types.chat import ChatCompletion, ChatCompletionChunk, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice

from .common_utils import RealTimeDisplayHandler

__all__ = ['StreamAccumulator']


def _chatcompletionchunk_delta_has_reasoning(delta):
    """
    检查delta是否包含推理内容

    DeepSeek等模型通过reasoning_content属性提供推理过程。

    参数:
        delta: ChatCompletionChunk的delta对象

    返回:
        bool: 是否包含推理内容
    """
    return hasattr(delta, 'reasoning_content') and delta.reasoning_content

def chunks_to_complete_response(chunks: List[ChatCompletionChunk]) -> ChatCompletion:
    """
    将流式chunk列表转换为完整的ChatCompletion响应

    遍历所有chunk，累积内容、推理内容和元数据，
    最终构造完整的ChatCompletion对象。

    参数:
        chunks: ChatCompletionChunk对象列表

    返回:
        完整的ChatCompletion对象

    处理逻辑:
        1. 从首个chunk提取基础信息（id、model等）
        2. 按choice.index分组累积内容
        3. 合并所有delta.content和delta.reasoning_content
        4. 从最后一个chunk获取usage信息
    """
    # 收集基础信息的变量
    id = None
    created = None
    model = None
    system_fingerprint = None

    # 用于收集所有choice的信息（按index分组）
    choices_data = {}
    usage = None

    for chunk in chunks:
        # 从第一个chunk获取基础信息
        if id is None:
            id = chunk.id
            created = chunk.created
            model = chunk.model
            system_fingerprint = chunk.system_fingerprint

        if chunk.choices:
            for choice in chunk.choices:
                index = choice.index
                delta = choice.delta

                # 初始化choice数据结构
                if index not in choices_data:
                    choices_data[index] = {
                        'content_parts': [],           # 普通内容片段
                        'reasoning_content_parts': []  # 推理内容片段
                    }
                choice_data = choices_data[index]

                # 累积内容片段
                if delta.content:
                    choice_data['content_parts'].append(delta.content)
                if _chatcompletionchunk_delta_has_reasoning(delta):
                    choice_data['reasoning_content_parts'].append(delta.reasoning_content)

                # 收集元数据（每次更新，保留最新值）
                choice_data['role'] = delta.role or choice_data.get('role')
                choice_data['function_call'] = delta.function_call or choice_data.get('function_call')
                choice_data['tool_calls'] = delta.tool_calls or choice_data.get('tool_calls')
                choice_data['finish_reason'] = choice.finish_reason or choice_data.get('finish_reason')

        # 获取usage信息（仅在最后一个chunk中）
        if chunk.usage:
            usage = chunk.usage

    # 构建所有choice对象
    choices = []
    for index in sorted(choices_data.keys()):
        choice_data = choices_data[index]

        # 构建消息对象
        message = dict(
            content=''.join(choice_data['content_parts']),        # 合并所有内容片段
            role=choice_data.get('role') or 'assistant',
            function_call=choice_data.get('function_call'),
            tool_calls=choice_data.get('tool_calls')
        )

        # 添加推理内容（如果有）
        if choice_data['reasoning_content_parts']:
            full_reasoning_content = ''.join(choice_data['reasoning_content_parts'])
            message['reasoning_content'] = full_reasoning_content

        message = ChatCompletionMessage(**message)

        # 构建Choice对象
        choice = Choice.model_construct(
            finish_reason=choice_data.get('finish_reason'),
            index=index,
            message=message,
            logprobs=None
        )
        choices.append(choice)

    # 创建完整的ChatCompletion对象
    chat_completion = ChatCompletion(
        id=id,
        choices=choices,
        created=created,
        model=model,
        object='chat.completion',
        system_fingerprint=system_fingerprint,
        usage=usage
    )

    return chat_completion


@dataclass
class StreamAccumulator(RealTimeDisplayHandler):
    """
    OpenAI流式响应累加器

    累积流式ChatCompletionChunk并提供转换为完整ChatCompletion的功能。
    继承RealTimeDisplayHandler以支持实时显示。

    属性:
        chunks: 累积的ChatCompletionChunk列表

    方法:
        add_chunk: 添加新的chunk并处理回调/显示
        to_complete_response: 将累积的chunks转换为完整响应
    """

    chunks: List[ChatCompletionChunk] = field(default_factory=list)

    def add_chunk(
        self,
        chunk: ChatCompletionChunk,
        callback: Optional[Callable[[str, bool, int], None]] = None,
        realtime_display: bool = False,
        show_thinking: bool = False
    ):
        """
        添加新的chunk并处理回调和实时显示

        参数:
            chunk: 新的ChatCompletionChunk对象
            callback: 可选的回调函数，签名为 callback(content, is_thinking, index)
            realtime_display: 是否实时显示内容
            show_thinking: 是否显示推理内容

        处理逻辑:
            1. 将chunk添加到累积列表
            2. 对所有choice执行回调（如果提供）
            3. 实时显示第一个choice的内容（如果启用）
        """
        self.chunks.append(chunk)
        if not chunk.choices:
            return

        # 处理所有choice的回调
        for choice in chunk.choices:
            index = choice.index
            delta = choice.delta

            if callback:
                # 回调推理内容
                if _chatcompletionchunk_delta_has_reasoning(delta):
                    callback(delta.reasoning_content, True, index)
                # 回调普通内容
                if delta.content:
                    callback(delta.content, False, index)

        # 实时显示第一个choice
        if realtime_display:
            delta = chunk.choices[0].delta
            if _chatcompletionchunk_delta_has_reasoning(delta):
                self._handle_realtime_display(delta.reasoning_content, True, show_thinking)
            if delta.content:
                self._handle_realtime_display(delta.content, False, False)

    def to_complete_response(self) -> ChatCompletion:
        """
        将累积的chunks转换为完整的ChatCompletion对象

        返回:
            完整的ChatCompletion对象
        """
        return chunks_to_complete_response(self.chunks)
