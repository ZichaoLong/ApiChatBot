"""
Httpx到OpenAI格式转换模块

提供将原始httpx响应转换为OpenAI SDK格式的工具函数。
适用于OpenAIHttpxChatBot，将原始HTTP响应转换为OpenAI类型。

主要功能:
- ParseTotalResponse: 解析完整JSON响应为ChatCompletion
- ProcessStreamResponse: 处理同步流式响应
- AsyncProcessStreamResponse: 处理异步流式响应
- SSE（Server-Sent Events）流式数据解析

使用场景:
    当使用httpx直接请求OpenAI兼容API时，需要将原始响应
    转换为OpenAI SDK的类型以保持接口一致性。
"""

import httpx
import json
from typing import Optional, Callable, Dict, Any
from openai.types.chat import ChatCompletion, ChatCompletionChunk, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice
from openai.types.chat.chat_completion_chunk import ChoiceDelta
from openai.types.chat.chat_completion_chunk import ChoiceDeltaFunctionCall, ChoiceDeltaToolCall
from openai.types.completion_usage import CompletionUsage

from . import _OpenAI

__all__ = ['ParseTotalResponse', 'ProcessStreamResponse', 'AsyncProcessStreamResponse']


def ParseTotalResponse(response: Dict[str, Any]) -> ChatCompletion:
    """
    将httpx返回的JSON响应转换为ChatCompletion对象

    参数:
        response: 原始JSON响应字典

    返回:
        ChatCompletion对象
    """
    return ChatCompletion(
        id=response["id"],
        choices=[
            Choice(
                finish_reason=choice["finish_reason"],
                index=choice["index"],
                logprobs=choice.get("logprobs"),
                message=ChatCompletionMessage(**choice['message'])
                # message=ChatCompletionMessage(
                #     content=choice["message"]["content"],
                #     role=choice["message"]["role"],
                #     function_call=choice["message"].get("function_call"),
                #     tool_calls=choice["message"].get("tool_calls")
                # )
            )
            for choice in response["choices"]
        ],
        created=response["created"],
        model=response["model"],
        object=response["object"],
        system_fingerprint=response.get("system_fingerprint"),
        usage=CompletionUsage(
            completion_tokens=response["usage"]["completion_tokens"],
            prompt_tokens=response["usage"]["prompt_tokens"],
            total_tokens=response["usage"]["total_tokens"]
        ) if "usage" in response else None
    )

def _parse_chunk_data(chunk_response: str) -> ChatCompletionChunk:
    """
    解析单个SSE chunk数据并转换为ChatCompletionChunk对象

    处理OpenAI流式响应的单个数据块，包括delta、function_call、
    tool_calls、reasoning_content等字段。

    参数:
        chunk_response: SSE数据块的JSON字符串

    返回:
        ChatCompletionChunk对象
    """
    chunk_json = json.loads(chunk_response)

    # 处理choices
    choices = []
    for choice_data in chunk_json.get("choices", []):
        delta_data = choice_data.get("delta", {})
        # 处理function_call
        function_call = None
        if "function_call" in delta_data:
            fc_data = delta_data["function_call"]
            function_call = ChoiceDeltaFunctionCall(
                arguments=fc_data.get("arguments"),
                name=fc_data.get("name")
            )
        # 处理tool_calls
        tool_calls = None
        if "tool_calls" in delta_data:
            tool_calls = []
            for tc_data in delta_data["tool_calls"]:
                tool_call = ChoiceDeltaToolCall(
                    index=tc_data.get("index"),
                    id=tc_data.get("id"),
                    function=tc_data.get("function"),
                    type=tc_data.get("type")
                )
                tool_calls.append(tool_call)
        # 创建ChoiceDelta对象
        delta = dict(
            content=delta_data.get("content"),
            function_call=function_call,
            role=delta_data.get("role"),
            tool_calls=tool_calls,
            refusal=delta_data.get("refusal")  # 处理refusal字段
        )
        if delta_data.get('reasoning_content'):
            delta['reasoning_content'] = delta_data['reasoning_content']
        delta = ChoiceDelta(**delta)
        # 创建ChunkChoice对象
        choice = ChunkChoice(
            delta=delta,
            finish_reason=choice_data.get("finish_reason"),
            index=choice_data.get("index", 0),
            logprobs=choice_data.get("logprobs")
        )
        choices.append(choice)
    # 处理usage信息
    usage = None
    usage_data = chunk_json['usage'] if 'usage' in chunk_json else None
    if usage_data:
        usage = CompletionUsage(
            completion_tokens=usage_data.get("completion_tokens", 0),
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0),
            completion_tokens_details=usage_data.get("completion_tokens_details"),
            prompt_tokens_details=usage_data.get("prompt_tokens_details")
        )

    return ChatCompletionChunk(
        id=chunk_json["id"],
        choices=choices,
        created=chunk_json["created"],
        model=chunk_json["model"],
        object=chunk_json["object"],
        system_fingerprint=chunk_json.get("system_fingerprint"),
        usage=usage,
        service_tier=chunk_json.get("service_tier")  # 处理service_tier字段
    )


def _process_stream_line(
    line: str,
    sa: _OpenAI.StreamAccumulator,
    callback: Optional[Callable[[str, bool, int], None]] = None,
    realtime_display: bool = True,
    show_thinking: bool = True
):
    """
    处理单行SSE流数据的通用逻辑

    解析 "data: " 开头的SSE行，将chunk添加到累加器。

    参数:
        line: SSE数据行
        sa: OpenAI流式累加器
        callback: 可选的回调函数
        realtime_display: 是否实时显示
        show_thinking: 是否显示思考过程

    返回:
        bool: True表示流结束（遇到[DONE]），False表示继续
    """
    if line.startswith('data: '):
        chunk_data = line[6:]  # 移除 'data: ' 前缀
        if chunk_data.strip() == '[DONE]':
            return True  # 表示结束
        try:
            chunk = _parse_chunk_data(chunk_data)
            sa.add_chunk(chunk, callback, realtime_display, show_thinking)
        except json.JSONDecodeError:
            pass  # 忽略无效的JSON
        # except Exception as e:
        #     print(f"处理chunk时出错: {e}")
    return False  # 表示继续

def ProcessStreamResponse(
    line_iterator,
    callback: Optional[Callable[[str, bool, int], None]] = None,
    realtime_display: bool = True,
    show_thinking: bool = True
):
    """
    处理同步流式响应

    迭代SSE行，累积chunk并转换为完整的ChatCompletion。

    参数:
        line_iterator: 行迭代器（如 response.iter_lines()）
        callback: 可选的回调函数
        realtime_display: 是否实时显示
        show_thinking: 是否显示思考过程

    返回:
        完整的ChatCompletion对象
    """
    sa = _OpenAI.StreamAccumulator()
    for line in line_iterator:
        if _process_stream_line(line, sa, callback, realtime_display, show_thinking):
            break
    return sa.to_complete_response()


async def AsyncProcessStreamResponse(
    line_iterator,
    callback: Optional[Callable[[str, bool, int], None]] = None,
    realtime_display: bool = True,
    show_thinking: bool = True
):
    """
    处理异步流式响应

    异步迭代SSE行，累积chunk并转换为完整的ChatCompletion。

    参数:
        line_iterator: 异步行迭代器（如 response.aiter_lines()）
        callback: 可选的回调函数
        realtime_display: 是否实时显示
        show_thinking: 是否显示思考过程

    返回:
        完整的ChatCompletion对象
    """
    sa = _OpenAI.StreamAccumulator()
    async for line in line_iterator:
        if _process_stream_line(line, sa, callback, realtime_display, show_thinking):
            break
    return sa.to_complete_response()


