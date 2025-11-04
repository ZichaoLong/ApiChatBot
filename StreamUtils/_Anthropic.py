"""
Anthropic Claude流式响应处理模块

提供Anthropic Claude流式Message响应的累加和转换功能。
将流式RawMessageStreamEvent累积并转换为完整的Message对象。

主要功能:
- StreamAccumulator: 累加流式事件并转换为完整响应
- 处理多种事件类型（message_start, content_block_*, message_delta等）
- 支持text、thinking、tool_use等多种内容块类型
- 实时显示和回调支持
"""

import json
from typing import List, Optional, Callable, Union, Dict, Any
from dataclasses import dataclass, field
from anthropic.types import Message, RawMessageStreamEvent
from .common_utils import RealTimeDisplayHandler

__all__ = ['StreamAccumulator']


def chunks_to_complete_response(chunks: List[RawMessageStreamEvent]) -> Message:
    """
    将流式事件列表转换为完整的Message

    处理Anthropic的流式事件序列，累积内容块并构造完整的Message对象。

    参数:
        chunks: RawMessageStreamEvent对象列表

    返回:
        完整的Message对象

    异常:
        ValueError: 如果chunks为空

    事件类型:
        - message_start: 消息开始，提供基础信息
        - content_block_start: 内容块开始
        - content_block_delta: 内容块增量更新
        - content_block_stop: 内容块结束
        - message_delta: 消息级别更新
        - message_stop: 消息结束
    """
    if not chunks:
        raise ValueError("No chunks provided")

    # 初始化响应数据
    message_data = {
        "id": "",
        "type": "message",
        "role": "assistant",
        "content": [],
        "model": "",
        "stop_reason": None,
        "stop_sequence": None,
        "usage": {"input_tokens": 0, "output_tokens": 0}
    }

    # 用于累积内容块（按index分组）
    content_blocks = {}

    for chunk in chunks:
        event_type = chunk.type

        if event_type == "message_start":
            # 设置消息的基本信息
            message = chunk.message
            message_data.update({
                "id": message.id,
                "model": message.model,
                "role": message.role,
                "usage": message.usage.dict() if message.usage else {"input_tokens": 0, "output_tokens": 0}
            })

        elif event_type == "content_block_start":
            # 开始新的内容块
            content_block = chunk.content_block
            index = chunk.index

            block_data = {"type": content_block.type,}

            # 根据类型初始化特定字段
            if content_block.type == "text":
                block_data["text"] = ""
            elif content_block.type == "tool_use":
                block_data["id"] = getattr(content_block, "id", None)
                block_data["name"] = getattr(content_block, "name", None)
                block_data["input"] = {}
            elif content_block.type == "thinking":
                block_data["thinking"] = ""
                block_data["signature"] = None

            # 通用可选字段
            if hasattr(content_block, "citations"):
                block_data["citations"] = content_block.citations

            content_blocks[index] = block_data

        elif event_type == "content_block_delta":
            # 累积内容块的增量更新
            index = chunk.index
            delta = chunk.delta

            assert index in content_blocks,"index 应在 content_blocks 中"
            if delta.type == "text_delta":
                content_blocks[index]["text"] += delta.text
            elif delta.type == "input_json_delta":
                # 累积工具调用的输入JSON
                if "partial_json" not in content_blocks[index]:
                    content_blocks[index]["partial_json"] = ""
                content_blocks[index]["partial_json"] += delta.partial_json
            elif delta.type == "thinking_delta":
                content_blocks[index]["thinking"] += delta.thinking
            elif delta.type == 'signature_delta':
                content_blocks[index]['signature'] = delta.signature

        elif event_type == "content_block_stop":
            # 完成内容块
            index = chunk.index
            assert index in content_blocks,"index 应在 content_blocks 中"
            if content_blocks[index]["type"] == "tool_use":
                # 解析完整的JSON输入
                if "partial_json" in content_blocks[index]:
                    content_blocks[index]["input"] = json.loads(content_blocks[index]["partial_json"])
                    del content_blocks[index]["partial_json"]
                else:
                    content_blocks[index]["input"] = {}
            # elif content_blocks[index]['type'] in ['text','thinking']:
            # 这种情形前面已经整合好，无需单独处理

        elif event_type == "message_delta":
            # 更新消息级别的信息
            delta = chunk.delta
            if hasattr(delta, "stop_reason") and delta.stop_reason:
                message_data["stop_reason"] = delta.stop_reason
            if hasattr(delta, "stop_sequence") and delta.stop_sequence:
                message_data["stop_sequence"] = delta.stop_sequence
            if hasattr(chunk, "usage") and chunk.usage:
                message_data['usage'] = chunk.usage.dict()
        # 忽略 ping 和 message_stop 事件

    # 构建最终的内容数组
    message_data['content'] = list(content_blocks.values())

    # 创建Message对象
    return Message.model_construct(**message_data)


@dataclass
class StreamAccumulator(RealTimeDisplayHandler):
    """
    Anthropic Claude流式响应累加器

    累积流式RawMessageStreamEvent并提供转换为完整Message的功能。
    继承RealTimeDisplayHandler以支持实时显示。

    属性:
        chunks: 累积的RawMessageStreamEvent列表
    """

    chunks: List[RawMessageStreamEvent] = field(default_factory=list)

    def add_chunk(
        self,
        chunk: RawMessageStreamEvent,
        callback: Optional[Callable[[str, bool, int], None]] = None,
        realtime_display: bool = False,
        show_thinking: bool = False
    ) -> None:
        """
        添加新的流式事件并处理回调和实时显示

        参数:
            chunk: 新的RawMessageStreamEvent对象
            callback: 可选的回调函数
            realtime_display: 是否实时显示内容
            show_thinking: 是否显示thinking内容
        """
        self.chunks.append(chunk)
        chunk_index = len(self.chunks) - 1

        # 提取文本内容和判断是否为thinking
        text_content = ""
        is_thinking = False

        if chunk.type == "content_block_delta":
            delta = chunk.delta
            if delta.type == "text_delta":
                text_content = delta.text
                is_thinking = False
            elif delta.type == "thinking_delta":
                text_content = delta.thinking
                is_thinking = True

        # 执行回调
        if callback and text_content:
            callback(text_content, is_thinking, chunk_index)

        # 实时显示
        if realtime_display and text_content:
            # 如果是thinking内容但不显示thinking，则跳过
            if is_thinking and not show_thinking:
                return
            self._handle_realtime_display(text_content, is_thinking, show_thinking)

    def to_complete_response(self) -> Message:
        """将累积的chunks转换为完整的Message对象"""
        return chunks_to_complete_response(self.chunks)

