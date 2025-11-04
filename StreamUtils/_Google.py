"""
Google Gemini流式响应处理模块

提供Google Gemini流式GenerateContentResponse响应的累加和转换功能。
将流式响应累积并转换为完整的GenerateContentResponse对象。

主要功能:
- StreamAccumulator: 累加流式chunk并转换为完整响应
- 合并文本parts（包括thinking和普通文本）
- 处理citation metadata
- 实时显示和回调支持

参考:
    Google genai流式响应处理: https://github.com/googleapis/python-genai/issues/1092
"""

from dataclasses import dataclass, field
from typing import Callable, Optional, List, Dict
from google.genai import types

from .common_utils import RealTimeDisplayHandler

__all__ = ['StreamAccumulator']


def _consolidate_contents(contents: List[types.Content]) -> types.Content:
    """
    合并多个Content对象为单个Content

    只需要合并文本类型的parts，其他类型的parts（如图片等）不会跨越chunks。
    相邻的同类型文本parts会被合并（thinking和普通文本分别合并）。

    参数:
        contents: Content对象列表

    返回:
        合并后的单个Content对象

    参考:
        https://github.com/googleapis/python-genai/issues/1092
    """
    if not contents:
        return types.Content(parts=[], role="model")

    consolidated_role = contents[0].role or "model"

    # 收集所有parts
    all_parts = []
    for content in contents:
        if content.parts:
            all_parts.extend(content.parts)

    # 合并相邻的同类型文本parts
    consolidated_parts = []
    i = 0

    while i < len(all_parts):
        current_part = all_parts[i]

        if hasattr(current_part, 'text') and current_part.text:
            # 判断是否为thinking类型
            is_thinking = bool(getattr(current_part, 'thought', False))
            consolidated_text = current_part.text

            # 合并后续相同类型的文本parts
            j = i + 1
            while (j < len(all_parts) and
                   hasattr(all_parts[j], 'text') and all_parts[j].text and
                   (hasattr(all_parts[j], 'thought') and all_parts[j].thought) == is_thinking):
                consolidated_text += all_parts[j].text
                j += 1

            # 创建合并后的Part
            if is_thinking:
                consolidated_parts.append(types.Part(text=consolidated_text, thought=True))
            else:
                consolidated_parts.append(types.Part(text=consolidated_text))
            i = j
        else:
            # 非文本类型（如图片）直接添加，不跨chunks
            consolidated_parts.append(current_part)
            i += 1

    return types.Content(parts=consolidated_parts, role=consolidated_role)


def _consolidate_citation_metadatas(
    citation_metadatas: List[Optional[types.CitationMetadata]]
) -> Optional[types.CitationMetadata]:
    """
    合并多个CitationMetadata对象

    收集所有citation sources并按startIndex排序。

    参数:
        citation_metadatas: CitationMetadata对象列表

    返回:
        合并后的CitationMetadata对象，如果没有有效引用则返回None
    """
    if not citation_metadatas:
        return None

    # 过滤掉None值和空的citation metadata
    valid_metadatas = [cm for cm in citation_metadatas if cm is not None]
    if not valid_metadatas:
        return None

    # 收集所有的citation sources
    all_citation_sources = []
    for metadata in valid_metadatas:
        all_citation_sources.extend(metadata.citations)

    if not all_citation_sources:
        return None

    # 按startIndex排序，确保引用顺序正确
    all_citation_sources.sort(key=lambda cs: cs.startIndex)

    return types.CitationMetadata(all_citation_sources)


def chunks_to_complete_response(chunks: List[types.GenerateContentResponse]) -> types.GenerateContentResponse:
    """
    将流式chunk列表转换为完整的GenerateContentResponse

    遍历所有chunk，按candidate分组累积内容、元数据和引用信息，
    最终构造完整的GenerateContentResponse对象。

    参数:
        chunks: GenerateContentResponse对象列表

    返回:
        完整的GenerateContentResponse对象

    异常:
        ValueError: 如果chunks为空

    处理逻辑:
        1. 按candidate_idx分组累积contents
        2. 合并文本parts和citation metadata
        3. 使用最后一个chunk的全局信息构建完整响应
    """
    if not chunks:
        raise ValueError("No chunks received")

    # 按candidate索引分组收集数据
    candidates_contents: Dict[int, List] = {}              # 内容列表
    candidates_metadata: Dict[int, Dict] = {}              # 元数据
    candidates_citation_metadatas: Dict[int, List] = {}    # 引用信息列表

    for chunk in chunks:

        if not chunk.candidates:
            continue

        for candidate_idx,candidate in enumerate(chunk.candidates):
            assert candidate.index is None, "当前(2025) Google Gemini API 返回值中，这一字段总是 None"

            candidate_dict = candidate.dict()
            candidate_dict.pop('content')
            # 初始化 candidate 的内容信息
            if candidate_idx not in candidates_contents:
                candidates_contents[candidate_idx] = []
            # 累积 candidate 的内容信息
            if not candidate.content:
                continue
            else:
                candidates_contents[candidate_idx].append(candidate.content)

            if 'citation_metadata' in candidate_dict:
                candidate_dict.pop('citation_metadata')
                # 初始化 candidate 的引用信息
                if candidate_idx not in candidates_citation_metadatas:
                    candidates_citation_metadatas[candidate_idx] = []
                candidates_citation_metadatas[candidate_idx].append(candidate.citation_metadata)

            # 初始化 candidate 的元信息
            if candidate_idx not in candidates_metadata:
                candidates_metadata[candidate_idx] = {}
            # 更新candidate的元信息（每次chunk都更新，保存最新状态）
            candidates_metadata[candidate_idx].update(candidate_dict)

    complete_candidates = []

    for candidate_idx in sorted(candidates_contents.keys()):
        # 整合 contents
        content = _consolidate_contents(candidates_contents[candidate_idx])

        # 提取元信息
        metadata = candidates_metadata[candidate_idx]

        # 提取citation_metadata
        citation_metadata = _consolidate_citation_metadatas(
            candidates_citation_metadatas[candidate_idx])
        if citation_metadata:
            metadata['citation_metadata'] = citation_metadata

        # 创建新 candidate
        complete_candidate = types.Candidate(content=content, **metadata)

        complete_candidates.append(complete_candidate)

    # 使用最后一个 chunk 的全局信息构建完整响应
    last_chunk_dict = chunks[-1].dict()
    last_chunk_dict.pop('candidates')
    complete_response = types.GenerateContentResponse(
        candidates=complete_candidates, **last_chunk_dict)

    return complete_response

@dataclass
class StreamAccumulator(RealTimeDisplayHandler):
    """
    Google Gemini流式响应累加器

    累积流式GenerateContentResponse并提供转换为完整响应的功能。
    继承RealTimeDisplayHandler以支持实时显示。

    属性:
        chunks: 累积的GenerateContentResponse列表
    """

    chunks: List[types.GenerateContentResponse] = field(default_factory=list)

    def add_chunk(
        self,
        chunk: types.GenerateContentResponse,
        callback: Optional[Callable[[str, bool, int], None]] = None,
        realtime_display: bool = False,
        show_thinking: bool = False
    ) -> None:
        """
        添加新的 chunk 并处理回调和实时显示

        【调用链】
        _handle_sync()
            ↓
        for chunk in stream:
            ↓
        add_chunk(chunk) [当前] × N
            ↓
        to_complete_response()

        【内部调用】
        1. self.chunks.append(chunk)
           - 累积 chunk 到列表

        2. 遍历所有 candidate 和 parts:
           for candidate in chunk.candidates:
               for part in candidate.content.parts:
                   ├─ 判断是否为思考: is_thinking = part.thought
                   ├─ callback(part.text, is_thinking, candidate_idx)
                   └─ self._handle_realtime_display(...)

        【被调用】
        - BaseSDKChatBot._handle_sync() - 同步流式处理
        - BaseSDKChatBot._handle_async() - 异步流式处理
        - GoogleSDKChatBot._handle_complete_response() - 完整响应也复用此方法

        【参数】
        chunk: GenerateContentResponse 对象
              - candidates[0].content.parts: Part 列表
              - part.text: 文本内容
              - part.thought: 是否为思考内容（True/False）
        callback: 回调函数，签名 callback(content: str, is_thinking: bool, index: int)
        realtime_display: 是否实时打印到终端
        show_thinking: 是否显示思考内容

        【返回】
        None（副作用：修改 self.chunks，执行回调和打印）

        【Google 特点】
        - 完整结构：每个 chunk 都是完整的 GenerateContentResponse
        - Parts 列表：一个 chunk 可能包含多个 part
        - 思考标识：通过 part.thought 属性区分思考和普通文本
        - 类型一致：chunk 与 complete response 类型相同
        """
        self.chunks.append(chunk)
        if not chunk.candidates:
            return

        # 对每个candidate处理callback和realtime_display
        for candidate_idx, candidate in enumerate(chunk.candidates):
            if not candidate.content or not candidate.content.parts:
                continue
            assert candidate.index is None, "当前(2025) Google Gemini API 返回值中，这一字段总是 None"

            for part in candidate.content.parts:
                if not part.text:
                    continue
                is_thinking = bool(getattr(part, 'thought', False))
                text = part.text

                # 执行回调
                if callback:
                    callback(text, is_thinking, candidate_idx)

                # 实时显示第一个candidate
                if realtime_display and candidate_idx == 0:
                    self._handle_realtime_display(text, is_thinking, show_thinking)

    def to_complete_response(self) -> types.GenerateContentResponse:
        """
        将累积的 chunks 转换为完整的 GenerateContentResponse 对象

        【调用链】
        _handle_sync()
            ↓
        for chunk in stream:
            add_chunk(chunk) × N
            ↓
        to_complete_response() [当前]
            ↓
        返回 GenerateContentResponse

        【内部调用】
        chunks_to_complete_response(self.chunks)
            ↓
        处理逻辑:
            1. 按 candidate_idx 分组收集:
               ├─ contents: 所有 Content 对象列表
               ├─ metadata: 元数据（finish_reason, safety_ratings等）
               └─ citation_metadatas: 引用信息列表
            2. 合并同类型的文本 parts:
               ├─ 相邻的普通文本 parts 合并
               └─ 相邻的思考 parts 合并
            3. 合并引用信息（按 startIndex 排序）
            4. 使用最后 chunk 的全局信息构建完整响应

        【被调用】
        - BaseSDKChatBot._handle_sync() - 流式处理完成后
        - BaseSDKChatBot._handle_async() - 异步流式处理完成后

        【返回】
        GenerateContentResponse 对象:
            - candidates: 合并后的完整候选
            - usage_metadata: 从最后 chunk
            - prompt_feedback: 从最后 chunk

        【Google 特点】
        - 完整结构合并：将多个完整的 GenerateContentResponse 合并
        - Parts 合并：相邻同类型的文本 parts 会被合并
        - Citation 处理：收集并排序所有引用信息
        - 类型一致：返回类型与输入 chunk 类型相同
        """
        return chunks_to_complete_response(self.chunks)

