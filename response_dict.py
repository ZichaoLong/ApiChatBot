#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ResponseDict - 简化响应字典的显示
"""

from typing import Any, Dict


class ResponseDict(dict):
    """
    响应字典的包装类，在交互环境中只显示 role 和 content

    继承自dict，完全兼容字典操作，但在打印时只显示关键的 role 和 content 字段，
    隐藏 _raw_dict 等冗余信息。所有字段仍然可以正常访问。

    使用示例:
        >>> response = chatbot.Chat(...)
        >>> print(response)  # 只显示 {'role': '...', 'content': '...'}
        >>> messages.append(response)  # 可直接追加
        >>> response['_raw_dict']  # 仍可访问所有字段
        >>> response.get('_usage')  # 字典方法都可用
    """

    def __repr__(self) -> str:
        """只显示 role 和 content 字段"""
        return str({
            'role': self.get('role', 'unknown'),
            'content': self.get('content', '')
        })

    def __str__(self) -> str:
        """字符串表示，与repr相同"""
        return self.__repr__()

    def full_repr(self) -> str:
        """返回完整的字典表示（包括所有字段）"""
        return dict.__repr__(self)

def _create_empty_unified_response(role: str = 'assistant', model: str = 'unknown') -> ResponseDict:
    """
    创建空的统一格式响应

    当SDK返回空响应或无有效内容时使用。

    参数:
        role: 角色名（'assistant' 或 'model'）
        model: 模型名称

    返回:
        空的统一格式ResponseDict
    """
    return ResponseDict({
        'role': role,
        'content': '',
        '_usage': {},
        '_model': model,
        '_finish_reason': 'unknown'
    })


