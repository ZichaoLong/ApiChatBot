"""
流式响应处理工具模块

本模块提供了各LLM提供商的流式响应处理工具：
- OpenAI: 流式ChatCompletion处理
- Google: 流式GenerateContentResponse处理
- Anthropic: 流式Message处理
- Httpx2OpenAI: 将原始httpx流式响应转换为OpenAI格式

主要功能：
- StreamAccumulator: 累积流式数据块为完整响应
- 实时显示处理
- 回调函数支持
- 思考过程/推理内容的特殊处理
"""

from . import _OpenAI as OpenAI
from . import _Google as Google
from . import _Anthropic as Anthropic
from . import Httpx2OpenAI
