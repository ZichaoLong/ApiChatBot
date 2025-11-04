# CLAUDE.md - Claude Code 快速参考

## 核心架构
3层 + 双向格式转换：`BaseChatBot`（格式转换）→ `BaseSDKChatBot`（流式处理）→ 提供商实现

## SDK源码访问
- 配置：`.claude/settings.local.json`
- SDK路径：`site-packages/{openai,anthropic,google/genai}`
- ⚠️ 验证API行为时应主动查阅SDK源码，不要猜测

## 提供商差异速查表

| 特性 | OpenAI | Google | Anthropic |
|-----|--------|--------|-----------|
| **输入格式** | `role+content` | `role+content`（自动转→`parts`） | `role+content` |
| **输出格式** | 统一格式字典（含`_raw_dict`） | 同左 | 同左 |
| **系统指令** | 插入消息列表首 | `config.system_instruction` | `system`参数 |
| **思考功能** | 自动（DeepSeek） | `thinking_config` | `thinking`参数 |
| **异步客户端** | `AsyncOpenAI()` | `client.aio.models` | `AsyncAnthropic()` |
| **流式chunk** | Delta增量 | 完整结构 | 事件流 |
| **关闭方法** | `close()` / `aclose()` | `close()` / `aclose()` (内部用`aio.aclose()`) | `close()` / `aclose()` |

## 代码约定

### 格式转换规则
- **输入**：`_normalize_messages()` - Google/Anthropic重写
  - Google: `_Google.py::_convert_to_google_format()` - 格式转换（content→parts）并过滤元数据
  - Anthropic: `_Anthropic.py::_normalize_messages()` - 仅过滤元数据字段
- **输出**：`_to_unified_format()` - 所有提供商必须实现，使用`model_dump(exclude_none=True)`生成`_raw_dict`

### 特殊限制
- **Httpx模式**：仅OpenAI接口（`interfacetype='openai'`）
- **SSL验证**：所有客户端支持`is_ssl_verify=False`，Google SDK需特殊处理（`Client.py::ApiSDKClient.__init__()`）
- **类型检查**：运行时（typeguard），不用mypy

## 扩展新提供商

参见 [ARCHITECTURE.md - 扩展新提供商](./ARCHITECTURE.md#-扩展新提供商)

## 命令

```bash
# 测试
python tests/test_providers.py --help

# 类型检查
# 项目使用 typeguard（运行时），不用 mypy
```

## 文档分工
- **CLAUDE.md**（本文件）：快速参考
- **ARCHITECTURE.md**：设计框架
- **README.md**：用户手册
- **代码docstring**：API文档
