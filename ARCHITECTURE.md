# ApiChatBot 架构设计

面向开发者的技术文档。详细方法调用见代码 docstring。

---

## 📐 整体架构（3层结构 + 双向格式转换）

```
┌──────────────────────────────────────────────────────────────────┐
│                            用户代码                              │
│  chatbot = ChatBotFunc(interfacetype, use_sdk)                   │
│  messages = [{'role': 'user', 'content': '...'}]  # 统一格式     │
│  response = chatbot.Chat(model, messages, stream=True)           │
│  # response = {'role': 'assistant', 'content': '...', ...}       │
└───────────────────────────────┬──────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────┐
│                       ChatBot抽象基类层                          │
│                                                                  │
│  BaseChatBot                                                     │
│    ├─ realtime_display                【是否实时显示输出】       │
│    ├─ show_thinking                   【是否显示思考过程】       │
│    ├─ close() / aclose()              【已实现，释放客户端资源】 │
│    ├─ Chat() / AsyncChat()            【已实现，用户入口】       │
│    │    ├─ _normalize_messages()      【钩子方法，可选重写】     │
│    │    │  抽象方法（子类必须实现）:                             │
│    │    ├─ send_request()             【发送API请求】            │
│    │    ├─ _handle_sync/_async()      【处理响应】               │
│    │    └─ _to_unified_format()       【响应格式统一化】         │
│    └─ _handle_complete_response()     【抽象方法，处理完整响应】 │
└───────────────────────────────┬──────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────┐
│                     SDKChatBot抽象基类层                         │
│                                                                  │
│  BaseSDKChatBot (extends BaseChatBot)                            │
│    ├─ api_key,base_url,is_async,...   【基本配置成员】           │
│    ├─ client                          【对话客户端，在子类生成】 │
│    │   └─ reset_client()              【已实现】                 │
│    │       └─ @property interfacetype 【抽象成员】               │
│    └─ _handle_sync/_async()           【已实现】                 │
│        ├─ if 流式：                                              │
│        │   └─ sa_factory()            【抽象方法】               │
│        │       ├─ StreamAccumulator.add_chunk() × N              │
│        │       └─ StreamAccumulator.to_complete_response()       │
│        └─ else 完整响应：                                        │
│            └─ _handle_complete_response() 【抽象方法】           │
└───────────────────────────────┬──────────────────────────────────┘
                                │
           ┌────────────────────┼───────────────────┐
           │                    │                   │
           ▼                    ▼                   ▼
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│ OpenAISDKBot     │  │ GoogleSDKBot     │  │ AnthropicSDKBot  │
│                  │  │                  │  │                  │
│ 继承:            │  │ 继承:            │  │ 继承:            │
│ BaseSDKChatBot   │  │ BaseSDKChatBot   │  │ BaseSDKChatBot   │
│                  │  │                  │  │                  │
│ 必须实现:        │  │ 必须实现:        │  │ 必须实现:        │
│ • interfacetype  │  │ • interfacetype  │  │ • interfacetype  │
│ • send_request() │  │ • send_request() │  │ • send_request() │
│    └─ client     │  │    └─ client     │  │    └─ client     │
│ • _to_unified()  │  │ • _to_unified()  │  │ • _to_unified()  │
│ • sa_factory()   │  │ • sa_factory()   │  │ • sa_factory()   │
│ • _handle_comp() │  │ • _handle_comp() │  │ • _handle_comp() │
│                  │  │                  │  │                  │
│                  │  │ 重写:            │  │ 重写:            │
│                  │  │ • _normalize()   │  │ • _normalize()   │
│                  │  │  (格式转换及过滤)│  │   (过滤元数据)   │
│                  │  │ • aclose()       │  │                  │
│                  │  │  (不同实现)      │  │                  │
└──────────────────┘  └──────────────────┘  └──────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│ OpenAIHttpxBot (HTTP实现，不同于SDK实现)                         │
│                                                                  │
│ 继承: BaseChatBot (直接继承，不经过 BaseSDKChatBot)              │
│                                                                  │
│ 实现:                                                            │
│ • client                【Httpx客户端】                          │
│ • close()               【继承，释放Httpx客户端资源】            │
│ • aclose()              【重写，释放Httpx客户端资源】            │
│ • send_request()        【构造HTTP请求】                         │
│ • _handle_sync/_async() 【手动处理HTTP响应】                     │
│ • _to_unified_format()  【响应格式统一化】                       │
│                                                                  │
│ 特点: 使用 Httpx2OpenAI 工具手动解析SSE流，适用于SDK不兼容场景   │
└──────────────────────────────────────────────────────────────────┘

         ┌────────────────StreamUtils辅助组件────────────────┐
         │                                                   │
         ▼                                                   ▼
┌──────────────────────────────┐  ┌──────────────────────────────┐
│ StreamAccumulator (各提供商) │  │ RealTimeDisplayHandler       │
│                              │  │ (实时显示基类)               │
│ • chunks: List[Chunk]        │  │                              │
│ • add_chunk()                │  │ • _thinking_displayed        │
│   ├─ chunks.append(chunk)    │  │ • _answer_displayed          │
│   ├─ callback(...)           │  │ • _handle_realtime_display() │
│   └─ _handle_realtime_...    │  │   ├─ 首次思考: 打印标题      │
│ • to_complete_response()     │  │   ├─ 首次回答: 打印标题      │
│   └─ chunks_to_complete()    │  │   └─ print(text, flush=True) │
│                              │  │                              │
│ 实现位置:                    │  │ 实现位置:                    │
│ • StreamUtils/_OpenAI.py     │  │ • StreamUtils/common_utils.py│
│ • StreamUtils/_Google.py     │  └──────────────────────────────┘
│ • StreamUtils/_Anthropic.py  │
└──────────────────────────────┘
         │
         │ (OpenAIHttpxBot 特殊工具)
         ▼
┌──────────────────────────────┐
│ Httpx2OpenAI                 │
│ (将httpx响应转OpenAI格式)    │
│                              │
│ • ParseTotalResponse()       │
│ • ProcessStreamResponse()    │
│ • AsyncProcessStreamResponse │
│                              │
│ 实现位置:                    │
│ • StreamUtils/Httpx2OpenAI.py│
└──────────────────────────────┘
```

**层次说明**:

- **第1层 BaseChatBot**: 接口定义 + 双向格式转换（`_normalize_messages`, `_to_unified_format`）
- **第2层 BaseSDKChatBot**: 流式处理实现（`_handle_sync/_async`）
- **第3层 提供商实现**: `send_request` + `sa_factory` + `_to_unified_format`
  - Google/Anthropic重写`_normalize_messages`（Google格式转换，Anthropic过滤元数据）
- **辅助组件 StreamUtils**: 流式响应累积（StreamAccumulator）、实时显示（RealTimeDisplayHandler）、HTTP响应转换（Httpx2OpenAI）

**双向格式转换**:
```
用户 → [统一格式] → _normalize_messages() → [提供商格式] → SDK
                                                            ↓
用户 ← [统一格式] ← _to_unified_format() ← [提供商响应] ← SDK
```

---

## 🔄 核心调用流程概览

### 同步流程

```
用户调用 Chat(model, messages, stream, raw_response, ...)
    ↓
1. _normalize_messages(messages) [消息格式标准化]
    → 标准格式 → 提供商格式（Google: content → parts）
    ↓
2. send_request(model, normalized_messages, stream, ...) [子类实现]
    → 调用 SDK 或 HTTP 客户端
    → 返回: Iterator[Chunk] (流式) 或 CompleteResponse (完整)
    ↓
3. _handle_sync(response) [处理响应]
    ↓
    ├─ 流式: StreamAccumulator.add_chunk() × N → to_complete_response()
    └─ 完整: _handle_complete_response()
    ↓
4. _to_unified_format(raw) [格式转换]
    → raw_response=False: 返回统一格式字典
    → raw_response=True: 返回原始SDK对象
```

### 异步流程

```
用户调用 AsyncChat(model, messages, stream, raw_response, ...)
    ↓
1. _normalize_messages(messages) [消息格式标准化]
    ↓
2. send_request(model, normalized_messages, stream, ...) [子类实现]
    → 返回类型根据 is_async 和 stream 决定：
    ├─ 流式: AsyncIterator（无需 await）
    └─ 完整: Awaitable（必须 await）
    ↓
3. await _handle_async(response) [异步处理响应]
    ↓
    ├─ isinstance(inputs, AsyncIterator): 异步流式处理
    └─ else: await inputs → 复用同步逻辑
    ↓
4. _to_unified_format(raw) [格式转换]
    → 根据 raw_response 参数返回
```

**详细调用链**: 参见各方法的 docstring

---

## 🔀 使用差异说明

各提供商的底层SDK和API设计不同，本节说明：
1. 用户如何配置（是否已统一）
2. 内部实现位置（方便查看源码）

### 1. 消息格式

**用户配置**：✅ 已统一，所有提供商使用相同格式

```python
# 统一格式 - 适用于所有提供商
messages = [
    {'role': 'user', 'content': 'Hello'},
    {'role': 'assistant', 'content': 'Hi!'}  # Google也可用'model'
]
```

**内部差异**：
- OpenAI: 原生支持 `role + content`
- Google: 原生使用 `role + parts`，内部自动转换
  - 转换实现：`_Google.py::_convert_to_google_format()`
  - 转换时过滤 `_` 开头的元数据字段
  - 向后兼容：Google原生格式仍支持
- Anthropic: 原生支持 `role + content`，但需过滤元数据字段
  - 过滤实现：`_Anthropic.py::_normalize_messages()`

---

### 2. 系统指令

**用户配置**：✅ 已统一，所有提供商使用相同方式

```python
# 在 Chat() 调用时传递 system_instruction 参数
response = chatbot.Chat(
    model='gpt-4o',
    messages=messages,
    system_instruction='你是一个有帮助的助手'
)
```

**内部差异**：
- OpenAI: 插入消息列表开头（`_OpenAI.py::_insert_openai_system_instruction()`）
- Google: 通过配置对象传递（`_Google.py::GoogleSDKChatBot.send_request()` 中设置 `config.system_instruction`）
- Anthropic: 作为独立参数（`_Anthropic.py::AnthropicSDKChatBot.send_request()` 中传递 `system=`）


---

### 3. 思考/推理功能

**用户配置**：⚠️ 部分统一，配置位置相同但参数不同

```python
# DeepSeek: 自动支持，无需配置
chatbot = ChatBot(api_key='sk-xxx', base_url='...')

# Google Gemini: 初始化时配置 thinking_budget
chatbot = ChatBot(
    api_key='xxx', base_url='...',
    thinking_budget=-1  # -1=自动，0=禁用，正数=token上限
)

# Anthropic Claude: 初始化时配置 thinking_budget
chatbot = ChatBot(
    api_key='xxx', base_url='...',
    thinking_budget=8192  # token上限，0或负数=禁用
)

# 统一访问方式
response = chatbot.Chat(model='...', messages=[...])
thinking = response['_thinking']  # 思考过程（如有）
```

**内部差异**：
- DeepSeek: 自动支持，SDK原生提供（`_OpenAI.py::_chatcompletion_message_has_reasoning()` 用于检测）
- Google: 配置 `thinking_config`（`_Google.py::GoogleSDKChatBot.send_request()` 中设置）
- Anthropic: 配置 `thinking` 参数（`_Anthropic.py::AnthropicSDKChatBot.send_request()` 中设置）

---

### 4. 异步调用

**用户配置**：✅ 已统一，所有提供商使用相同方式

```python
# 初始化时指定异步模式
chatbot = ChatBot(api_key='xxx', base_url='...', is_async=True)

# 使用 AsyncChat() 异步调用
response = await chatbot.AsyncChat(model='gpt-4o', messages=[...])

# 关闭异步客户端
await chatbot.aclose()
```

**内部差异**：
- OpenAI: 使用 `AsyncOpenAI()` 客户端（`Client.py::ApiSDKClient` 中处理）
- Google: 使用 `client.aio.models` 接口（`_Google.py::GoogleSDKChatBot.send_request()` 中选择）
- Anthropic: 使用 `AsyncAnthropic()` 客户端（`Client.py::ApiSDKClient` 中处理）

---

### 5. 流式响应

**用户配置**：✅ 已统一，所有提供商使用相同方式

```python
chatbot = ChatBot(
    api_key='xxx', base_url='...',
    realtime_display=True, # 实时打印
    show_thinking=True     # 显示思考过程
)
# 流式调用
response = chatbot.Chat(model='gpt-4o', messages=[...], stream=True)

# 异步流式
response = await chatbot.AsyncChat(model='gpt-4o', messages=[...], stream=True)
```

**内部差异**（Chunk结构和累积逻辑）：
- OpenAI: Delta增量模式（`StreamUtils/_OpenAI.py::chunks_to_complete_response()`）
- Google: 完整结构chunk（`StreamUtils/_Google.py::chunks_to_complete_response()`）
- Anthropic: 事件驱动模式（`StreamUtils/_Anthropic.py::_process_stream_events()`）

**关键方法**：
- `StreamAccumulator.add_chunk()` - 累积chunk、执行回调、实时显示
- `StreamAccumulator.to_complete_response()` - 将chunks转换为完整响应
- `RealTimeDisplayHandler._handle_realtime_display()` - 格式化打印（`StreamUtils/common_utils.py`）


---

### 6. 返回格式

**用户配置**：✅ 已统一，所有提供商返回相同格式

```python
# 默认返回统一格式字典
response = chatbot.Chat(model='gpt-4o', messages=[...])

# 统一格式结构
{
    'role': 'assistant',           # 或 'model' (Google)
    'content': '回答内容',
    '_thinking': '思考内容',       # 可选，如有推理功能
    '_usage': {                    # token统计
        'prompt_tokens': int,
        'completion_tokens': int,
        'total_tokens': int
    },
    '_model': 'gpt-4o',           # 使用的模型
    '_finish_reason': 'stop',     # 停止原因
    '_raw_dict': {...}            # 完整原始SDK响应
}

# 直接追加到消息列表
messages.append(response)

# 获取原始SDK响应对象（如需要）
raw = chatbot.Chat(model='gpt-4o', messages=[...], raw_response=True)
```

**内部差异**（原始SDK响应类型）：
- OpenAI: `ChatCompletion`（`_OpenAI.py::_chatcompletion_to_unified_format()`）
- Google: `GenerateContentResponse`（`_Google.py::_generatecontent_response_to_unified_format()`）
- Anthropic: `Message`（`_Anthropic.py::_message_to_unified_format()`）

---

### 7. 差异速查表

快速查看各提供商在用户使用和内部实现上的差异：

| 特性 | 用户使用 | OpenAI实现 | Google实现 | Anthropic实现 |
|-----|---------|-----------|-----------|--------------|
| **消息格式** | ✅ 统一 | 原生支持 | 自动转换parts | 原生支持 |
| **系统指令** | ✅ 统一 | 插入消息列表 | config配置 | system参数 |
| **思考功能** | ⚠️ 参数不同 | 自动支持 | thinking_budget | thinking_budget |
| **异步调用** | ✅ 统一 | AsyncOpenAI | client.aio | AsyncAnthropic |
| **流式响应** | ✅ 统一 | Delta增量 | 完整结构 | 事件驱动 |
| **返回格式** | ✅ 统一 | ChatCompletion | GenerateContentResponse | Message |

**图例**：
- ✅ 统一：用户使用方式完全相同
- ⚠️ 差异：用户需根据提供商使用不同配置

---

## 🔧 扩展新提供商

```python
# 1. 创建 _NewProvider.py（根目录）
class NewProviderBot(BaseSDKChatBot):
    def __init__(...): pass
    def send_request(...): pass  # 调用SDK
    def sa_factory(): return StreamUtils.NewProvider.StreamAccumulator()
    def _handle_complete_response(...): pass
    def _to_unified_format(raw):
        return {
            'role': 'assistant',
            'content': ...,
            '_model': ...,
            '_finish_reason': ...,
            '_usage': {...},
            '_raw_dict': raw.model_dump(exclude_none=True)  # 必需
        }
    # 可选：重写 _normalize_messages() 如需格式转换

# 2. 创建 StreamUtils/_NewProvider.py
# 3. 更新 __init__.py 的 ChatBotFunc()
# 4. 更新 config.py 映射
# 5. 更新本文档的差异对比表
```

---

## 📚 相关资源

- **代码文档**: 所有关键方法的 docstring 包含详细的调用链说明
- **单元测试**: `tests/test_providers.py` - 覆盖所有提供商和执行模式的测试
- **配置管理**: 各提供商的配置和API密钥管理（参见使用此模块的项目配置）

---

**文档分工**:
- 本文档：宏观架构 + 提供商差异对比
- 代码 docstring：详细的方法调用链和参数说明
