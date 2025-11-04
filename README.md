# ApiChatBot - 多提供商 LLM 接口抽象层

统一的聊天机器人 API 封装，支持 OpenAI、Google Gemini、Anthropic Claude。提供一致的接口、统一的消息格式和响应格式。

## 核心特性

- **统一接口**：`Chat()` / `AsyncChat()` 适用所有提供商
- **统一格式**：输入输出使用标准格式，内部自动转换
- **多模式**：SDK / HTTP，同步 / 异步，流式 / 完整
- **思考支持**：DeepSeek、Gemini、Claude 的推理功能

## 快速开始

### 安装依赖

```bash
pip install openai anthropic google-genai httpx python-dotenv typeguard
```

### 基础使用

```python
from ApiChatBot import ChatBotFunc

# 创建聊天机器人实例
ChatBot = ChatBotFunc(interfacetype='openai', use_sdk=True)
chatbot = ChatBot(api_key='sk-xxx', base_url='https://api.openai.com/v1')

# 对话
messages = [{'role': 'user', 'content': '你好'}]
response = chatbot.Chat(model='gpt-4o', messages=messages)

# 或从 .env 配置加载（参见"环境变量配置"章节）
# from ApiChatBot.config import load_default_config
# config = load_default_config()
```

### 消息格式 - 统一标准格式

所有提供商使用统一格式（OpenAI 格式），内部自动转换：

```python
messages = [
    {'role': 'user', 'content': '你好'},
    {'role': 'assistant', 'content': '你好！'},  # 或 'model'（Google）
    {'role': 'user', 'content': '介绍一下你自己'}
]
```

**注**: Google 原生 `parts` 格式仍兼容，但推荐统一格式

### 返回格式 - 可直接追加

返回统一格式字典，可直接追加到消息列表：

```python
response = chatbot.Chat(model='gpt-4o', messages=messages)

# 核心字段
response['role']       # 'assistant' 或 'model'
response['content']    # 回答内容

# 元数据（_前缀）
response['_thinking']  # 思考过程（如有）
response['_usage']     # Token统计
response['_raw_dict']  # 完整SDK响应

# 直接追加
messages.append(response)
```

详见 [ARCHITECTURE.md - 返回格式](./ARCHITECTURE.md#6-返回格式)

## 支持的提供商

### 原生支持
- **OpenAI**: GPT-4o, GPT-4, etc.
- **Google Gemini**: Gemini 2.5 Flash, Gemini 2.5 Pro
- **Anthropic**: Claude Sonnet 4

### OpenAI 兼容接口
多个提供商使用 `interfacetype='openai'`：Moonshot (Kimi),Aliyun (Qwen),DeepSeek,OpenRouter

完整提供商映射见 `config.py`

## 高级特性

### 思考/推理功能

部分提供商支持推理功能，返回的 `response['_thinking']` 包含思考过程：

```python
# DeepSeek: 自动支持，无需配置
chatbot = ChatBot(api_key='sk-xxx', base_url='...')
response = chatbot.Chat(model='deepseek-reasoner', messages=[...])
print(response['_thinking'])  # 思考过程

# Google Gemini: 通过 thinking_budget 参数配置
chatbot = ChatBot(api_key='xxx', base_url='...', thinking_budget=-1)  # -1=自动, 正数则为token上限

# Anthropic Claude: 通过 thinking_budget 参数配置
chatbot = ChatBot(api_key='xxx', base_url='...', thinking_budget=8192)  # token上限
```

**注**: 各提供商的实现细节不同，具体参数和行为见各 ChatBot 类的 `__init__()` 和 `send_request()` 实现

### 常用参数

```python
# 流式响应（实时显示）
response = chatbot.Chat(model='gpt-4o', messages=[...], stream=True)

# 异步调用
response = await chatbot.AsyncChat(model='gpt-4o', messages=[...])

# 禁用 SSL 验证（代理环境）
chatbot = ChatBot(api_key='sk-xxx', is_ssl_verify=False)

# 系统指令
response = chatbot.Chat(model='gpt-4o', messages=[...],
                        system_instruction='你是一个有帮助的助手')
```

## 环境变量配置

可选：通过 `.env` 文件配置 API 密钥和 Base URL：

```bash
cp .env.example .env  # 复制模板并编辑
```

**配置项**:
- **API Keys（必填）**: `OPENAI_API_KEY`, `GEMINI_API_KEY`, `ANTHROPIC_API_KEY`
- **Base URLs（可选）**: `OPENAI_BASE_URL`, `GOOGLE_BASE_URL`, `ANTHROPIC_BASE_URL`
  - 不配置则使用 `config.py` 的默认值
  - 可用于代理或自定义端点

**使用配置**:
```python
from ApiChatBot.config import load_default_config

config = load_default_config()
chatbot = ChatBot(
    api_key=config['API_KEYS']['openai'],
    base_url=config['BASE_URLS']['openai']
)
```

## 测试

```bash
python tests/test_providers.py --help
```

## 架构文档

详细技术设计见 **[ARCHITECTURE.md](./ARCHITECTURE.md)**：
- 3层架构结构 + 双向格式转换
- 提供商差异对比（7个维度）
- StreamUtils 工作机制
- 扩展指南和调试技巧

## 开发约定

- **类型检查**: 使用运行时类型检查（typeguard），不使用 mypy
- **扩展提供商**: 参见 [ARCHITECTURE.md - 扩展新提供商](./ARCHITECTURE.md#-扩展新提供商)
- **代码文档**: 关键方法的 docstring 包含详细调用链
