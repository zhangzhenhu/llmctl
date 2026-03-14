# llmctl

LLM 服务验证 CLI 工具，用于测试和验证各种大语言模型服务。支持 OpenAI、Gemini、Claude、Ollama、DeepSeek 等多种服务商，并兼容任何实现了 OpenAI API 格式的服务。

## 功能特性

- **多服务商支持**：OpenAI、Gemini (Google)、Anthropic (Claude)、Ollama、DeepSeek、XAI、Groq、Mistral 等
- **OpenAI 兼容接口**：支持阿里云、DashScope、本地部署等任何兼容 OpenAI API 的服务
- **模型列表**：查看服务商支持的所有模型
- **流式输出**：实时流式返回聊天内容
- **思考/推理模式**：支持兼容服务商的推理功能
- **灵活配置**：支持 YAML/JSON 配置文件或命令行参数

## 安装

### 使用 Homebrew（推荐）

```bash
# 安装预编译的二进制文件
brew install zhangzhenhu/llmctl/llmctl
```

支持的平台：macOS (Apple Silicon & Intel)、Linux (arm64 & x86_64)

如果你的平台不被支持，可以从源码编译安装：

```bash
brew install --build-from-source zhangzhenhu/llmctl/llmctl
```

### 从源码构建

```bash
git clone https://github.com/zhangzhenhu/llmctl.git
cd llmctl
cargo build --release
./target/release/llmctl --help
```

### 预编译二进制文件

可从 [GitHub Releases](https://github.com/zhangzhenhu/llmctl/releases) 下载适用于 macOS、Linux、Windows 的预编译二进制文件，解压后将 llmctl 可执行文件添加到系统 $PATH 中即可使用

## 快速开始

### 1. 创建配置文件

```bash
# 初始化 YAML 格式配置
llmctl --init yaml

# 或 JSON 格式
llmctl --init json
```

### 2. 编辑配置文件

```yaml
# llm.yaml
provider: "openai-compatible"
base_url: "https://api.openai.com/v1"
api_key: "your-api-key-here"
model: "gpt-4o"
stream: true
context:
  - role: "system"
    content: "You are a helpful assistant."
  - role: "user"
    content: "Hello, world!"
```

### 3. 开始聊天

```bash
llmctl -c llm.yaml
```

## 使用方法

### 命令行选项

```bash
llmctl [选项]

选项:
  -c, --config <路径>          配置文件路径
  -m, --model <字符串>         模型名称
  -l, --list                   列出可用模型
  -p, --provider <字符串>      服务商名称
  -u, --url <字符串>           API 基础地址
  -s, --secret <字符串>        API 密钥
      --stream                 启用流式输出
      --init <格式>            初始化配置文件 (yaml/json)
      --init-path <路径>       自定义配置文件路径
  -t, --convert <输入>         转换配置文件格式
```

### 使用示例

#### 列出可用模型

```bash
llmctl -c llm.yaml -l
```

#### 指定模型聊天

```bash
llmctl -c llm.yaml -m gpt-4-turbo
```

#### 流式输出

```bash
llmctl -c llm.yaml --stream
```

#### 使用环境变量设置 API 密钥

```bash
export LLM_API_KEY="your-api-key"
llmctl -c llm.yaml
```

### 支持的服务商

| 服务商 | 配置值 | 备注 |
|--------|--------|------|
| OpenAI | `openai` | |
| Google Gemini | `gemini` 或 `google` | |
| Anthropic Claude | `anthropic` 或 `claude` | |
| Ollama | `ollama` | 本地部署 |
| DeepSeek | `deepseek` | |
| XAI | `xai` | |
| Groq | `groq` | |
| Mistral | `mistral` | |
| OpenAI 兼容 | `openai-compatible`、`aliyun`、`dashscope` | 自定义端点 |

## 配置参考

### YAML 格式

```yaml
provider: "openai-compatible"    # 服务商名称
base_url: "https://api.openai.com/v1"  # API 基础地址
api_key: "sk-..."                 # API 密钥
model: "gpt-4o"                   # 模型名称
stream: false                     # 启用流式输出
max_tokens: 2048                 # 最大生成 token 数
temperature: 0.7                  # 采样温度 (0-2)
top_p: 1.0                        # top-p 采样
top_k: 40                         # top-k 采样
timeout_seconds: 60               # 请求超时时间
system: "You are a helpful assistant."  # 系统提示词
context:                          # 对话历史
  - role: "user"
    content: "你的消息"
```

### 推理配置

支持思考/推理功能的服务商：

**OpenAI** (使用 `reasoning_effort`):
```yaml
enable_thinking: true
reasoning_effort: "high"          # low, medium, high
```

**Anthropic** (使用 `reasoning_budget_tokens`):
```yaml
enable_thinking: true
reasoning_budget_tokens: 1024      # 思考过程最大 token 数
```

## 错误处理

工具会为常见问题提供友好的错误提示：

- API 密钥无效
- 网络错误（DNS 解析失败、连接被拒绝、请求超时）
- 模型不存在
- 请求频率限制
- 服务器内部错误

## 开发

### 构建

```bash
cargo build
```

### 运行测试

```bash
cargo test
```

### 发布版本构建

```bash
cargo build --release
```

## 许可证

MIT License
