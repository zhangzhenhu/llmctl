# llmctl

LLM 服务验证 CLI 工具，用于测试和验证各种大语言模型服务。支持 OpenAI、Gemini、Claude、Ollama、DeepSeek 等多种服务商，并兼容任何实现了 OpenAI API 格式的服务。

## 功能特性

- **多服务商支持**：OpenAI、Gemini (Google)、Anthropic (Claude)、Ollama、DeepSeek、XAI、Groq、Mistral 等
- **OpenAI 兼容接口**：支持阿里云、DashScope、本地部署等任何兼容 OpenAI API 的服务
- **模型列表**：查看服务商支持的所有模型
- **流式输出**：实时流式返回聊天内容
- **思考/推理能力**：使用 genai 原生 reasoning 捕获，并通过 vendored `extra_body` 透传支持阿里云 `enable_thinking` 等 provider-specific 控制
- **灵活配置**：支持 YAML/JSON 配置文件或命令行参数

## 安装

### 使用 Homebrew（推荐）

```bash
# 1）先添加 tap
brew tap zhangzhenhu/llmctl

# 2）再安装预编译二进制
brew install llmctl
```

支持的平台：macOS (Apple Silicon & Intel)、Linux (arm64 & x86_64)

说明：

- 某些环境下直接执行 `brew install zhangzhenhu/llmctl/llmctl`，在隐式 tap 解析阶段可能触发额外的 GitHub 认证提示。
- 显式先 `brew tap`，通常可以避免这个提示。

### 使用 Cargo Git 安装

```bash
# 从源码安装，因为 llmctl 使用 vendored genai patch
cargo install --git https://github.com/zhangzhenhu/llmctl.git
```

llmctl 当前不发布到 crates.io。原因是项目使用了一小段 vendored genai patch，用来支持上游 genai 尚未发布的 OpenAI-compatible provider 行为：

- 请求体 `extra_body` 透传，用于阿里云/DashScope `enable_thinking=false` 等控制项；
- 兼容流式 chunk 中的 `usage:null`，避免把有效响应记录成 usage 反序列化错误。

这样可以继续保持 genai 主运行时，同时避免在 llmctl 内自研 OpenAI adapter。补丁清单和升级检查见 `docs/vendored_genai_patch.md`。

如果你只想安装当前本地源码版本：

```bash
cargo install --path .
```

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
version: 2
active_provider: openai_main
providers:
  openai_main:
    adapter: openai
    model: gpt-4o
    api_key_env: OPENAI_API_KEY
context:
  - role: system
    content: You are a helpful assistant.
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
  -c, --config <路径>          配置文件路径（YAML/JSON）
  -m, --model <字符串>         模型名称
  -l, --list                   列出可用模型
      --list-presets           列出内置 provider 预设并退出
      --message <字符串>       追加用户消息（可重复）
  -p, --provider <字符串>      服务商 adapter 或别名
  -P, --profile <名称>         使用配置文件中的 provider profile 名称（v2）
  -u, --url <字符串>           API 基础地址
  -s, --secret <字符串>        API 密钥
  -k, --key <字符串>           API 密钥（--secret 别名）
      --stream                 启用流式输出
      --no-stream              禁用流式输出（仅本次运行）
  -v, --version                显示版本
  -i, --init <格式>            初始化配置文件: yaml/json
      --init-path <路径>       自定义配置文件路径
  -t, --convert <输入>         转换配置文件格式
      --endpoint <模式>        OpenAI 接口模式: auto|responses|chat-completions
      --reasoning <模式>       统一推理控制: off|auto|low|medium|high|xhigh|max|budget:<n>
      --dry-run                打印解析后的执行计划，不发请求
      --doctor-config          校验配置并打印诊断
      --legacy-runtime         显式使用 legacy llm 运行时
      --allow-sdk-default-api  允许 OpenAI endpoint 回退到 SDK 默认行为
```

## 自动化测试

CLI 回归测试用例已集中在：

- `tests/cases/cli_dry_run_cases.yaml`
- `tests/cli_dry_run_cases.rs`

运行方式：

```bash
./scripts/run-cli-dry-run-tests.sh
```

或者：

```bash
cargo test --test cli_dry_run_cases -- --nocapture
```

### 使用示例

#### 无配置文件快速启动

```bash
# OpenAI 快速启动（读取 OPENAI_API_KEY）
llmctl --provider openai --message "hello"

# 阿里云快速启动（读取 ALIYUN_API_KEY）
llmctl --provider aliyun --message "你好"

# provider 别名模式同样支持：
llmctl --provider dashscope --message "你好"
```

#### 选择配置里的 Profile（v2）

```bash
# 使用配置中的 providers.anthropic_main
llmctl -c llm.yaml -P anthropic_main --message "hello"

# 保持 profile 不变，仅临时覆盖 adapter/preset
llmctl -c llm.yaml -P openai_main --provider dashscope --message "你好"
```

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
# 当前默认就是流式输出
llmctl -c llm.yaml

# 仅本次关闭流式
llmctl -c llm.yaml --no-stream
```

#### 统一推理控制

```bash
# 在支持的模型上提升推理强度
llmctl --provider openai --model gpt-5 --reasoning high --message "hello"

# 预算模式
llmctl --provider gemini --reasoning budget:8000 --message "hello"
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
version: 2
active_provider: openai_main
defaults:
  stream: true
  timeout_seconds: 60
  openai_api: auto
  reasoning: auto
providers:
  openai_main:
    adapter: openai
    model: gpt-4o
    api_key_env: OPENAI_API_KEY
    reasoning: high
    max_tokens: 2048
    temperature: 0.7
    top_p: 1.0
context:
  - role: "user"
    content: "你的消息"
```

### 推理配置

统一使用 `reasoning`：

```yaml
defaults:
  reasoning: auto
providers:
  openai_main:
    reasoning: high
  gemini_main:
    reasoning: budget:8000
```

可选值：

- `off`：关闭推理内容捕获，并应用已知服务商控制项，例如阿里云 `enable_thinking=false`
- `auto`：由服务商模型自行决定，客户端做归一化解析
- `low|medium|high|xhigh|max`
- `budget:<n>`：预算模式（对支持预算映射的服务商生效）

兼容说明：

- provider profile 里的旧字段 `reasoning_effort` 仍可继续使用（兼容别名）。

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
