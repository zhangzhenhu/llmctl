# llmctl

LLM 服务验证 CLI 工具，用于测试和验证各种大语言模型服务。支持 OpenAI、Gemini、Claude、Ollama、DeepSeek 等多种服务商，并兼容任何实现了 OpenAI API 格式的服务。

## 2.2.0 更新

- 扩展内置 adapter 支持范围，使其更贴近 vendored `genai 0.7` 的 adapter 集合
- 新增匿名消息快捷输入，支持直接使用 `llmctl "hello"`，无需反复写 `--message`
- 将剩余的 vendored `genai` patch 拆分为更聚焦的 `error-diagnostics` 和 `stream-provider-model` 两个补丁文件

## 功能特性

- **多服务商支持**：OpenAI、Gemini (Google)、Anthropic (Claude)、Ollama、DeepSeek、XAI、Groq、Cohere 等
- **OpenAI 兼容接口**：支持阿里云、DashScope、本地部署等任何兼容 OpenAI API 的服务
- **模型列表**：查看服务商支持的所有模型
- **流式输出**：实时流式返回聊天内容
- **思考/推理能力**：使用 genai 原生 reasoning 捕获，并通过 vendored `extra_body` 透传支持阿里云 `enable_thinking` 等 provider-specific 控制
- **灵活配置**：支持 YAML/JSON 配置文件或命令行参数
- **代理控制**：默认行为是“继承 reqwest/系统代理设置”；可按 profile 配置或用 `--no-proxy` 单次强制直连

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
active_profile: openai_main
defaults:
  no_proxy: false
profiles:
  openai_main:
    adapter: openai
    model: gpt-4o
    api_key_env: OPENAI_API_KEY
    # no_proxy: true
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
llmctl [选项] [MESSAGE...]

选项:
  -c, --config <路径>          配置文件路径（仅支持 v2 YAML/JSON）
  -m, --model <字符串>         模型名称
  -l, --list                   列出可用模型
      --list-adapters          列出支持的 adapter、别名和内置默认值
      --message <字符串>       追加用户消息（可重复）
      [MESSAGE]...             匿名消息快捷输入；剩余单词会自动拼成一条消息
  -p, --adapter <字符串>       adapter 名称或别名
  -P, --profile <名称>         使用配置文件中的 profile 名称（v2）
  -u, --base-url <字符串>      API 基础地址
  -s, --secret <字符串>        API 密钥
  -k, --key <字符串>           API 密钥（--secret 别名）
      --stream                 启用流式输出
      --no-stream              禁用流式输出（仅本次运行）
      --no-proxy               禁用 llmctl 管理的所有 HTTP client 代理
  -v, --version                显示版本
  -i, --init <格式>            初始化配置文件: yaml/json
      --init-path <路径>       自定义配置文件路径
  -t, --convert <输入>         在 YAML/JSON 之间转换 v2 配置
      --api-mode <模式>        API 模式: auto|responses|chat-completions
      --reasoning <模式>       统一推理控制: off|auto|low|medium|high|xhigh|max|budget:<n>
      --dry-run                打印解析后的执行计划，不发请求
      --doctor-config          校验配置并打印诊断
```

代理解析优先级统一为：

- CLI `--no-proxy`
- `profiles.<name>.no_proxy`
- `defaults.no_proxy`
- 否则继承 reqwest/系统代理设置

在 macOS 上，这个“继承系统代理”模式即使 shell 里没有 `*_proxy` 环境变量，也仍可能走系统代理。要强制直连，可使用 `--no-proxy`，或在配置里设置 `no_proxy: true`。

当前运行时架构见 [docs/当前架构.md](docs/当前架构.md)。`docs/` 下较早的设计/稳定化文档保留为迁移历史，不再代表当前实现。

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
# 匿名消息快捷输入
llmctl "hello"

# OpenAI 快速启动（读取 OPENAI_API_KEY）
llmctl --adapter openai --message "hello"

# 阿里云快速启动（读取 ALIYUN_API_KEY）
llmctl --adapter aliyun --message "你好"

# adapter 别名模式同样支持：
llmctl --adapter ds --message "你好"
```

#### 选择配置里的 Profile（v2）

```bash
# 使用配置中的 profiles.anthropic_main
llmctl -c llm.yaml -P anthropic_main --message "hello"

# 保持 profile 不变，但临时切换本次运行的 adapter 身份
llmctl -c llm.yaml -P openai_main --adapter aliyun --message "你好"
```

#### 列出可用模型

```bash
llmctl -c llm.yaml -l
```

现在 `--list` 会额外打印模型列表来源，便于区分结果来自显式自定义 `base_url` 路由，还是默认的 genai client 路由。

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
llmctl --adapter openai --model gpt-5 --reasoning high --message "hello"

# 预算模式
llmctl --adapter gemini --reasoning budget:8000 --message "hello"
```

#### 使用环境变量设置 API 密钥

```bash
export LLM_API_KEY="your-api-key"
llmctl -c llm.yaml
```

### 支持的 Adapter

| Adapter | 常用别名 | 备注 |
|---------|----------|------|
| `openai` | `oi`, `oai`, `openai-compatible` | OpenAI-compatible 协议族 |
| `aliyun` | `ali`, `dashscope`, `ds` | DashScope / 阿里云 |
| `anthropic` | `claude`, `anth` | Anthropic 原生协议 |
| `gemini` | `google`, `gmi` | Gemini 原生协议 |
| `ollama` | `ol` | 本地 Ollama 部署 |
| `deepseek` | `dsk` | |
| `xai` | `grok` | |
| `groq` | `gq` | |
| `cohere` | `co` | Cohere 原生协议 |
| `fireworks` | `fw` | |
| `together` | `tg` | |
| `zai` | `zhipu`, `zhi` | Z.ai / 智谱 |
| `aihubmix` | `ahm` | OpenAI-compatible 网关 |
| `mimo` |  | 小米 Mimo |
| `moonshot` |  | Moonshot AI |
| `nebius` |  | Nebius AI Studio |
| `ollama_cloud` | `ollama-cloud` | Ollama 云端接口 |
| `vertex` |  | Vertex AI namespaced 模型 |
| `github_copilot` | `github-copilot` | GitHub Models 网关 |
| `opencode_go` | `opencode-go` | OpenCode Go 网关 |
| `bedrock_api` | `bedrock-api` | AWS Bedrock Bearer Token API |
| `open_router` | `open-router`, `openrouter` | OpenRouter 网关 |
| `minimax` |  | MiniMax Anthropiс-compatible 接口 |
| `baidu` |  | 百度千帆 |
| `bigmodel` |  | BigModel / 智谱开放平台 |

## 配置参考

### YAML 格式

`v2` 就是 `llmctl` 当前使用的配置 schema 版本。

```yaml
version: 2
active_profile: openai_main
defaults:
  stream: true
  timeout_seconds: 60
  api_mode: auto
  reasoning: auto
profiles:
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
profiles:
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

配置说明：

- 现在只支持 v2 配置结构。
- profile 里的 `reasoning_effort` 仍可继续使用，但它只是 v2 内部的兼容别名。

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
