---
name: review-existing-code-analysis
description: Use when doing a comprehensive review of an existing codebase, module, or product flow in this repository, especially when the goal is to systematically audit product features, technical features, modules, files, functions, architecture, implementation quality, Apple-platform compliance, comments, naming, and maintainability, while producing Chinese intermediate artifacts and a final Chinese review report.
---

# Review Existing Code Analysis

用于对现有代码/产品做尽量不漏的系统性审查。

这个 skill 的核心不是“写一份好看的报告”，而是强约束 AI 的工作方式，让 AI 必须按覆盖顺序逐步落盘、逐项检查、反复交叉核对，从而尽量找出所有问题。

## 目标

- 尽量发现所有重要问题，包括全局问题和局部问题。
- 不只找 feature 层问题，也找模块、文件、函数级问题。
- 不只找大问题，也记录低级错误，例如命名、注释、文案、明显不合理的符号或拼写。
- 让中间产物真正参与后续判断，而不是最后回填。

注意：

- 不能承诺 100% 找出全部错误。
- 但必须采用“宁可多记可疑点，也不要轻易放过”的策略。
- 当范围可控时，优先逐文件、逐段、必要时逐行审查。

## 硬约束

- 中间文档和最终报告统一使用中文。
- 必须按步骤执行，并且每完成一步就立刻写入对应文件。
- 不允许先在脑中完成后续多步，再一次性回填多个文件。
- 后续步骤必须显式引用前一步产物，不能绕过。
- 如果发现单点问题，必须横向搜索同类实现，判断是不是系统性问题。
- 如果模块范围较小，必须按文件顺序覆盖审查。

## 总体方法

这个 skill 使用三条审查线并行约束 AI：

### 1. Feature 线

从产品和技术能力出发，检查“系统应该怎么工作”。

关注：

- 产品 feature 是否完整
- technical feature 是否成立
- 一个 feature 跨哪些模块、文件、函数实现
- 跨模块协作是否真的支撑了该 feature
- 产品承诺、UI 文案、真实实现是否一致

### 2. Code 线

从模块、文件、类型、函数出发，检查“代码实际上怎么写的”。

关注：

- 局部 bug
- 逻辑错误
- 边界条件问题
- 架构问题
- 冗余代码
- 死代码
- 命名问题
- 注释问题
- Apple 平台规范问题

### 3. Cross-check 线

把前两条线交叉核对，专门防漏。

关注：

- 有没有 feature 没被代码完整覆盖
- 有没有模块/文件/函数没有被任何 feature 覆盖到
- 有没有单点问题其实是系统性模式
- 有没有高复杂度区域却没有进入任何 finding

## 执行步骤

严格按下面顺序执行。

### Step 0. 固定范围与假设

先写：

- `00-review-scope-and-assumptions.md`

记录：

- review 范围
- 不在范围内的部分
- 静态审查还是运行时审查
- 关键依赖与外部集成
- 当前假设和未确认点

完成并写盘后，才能进入下一步。

### Step 1. 先做 Product Feature Map

写：

- `01-product-feature-map.md`

要求：

- 不只是列页面和功能名。
- 必须逐项写清：设计目的、解决的问题、主流程、关键状态、关键规则、风险点。
- 每个 feature 尽量标出入口、关键用户路径、异常路径、恢复路径。

这一份文件的作用是定义“系统对用户承诺了什么”。

完成并写盘后，才能进入下一步。

### Step 2. 再做 Technical Feature Map

写：

- `02-technical-feature-map.md`

要求：

- 不只是列模块名、类名、函数名。
- 必须逐项写清：职责、真相源、依赖关系、关键规则、实现路径、易错点。
- 对每个 technical feature，尽量标出关联模块、关键文件、关键类型、关键函数。

这一份文件的作用是定义“系统靠什么技术结构实现这些能力”。

完成并写盘后，才能进入下一步。

### Step 3. 做 Feature 线审查

写：

- `03-architecture-and-code-quality-review.md`
- `04-product-and-ux-review.md`

做法：

- 以 `01-product-feature-map.md` 和 `02-technical-feature-map.md` 为 checklist。
- 对每个 product feature 逐一判断是否有问题。
- 对每个 technical feature 逐一判断是否有问题。
- 每个 feature 都要问：
  - 它真的成立吗
  - 它是否被完整实现
  - 它跨模块、跨文件、跨函数的链路是否闭环
  - 它是否存在状态缺口、逻辑缺口、承诺过度、实现偏差

这里主要抓：

- 全局问题
- 跨模块问题
- 流程问题
- 产品承诺与实现不一致
- 架构性问题

完成并写盘后，才能进入下一步。

### Step 4. 做 Module / File / Function 线审查

继续写入：

- `03-architecture-and-code-quality-review.md`

做法：

- 按 `模块 -> 文件 -> 关键类型 -> 关键函数` 顺序覆盖审查。
- 范围小的时候，按文件顺序逐个审查。
- 核心文件必须逐代码块检查；必要时逐行检查。

每一层至少都要判断：

- 是否有 bug 或逻辑错误
- 是否有边界条件问题
- 是否有重复实现、死代码、遗留分支
- 是否有职责混乱、依赖反向、抽象泄漏
- 是否有命名问题、注释问题、文档缺失
- 是否有 Apple 平台规范、无障碍、权限、隐私、生命周期、响应性问题

这一层主要抓：

- 局部问题
- 低级错误
- 函数级错误
- 文件级设计问题

完成并写盘后，才能进入下一步。

### Step 5. 做 Cross-check 防漏审查

写：

- `05-cross-cutting-review.md`

必须完成这几类交叉检查：

- 从 feature 反查代码实现是否真的闭环
- 从代码实现反查 UI/文案是否有过度承诺
- 从单点问题反查同类实现是否普遍存在
- 检查是否有模块/文件/函数没有进入任何审查结论
- 检查是否有复杂区域被错误地当成“无问题”

这一步用于抓：

- 系统性问题
- 横向重复问题
- 漏审区域
- 前面两条线之间的矛盾

完成并写盘后，才能进入下一步。

### Step 6. 输出报告并二次校验

先写：

- `06-final-report-draft.md`

再写：

- `07-final-report-validation.md`
- `08-final-report.md`

要求：

- 最终报告必须区分严重问题和低级问题。
- 必须区分单点问题和系统性问题。
- 必须保留“已检查但未发现问题”的重点区域，证明覆盖不是假的。
- 如果某些点证据不足，要明确标为“需要运行时验证”，不能假装确认。

## 审查记录要求

在审查文件中，既要记录 finding，也要记录已检查区域。

至少使用这些状态：

- `finding`
- `checked with no finding`
- `needs runtime verification`
- `needs product decision`
- `deferred`

如果只记录 finding，不记录已检查区域，这次 review 视为覆盖不足。

## 必查问题类型

不管从哪条线进入，都必须检查：

- 产品逻辑问题
- 跨模块流程问题
- technical feature 不成立
- 架构设计问题
- 模块边界问题
- 文件/函数级 bug
- 边界条件错误
- 冗余代码和死代码
- 命名问题和 API 设计问题
- 注释缺失、注释误导、TODO/FIXME 不充分
- Apple 平台规范、无障碍、权限、隐私、生命周期、响应性问题
- UI / 文案 / 产品承诺与实现不一致的问题

## 小范围模块的额外要求

如果 review 范围是一个较小模块或单一 feature 目录：

- 必须按文件顺序逐个审查。
- 核心文件优先逐代码块检查。
- 如果目标是“尽量不漏”，核心规则和核心函数要逐行检查。

## finding 标准

只要满足下面任一条件，就应该记录：

- 可能导致错误行为
- 可能导致错误数据
- 可能误导用户
- 可能违反 Apple 平台建议或常见最佳实践
- 可能增加维护成本或扩展成本
- 可能说明当前结构本身有问题
- 即使是小问题，但能反映出更大的代码质量问题

这意味着：

- 不要只盯大 bug。
- 变量名错误、注释错误、文案错误、明显不合理的符号或拼写问题，也可以记录。
- 但最终报告里要区分严重问题和低级问题，避免混在一起。

## 建议产物

默认使用这些文件：

- `00-review-scope-and-assumptions.md`
- `01-product-feature-map.md`
- `02-technical-feature-map.md`
- `03-architecture-and-code-quality-review.md`
- `04-product-and-ux-review.md`
- `05-cross-cutting-review.md`
- `06-final-report-draft.md`
- `07-final-report-validation.md`
- `08-final-report.md`

具体模板见：

- [references/report-structure.md](references/report-structure.md)
- [references/review-criteria.md](references/review-criteria.md)
- [references/deep-review-checklists.md](references/deep-review-checklists.md)
