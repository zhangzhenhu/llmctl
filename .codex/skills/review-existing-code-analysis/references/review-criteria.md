# Review Criteria

用这些标准判断一个问题是否值得报告、该如何分类，以及为什么它重要。

## 基线标准

优先基线：

- Apple Human Interface Guidelines
- Apple Accessibility guidance
- Apple Responsiveness guidance
- Swift API Design Guidelines
- Apple App Review Guidelines
- Swift/SwiftUI 常见工程最佳实践

如果某条建议直接来自这些基线，在报告里要点明来源类型，例如“依据 Apple 无障碍基线”或“依据 Swift API 设计习惯”。

## 技术审查维度

### Correctness

检查：

- 分支逻辑是否错误
- 状态转换是否可能非法或失真
- 派生数据是否会偏离真相源
- 日期、时区、locale、精度、边界值是否安全

### Completeness

检查：

- 错误处理是否缺失
- 空态、恢复、迁移、清理逻辑是否缺失
- 是否存在只做了一半的功能
- UI 是否暗示了未真正支持的能力

### Architecture and ownership

检查：

- 职责边界是否清晰
- source of truth 是否唯一
- 模块拆分是否合理
- 依赖方向是否健康
- 规则、策略、阈值是否散落
- 是否存在巨型对象、抽象泄漏、无效抽象

### Redundancy and maintainability

检查：

- 是否有重复逻辑、重复模型转换、重复 UI 拼装
- 是否有近似重复组件却未抽象出清晰边界
- 是否有死代码、休眠开关、遗留配置
- 是否让未来扩展成本异常高

### Concurrency and lifecycle

检查：

- MainActor 使用是否正确
- UI 更新是否可能发生在错误线程
- async 流程是否可能和生命周期打架
- 前后台切换、初始化、销毁、恢复链路是否安全

### Persistence, sync, and data integrity

检查：

- 保存点是否完整
- 写入是否存在部分成功
- 内存态和持久化态是否可能不一致
- widget / 通知 / 导出 / 分享等跨表面数据是否可能陈旧
- 是否缺少迁移和兼容处理

### Platform quality

检查：

- Apple 框架使用是否合规
- 无障碍支持是否缺失
- 是否存在主线程阻塞、卡顿、长任务无反馈
- 权限、隐私、敏感数据处理是否合理
- 是否存在 App Review 风险

### Comments and documentation

检查：

- 核心类型和复杂逻辑是否缺少必要注释
- 是否缺少设计边界、调用顺序、约束说明
- TODO / FIXME 是否足够具体
- 注释是否过时、误导、与真实代码脱节
- 是否存在关键上下文只藏在实现细节里，没有任何文档痕迹

### Testability

检查：

- 关键逻辑是否缺少可验证路径
- 是否存在无法隔离验证的硬编码策略
- 回归风险高的规则是否没有测试覆盖或可测试 seam

## 产品审查维度

### Feature logic

检查：

- 功能是否真的解决其宣称要解决的问题
- 流程是否有开始、延续、完成、失败、恢复
- 产品承诺和真实能力是否一致

### Usability

检查：

- 关键操作是否容易发现
- 类似场景是否一致
- 空态、错误态、权限态是否完整
- 编辑、撤销、恢复、补录是否合理

### Trust and safety

检查：

- 敏感数据处理是否符合用户预期
- 对不确定性、预测性结果是否表达诚实
- 是否会让用户把建议、预测、提示误认为事实

### Platform fit

检查：

- 是否符合 iOS 常见交互预期
- 是否符合可读性、可访问性、原生感
- 是否无充分理由地偏离 Apple 推荐体验

## Finding schema

有价值的 finding 至少包含：

- `ID`
- `Area`
- `Type`
- `Theme`
- `Severity`
- `Priority`
- `Confidence`
- `Evidence`
- `Impact`
- `Root cause`
- `Recommendation`

推荐 `Type`：

- `bug`
- `logic gap`
- `architecture issue`
- `redundancy`
- `comment gap`
- `incomplete`
- `best-practice gap`
- `ux gap`
- `risk`

推荐 `Theme`：

- `correctness`
- `data integrity`
- `concurrency`
- `architecture`
- `platform quality`
- `platform fit`
- `testability`
- `comments`
- `ux`
- `accessibility`
- `discoverability`
- `trust`

## Severity guidance

- `critical`: 崩溃、严重数据错误、严重隐私/合规问题、核心功能失效
- `high`: 关键流程或关键判断明显错误，或存在强烈发布风险
- `medium`: 明确值得修复的质量问题，但不会立即造成灾难性后果
- `low`: 有清晰收益的改进项，但风险较低

## Priority guidance

- `P0`: 必须最先处理；会阻塞发布、阻塞可信验证、或风险过高
- `P1`: 下一阶段优先处理；显著降低用户风险或稳定关键能力
- `P2`: 计划性治理；主要改善架构、维护性、一致性、覆盖
- `P3`: 打磨和低风险优化

## Confidence guidance

- `confirmed`: 有直接代码、行为或文档证据
- `likely`: 证据较强，但还包含小部分假设
- `needs runtime verification`: 静态证据不够，需要运行验证

## 反漏报规则

发现一个问题后，不要立刻停在单点，继续做两件事：

- 横向搜索同类实现，判断这是单点 bug 还是系统性模式
- 反向检查上层设计，判断这是不是架构/职责问题的表象

## 反吹毛求疵规则

不要因为“不是我最喜欢的写法”就报告问题。

只有影响以下至少一项，才值得进入正式报告：

- 用户价值
- 正确性
- 可靠性
- 可维护性
- Apple 平台一致性
- 隐私/信任/合规
- 后续扩展成本
