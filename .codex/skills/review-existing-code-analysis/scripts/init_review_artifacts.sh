#!/bin/zsh

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: zsh .codex/skills/review-existing-code-analysis/scripts/init_review_artifacts.sh <scope-slug> [output-root]" >&2
  exit 1
fi

scope_slug="$1"
output_root="${2:-docs/reviews}"
date_stamp="$(date +%F)"
report_dir="${output_root}/${date_stamp}-${scope_slug}"

mkdir -p "$report_dir"

cat > "${report_dir}/00-review-scope-and-assumptions.md" <<EOF
# Review 范围与假设

- 日期: ${date_stamp}
- 范围: ${scope_slug}
- 状态: 进行中

## 本次 review 范围

## 明确不在范围内的部分

## 审查方式

- 静态代码审查:
- 运行时验证:

## 关键依赖与外部集成

## 关键假设

## 需要额外验证的问题
EOF

cat > "${report_dir}/01-product-feature-map.md" <<EOF
# 产品功能设计说明

- 日期: ${date_stamp}
- 范围: ${scope_slug}
- 状态: 进行中

## 文档目标

- 以设计说明文档的方式梳理用户可见能力，而不是简单列功能树。
- 在探索过程中持续更新。
- 对已确认事实与推断项分别标记 \`confirmed\` 和 \`inferred\`。

## 功能设计说明

## 开放问题

## 备注
EOF

cat > "${report_dir}/02-technical-feature-map.md" <<EOF
# 技术设计与实现说明

- 日期: ${date_stamp}
- 范围: ${scope_slug}
- 状态: 进行中

## 文档目标

- 以设计/实现说明文档的方式梳理技术结构，而不是简单列模块树。
- 记录真相源、职责边界、依赖关系、规则策略、集成点和生命周期约束。
- 在探索过程中持续更新。

## 技术设计说明

## 开放问题

## 备注
EOF

cat > "${report_dir}/03-architecture-and-code-quality-review.md" <<EOF
# 架构与代码质量审查

- 日期: ${date_stamp}
- 范围: ${scope_slug}
- 状态: 未开始

## 审查要求

- 使用 \`02-technical-feature-map.md\` 作为 checklist。
- 逐点审查架构、职责、边界、冗余、实现质量、注释文档质量。
- 重要区域即使没有问题，也要记录 \`checked with no finding\`。
- 对每个 finding 记录类型、主题、严重级别、优先级和置信度。

## Findings

## Checked With No Finding

## Needs Runtime Verification

## Needs Product Decision

## Deferred
EOF

cat > "${report_dir}/04-product-and-ux-review.md" <<EOF
# 产品与体验审查

- 日期: ${date_stamp}
- 范围: ${scope_slug}
- 状态: 未开始

## 审查要求

- 使用 \`01-product-feature-map.md\` 作为 checklist。
- 逐点审查产品逻辑、流程闭环、用户心智、平台适配、信任与表达准确性。
- 重要区域即使没有问题，也要记录 \`checked with no finding\`。
- 对每个 finding 记录主题、严重级别、优先级和置信度。

## Findings

## Checked With No Finding

## Needs Product Decision

## Open Questions
EOF

cat > "${report_dir}/05-cross-cutting-review.md" <<EOF
# 交叉维度专项审查

- 日期: ${date_stamp}
- 范围: ${scope_slug}
- 状态: 未开始

## 审查要求

- 汇总跨模块、跨层次、系统性问题。
- 重点覆盖 Apple 平台规范、无障碍、权限与隐私、响应性、注释文档、测试性。

## Findings

## Checked With No Finding

## Needs Runtime Verification

## Open Questions
EOF

cat > "${report_dir}/06-final-report-draft.md" <<EOF
# 最终报告初稿

- 日期: ${date_stamp}
- 范围: ${scope_slug}
- 状态: 初稿

## 执行摘要

## Review 范围与方法

## 产品设计摘要

## 技术设计摘要

## 覆盖摘要

## 优先级概览

## 按优先级分组的 Findings

### P0

### P1

### P2

### P3

## 按主题归纳的问题索引

## 已检查但未发现问题的重点区域

## 系统性问题总结

## 分阶段整改方案

### Phase 0 验证与基线

### Phase 1 用户风险修复

### Phase 2 架构治理与覆盖补强

### Phase 3 体验与打磨

## 开放问题

## 需要运行时验证的点
EOF

cat > "${report_dir}/07-final-report-validation.md" <<EOF
# 最终报告校验

- 日期: ${date_stamp}
- 范围: ${scope_slug}
- 状态: 未开始

## 校验要求

- 针对每个 finding 回查代码和中间产物。
- 去掉夸大表述。
- 降级或删除证据不足的项。
- 如果一个 finding 实际上混合了多个问题，要拆开。
- 复查主题、严重级别、优先级、阶段和置信度是否合理。
- 确保整改阶段顺序符合依赖关系。

## 校验记录
EOF

cat > "${report_dir}/08-final-report.md" <<EOF
# 最终报告

- 日期: ${date_stamp}
- 范围: ${scope_slug}
- 状态: 最终版

## 执行摘要

## Review 范围与方法

## 优先级概览

## 按优先级分组的 Findings

### P0

### P1

### P2

### P3

## 按主题归纳的问题索引

## 架构层总结

## 产品层总结

## 平台规范 / 注释 / 测试性总结

## 已检查但未发现问题的重点区域

## 分阶段整改方案

### Phase 0 验证与基线

### Phase 1 用户风险修复

### Phase 2 架构治理与覆盖补强

### Phase 3 体验与打磨

## 开放问题

## 需要运行时验证的点

## 建议的下一步
EOF

echo "$report_dir"
