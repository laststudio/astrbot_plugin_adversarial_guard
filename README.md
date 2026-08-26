# astrbot_plugin_adversarial_guard

GAN 式双 AI 对抗人格守护插件。针对 AstrBot 主 AI（B）"不听话 / 不尊崇指令"的问题——例如人格提示词明确禁止 markdown 却仍然输出 `**加粗**`——引入第二个判官 AI（A），逐轮审查 B 的输出是否遵守人格提示词，违规即修复并施以"奖惩"，模拟强化学习让 B 越用越听话。

> **v1.1.0 新增：三模型长期记忆**。引入第三个模型 C（记忆管理员），在每轮审查/修复完成后于**后台**把违规教训浓缩为一条可执行教训；再由判官 A 从**忠实性 / 简洁性 / 隐私过滤**三个维度复审，通过后写入持久 `memory.json`。主 AI(B) 每轮请求前读取该文件，以 `<guard_memory>` 块注入长期教训——跨会话轮次持续纠偏，不再只依赖短期反馈窗口。

## 架构

```
用户消息 → [B: 主AI生成回复] → [A: 判官AI审查] → 通过 → 正常发送
                                      ↓ 违规
                          奖惩记账(severity扣分) + 对抗修复循环
                            ├─ ① fast_fix：直接采用判官的改写（复检通过即用）
                            ├─ ② 降级 regen：带违规清单驳回，令B重新生成(≤N轮)
                            └─ ③ 全部失败：择优放行 + 日志告警
                          ↓（v1.1.0，后台异步，不阻塞回复发送）
                 [C: 记忆管理员] 提炼教训草稿 {"rule", "lesson", "keep"}
                                      ↓
                 [A: 判官复审] 忠实性 / 简洁性 / 隐私过滤（三维）
                       ├─ 通过 → 写入持久 memory.json
                       ├─ 轻微问题 → 采纳 A 的修正版写入
                       └─ 捏造事实 → 丢弃
下一轮请求时：
   注入 <guard_memory> 长期教训（读持久文件，同规则合并演进）
   注入 <guard_feedback> 惩罚记忆（临时内容块，不写入历史、不破坏缓存）
```

- **判官 A** 收到 `[人格提示词]+[近几轮对话]+[本轮用户消息]+[待审回复]`，返回严格 JSON 裁决：`{compliant, score, violations[{rule, evidence, severity, suggestion}], rewrite}`。
- **记忆 C**（v1.1.0）收到本轮裁决与修复结果 JSON，输出 `{rule, lesson, keep}`；`rule` 概括人格规则本身（非违规现象），`lesson` 为一句可执行的规避指令。
- **A 复审**（v1.1.0）：忠实性（不得捏造裁决外的问题）/ 简洁性（一句可执行）/ 隐私（不得含 QQ 号、昵称等敏感信息）；轻微问题返回 `fixed_lesson` 修正版采纳写入，捏造事实直接丢弃。
- **RL 模拟**：插件层无法真正微调模型，因此用"纪律分（带衰减）+ 近 N 轮违规清单注入后续请求 + 长期教训记忆"的方式模拟策略梯度反馈；违规会按 severity 扣分，修复成功小幅加分。

## 安装

将本目录放入 `AstrBot/data/plugins/` 后在 WebUI 重载插件即可。无额外第三方依赖。

## 配置（WebUI 插件配置）

| 配置项 | 默认 | 说明 |
|---|---|---|
| enable | true | 总开关 |
| violation_action | fast_fix_then_regen | fast_fix_then_regen / regen_only / record_only |
| max_regen_rounds | 2 | 对抗循环最大重生成轮数 |
| score_threshold | 70 | 合规分数阈值（0-100） |
| judge_provider_id | 空 | 判官独立提供商 ID，留空跟随主 AI |
| judge_model | 空 | 判官强制模型名，留空用提供商默认；重生成始终用主模型 |
| feedback_injection | true | 是否向后续请求注入惩罚记忆 |
| feedback_window | 5 | 反馈统计窗口（最近 N 次审查） |
| reward_decay | 0.8 | 纪律分衰减系数 |
| judge_timeout | 30 | 单次判官/重生成调用超时秒数 |
| max_judge_chars | 4000 | 送审文本截断长度 |
| history_turns_for_judge | 6 | 提供给判官的上下文条数上限 |
| notify_on_fix | false | 修复后在聊天中发送提示 |

> 成本提示：默认全量判定，每条 LLM 回复会增加一次判官调用。若使用主 AI 同款大模型，token 开销约翻倍；建议为判官配置便宜的小模型（judge_provider_id / judge_model）。
>
> 延迟提示：违规时修复流程（复检 + 最多 N 轮重生成）在回复送达前内联执行；按 judge_timeout=30s 的最坏情况，一条消息可能被阻塞数分钟。对延迟敏感的场景可调低 judge_timeout 或改用 record_only 策略。
>
> v1.1.0 成本与延迟：C 总结 + A 复审仅在**违规轮**触发，且**后台异步**执行（回复先送达用户，零额外延迟）；每条违规回复额外增加 2 次调用（C + A 复审），建议为 C 配置便宜小模型（summarizer_provider_id / summarizer_model）。任一环节失败即静默丢弃本轮记忆（fail-open），不影响聊天。

## 指令

| 指令 | 说明 |
|---|---|
| `/guard` | 查看状态与帮助 |
| `/guard on` / `/guard off` | 本会话开启/关闭审查 |
| `/guard stats` | 纪律分、审查/违规/修复计数、高频违规 Top5 |
| `/guard memory` | 查看本会话长期记忆（C 总结 + A 复审的教训条目） |
| `/guard forget <序号\|all>` | 删除长期记忆条目（管理员） |
| `/guard test <文本>` | 以本会话最近的人格提示词为基准手动审查任意文本 |
| `/guard reset` | 清空本会话账本（管理员） |

## 可靠性设计

- 判官调用走 `provider.text_chat()` 直连，不经过主管线，**无递归风险**；
- 判官超时 / 无可用提供商 / JSON 解析失败 → 一律放行原文，只记日志，绝不阻塞正常聊天；
- 修改回复仅替换消息链中的纯文本组件，图片等组件保留；
- 快照按事件隔离、TTL+容量双保险清理，并发多会话账本按键隔离；
- 长期记忆（v1.1.0）：同规则教训**合并演进**（次数+1、lesson 更新）而非无限堆积；条目按会话上限裁剪；C/A 任一失败静默丢弃本轮；后台任务在插件卸载时等待收尾（≤10s）；
- 数据持久化于 `data/plugin_data/astrbot_plugin_adversarial_guard/`（`ledger.json` 奖惩账本 + `memory.json` 长期记忆），均为原子写。

## 局限

- "强化学习"为上下文层面的模拟（奖惩记账+反馈注入+长期记忆），不改变模型权重；效果依赖所接入模型对注入指令的遵从度。
- 判官由 LLM 充当，存在误判可能；可通过 score_threshold、violation_action=record_only 等配置调节激进程度。
- 长期记忆由 C 提炼、A 复审，仍可能存在个别偏差；可用 `/guard memory` 查看、`/guard forget` 删除不当条目。

## 更新记录

### v1.1.0
- 新增**第三模型 C（记忆管理员）**：违规轮后台异步提炼教训草稿（`{rule, lesson, keep}`）
- 新增**判官 A 三维复审**：忠实性（不得捏造）/ 简洁性（一句可执行）/ 隐私过滤（脱敏 QQ 号等），轻微问题采纳修正版，捏造直接丢弃
- 新增**持久长期记忆** `memory.json`：按会话隔离，同规则合并演进，容量上限可配
- 主 AI 每轮请求前注入 `<guard_memory>` 长期教训块（与 `<guard_feedback>` 窗口反馈互补）
- 新增指令 `/guard memory`、`/guard forget <序号|all>`
- 新增配置：memory_enabled / memory_injection / summarizer_provider_id / summarizer_model / memory_max_entries / summary_review_enabled
