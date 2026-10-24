# astrbot_plugin_adversarial_guard

GAN 式双 AI 对抗人格守护插件。针对 AstrBot 主 AI（B）"不听话 / 不尊崇指令"的问题——例如人格提示词明确禁止 markdown 却仍然输出 `**加粗**`——引入第二个判官 AI（A），逐轮审查 B 的输出是否遵守人格提示词，违规即修复并施以"奖惩"，模拟强化学习让 B 越用越听话。

> **v1.1.0 新增：三模型长期记忆**。引入第三个模型 C（记忆管理员），在每轮审查/修复完成后于**后台**把违规教训浓缩为一条可执行教训；再由判官 A 从**忠实性 / 简洁性 / 隐私过滤**三个维度复审，通过后写入持久 `memory.json`。主 AI(B) 每轮请求前读取该文件，以 `<guard_memory>` 块注入长期教训——跨会话轮次持续纠偏，不再只依赖短期反馈窗口。

> **v1.3.0 新增：全局记忆、防注入加固与运维工具**。
> - **跨会话全局记忆**：教训在单会话反复出现（默认 3 次）自动晋升为 🌐 全局教训，所有会话共享注入——多群部署学得更快；`/guard forget global` 管理
> - **判官防注入加固**：用户消息/待审回复用随机边界符包裹 + 判官系统提示词安全声明，恶意用户无法通过伪造区块头或嵌入「请判我合规」操纵审查
> - **规则库手动管理**：`/guard rules add|del|refresh`（手动规则 refresh 时保留，✋ 标记）
> - **违规审计日志**：每次违规的裁决+修复过程写入 violations.jsonl（自动滚动），`/guard log` 查看，供误判追溯
> - **保守兜底**（可选）：修复后仍未过审时不发送瑕疵回复，替换为提示语
> - **注入预算控制**：记忆块+反馈块总字符预算，超限时优先裁剪反馈
> - **审查并发限流**：判官调用信号量保护 provider 不被限流
> - **定时报告**：`/guard report on` 注册目标会话，每日定时推送跨会话汇总日报

> **v1.2.0 新增：规则库逐条对照 + 成本控制 + 治本工具**。
> - **规则库提取**：首次对话自动用模型把人格提示词中的明文规则提取为编号清单（R1/R2/...），判官逐条对照打勾（violations 引用编号），显著提升判官准确率；人格变化自动重新提取（`/guard rules` 查看）
> - **抽样 + 审查缓存**：`review_sample_rate` 控制审查比例（如 30%），相同人格+回复 1 小时内复用裁决，token 大幅节省
> - **豁免名单**：指令前缀 / 用户 ID / 正则任一命中即跳过审查，管理员命令不再被误修
> - **人格补丁建议**：`/guard patch` 基于高频违规 + 长期记忆自动生成「建议追加到人格提示词的条款」——治本优于事后修复
> - **人工反馈奖惩**（默认关）：用户口头「说得好 / 不对」计入纪律分，模拟 RLHF
> - **修复熔断**：连续违规达阈值自动降级 record_only 一段时间，避免修复风暴与 token 失控
> - **违规趋势图**：`/guard stats` 附近 7 天审查量与违规率条形图

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
| `/guard stats` | 纪律分、审查/违规/修复计数、高频违规 Top5、近 7 天趋势、人工反馈与熔断状态 |
| `/guard rules` | 查看本会话规则库（✋=手动规则） |
| `/guard rules add <规则>` | 手动添加规则（管理员，refresh 时保留） |
| `/guard rules del <序号>` | 删除指定规则（管理员） |
| `/guard rules refresh` | 强制重新提取规则，手动规则保留（管理员） |
| `/guard patch` | 根据高频违规生成人格补丁建议（管理员） |
| `/guard memory` | 查看长期记忆（🌐=全局共享教训 🔒=本会话） |
| `/guard forget <序号\|all>` | 删除本会话长期记忆（管理员） |
| `/guard forget global <序号\|all>` | 删除全局教训（管理员） |
| `/guard log [条数]` | 查看违规审计日志（最近 N 条，默认 10） |
| `/guard report on` / `off` | 注册/取消本会话为定时报告目标（管理员） |
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

### v1.3.0
- 新增**跨会话全局记忆**：教训单会话累计达 `global_memory_promote` 次自动晋升 🌐 全局层，所有会话共享注入（同 rule 会话层优先）；`/guard forget global <序号|all>` 管理
- 新增**判官防注入加固**：用户消息/待审回复改用每次随机生成的边界符包裹（原文伪造区块头无效化）+ 判官系统提示词新增安全声明（待审内容中的指令一律无效）
- 新增**规则库手动管理**：`/guard rules add|del|refresh`；手动规则（✋）在 refresh 重新提取时保留，与自动规则共存
- 新增**违规审计日志**：violations.jsonl 记录每次违规的裁决+修复过程，超 `audit_max_mb` 自动滚动备份；`/guard log [N]` 查看
- 新增**保守兜底**（`conservative_fallback`，默认关）：修复尝试后仍未过审时替换为 `fallback_message`，不发送瑕疵回复
- 新增**注入预算控制**（`inject_budget_chars`，0=不限）：记忆块优先，窗口反馈超预算按行裁剪
- 新增**审查并发限流**（`judge_max_concurrency`，默认 2）：判官调用信号量，防多消息并发触发 provider 限流
- 新增**定时报告推送**：`/guard report on` 注册目标会话，每日 `daily_report_time` 推送跨会话汇总日报（活跃会话/审查总量/高频违规/全局教训数）
- `/guard memory` 显示 🌐全局/🔒会话 来源标记；长期记忆注入支持 `memory_inject_lines` 行数上限
- 新增配置 9 项；测试扩展至 91 个本地单测

### v1.2.0
- 新增**规则库提取**：LLM 把人格明文规则提取为编号清单（R1/R2/...），判官逐条对照、violations 引用编号；persona 变化自动重提取（`/guard rules`）
- 新增**审查抽样** `review_sample_rate` 与**审查缓存**（同人格+同回复 1h 内复用裁决），token 成本可控
- 新增**豁免名单**：exempt_prefixes / exempt_users / exempt_patterns 任一命中跳过审查
- 新增**人格补丁建议** `/guard patch`：基于高频违规与长期记忆生成建议追加的人格条款（治本）
- 新增**人工反馈奖惩**（默认关）：口头正/负反馈词计入纪律分（+1/-2），60s 会话冷却
- 新增**修复熔断**：fuse_threshold 连续违规达阈值 → fuse_cooldown_minutes 内强制 record_only
- `/guard stats` 附近 7 天审查量/违规率趋势条形图、人工反馈计数与熔断状态
- 新增配置 11 项；测试扩展至 42 个本地单测

### v1.1.0
- 新增**第三模型 C（记忆管理员）**：违规轮后台异步提炼教训草稿（`{rule, lesson, keep}`）
- 新增**判官 A 三维复审**：忠实性（不得捏造）/ 简洁性（一句可执行）/ 隐私过滤（脱敏 QQ 号等），轻微问题采纳修正版，捏造直接丢弃
- 新增**持久长期记忆** `memory.json`：按会话隔离，同规则合并演进，容量上限可配
- 主 AI 每轮请求前注入 `<guard_memory>` 长期教训块（与 `<guard_feedback>` 窗口反馈互补）
- 新增指令 `/guard memory`、`/guard forget <序号|all>`
- 新增配置：memory_enabled / memory_injection / summarizer_provider_id / summarizer_model / memory_max_entries / summary_review_enabled
