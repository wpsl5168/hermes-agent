# Anthropic/Claude Memory 机制深度调研报告

## 调研来源
1. support.claude.com — memory 相关帮助文章（主文章+导入导出文章）
2. docs.anthropic.com — Memory Tool API 文档（memory_20250818）
3. docs.anthropic.com — Context Windows / Compaction 文档
4. anthropic.com/news — 无专门 memory 博文（搜索确认）
5. Claude Code 中的 CLAUDE.md memory 系统
6. MCP 规范 — 无专门 memory specification

---

## 一、Claude Memory 的分层架构

Claude 的 memory 实际上至少分 **4 层**，分布在不同产品形态中：

### 第1层：对话内上下文（Context Window）
- 单次对话中的消息历史，即模型的"工作记忆"
- 最大可达 1M tokens（取决于模型）
- 随对话进行线性增长
- 配套机制：**Compaction**（服务端自动摘要压缩旧上下文）、**Context Editing**（客户端清理 tool results/thinking blocks）

### 第2层：跨对话搜索（Chat Search / RAG）
- 付费用户（Pro/Max/Team/Enterprise）可用
- Claude 通过 RAG（检索增强生成）搜索用户历史对话
- 搜索范围：项目外所有聊天 + 单个项目内的对话
- 在对话中以 tool call 形式呈现
- 设置路径：Settings > Capabilities > "Search and reference chats"

### 第3层：持久记忆摘要（Memory Summary / Persistent Memory）
- **所有用户可用**（Free/Pro/Max/Team/Enterprise）
- Claude 自动从聊天历史中生成记忆
- **每24小时**自动更新一次综合摘要
- 分为两个独立空间：
  - **个人记忆摘要**：跨所有非项目对话的综合记忆
  - **项目记忆摘要**：每个项目有独立的记忆空间和专属摘要
- 记忆内容聚焦于工作相关上下文

### 第4层：API Memory Tool（开发者用）
- 类型标识：memory_20250818（GA 状态）
- **客户端工具**（client-side tool）——存储完全由开发者控制
- 通过文件目录 /memories 存储和检索信息
- 支持跨会话、跨 context window 的持久化

### 额外层：Claude Code 的 CLAUDE.md
- 用户级记忆：~/.claude/CLAUDE.md
- 项目级记忆：./CLAUDE.md
- /memory 命令配置内置记忆系统
- 支持 auto-memory（自动保存偏好/纠正/模式）
- 支持 auto-dream（定期整理记忆，类比 REM 睡眠整合记忆）

---

## 二、存储格式

### Claude.ai 产品端（第2-3层）
- **纯文本摘要格式**：记忆以自然语言文本摘要形式存储
- 用户可在 Settings > Capabilities > "View and edit memory" 中直接查看
- 搜索历史对话使用 **RAG（检索增强生成）** 技术，底层可能有向量索引
- 记忆摘要随对话数据一起加密存储（encryption at rest）

### API Memory Tool（第4层）
- **纯文本文件格式**：以文件目录结构存储在 /memories 下
- 文档示例使用 XML 格式文件（如 customer_service_guidelines.xml）
- 完全由开发者决定后端：文件系统、数据库、云存储、加密文件等均可
- 不是向量数据库，是结构化的文本文件

### Claude Code（CLAUDE.md）
- **Markdown 纯文本文件**
- 存储在文件系统中，可纳入 git 版本控制

---

## 三、用户能否导出/迁移 Memory？

### 可以！
- **导出**：Settings > Capabilities > "View and edit your memory" 可查看完整记忆；也可在对话中要求 Claude "Write out your memories of me verbatim"
- **导入**：支持从其他 AI 服务导入记忆
  - Settings > Capabilities > Memory 部分 > "Start import"
  - 或主页的 "Import memory to Claude" 卡片
  - 提供了标准化的导出提示词模板
  - 导入后 24 小时内更新记忆
- **适用范围**：Free/Pro/Max 用户可用（web 和 Desktop）
- **注意**：导入功能仍为实验性质
- **企业数据导出**：Memory 数据包含在标准数据导出中

---

## 四、API 层面有无 Memory 接口？

### 有！Memory Tool API
- API 类型：memory_20250818（GA 状态）
- 客户端工具，需要开发者自行实现存储后端
- SDK 提供辅助类：Python 的 BetaAbstractMemoryTool、TypeScript 的 betaMemoryTool
- 支持 Zero Data Retention（ZDR）

### 注意
Claude.ai 产品中的持久记忆（Memory Summary）并没有公开 API 接口。Memory Tool 是给开发者构建自己的 memory 系统用的。

---

## 五、Memory 的增删改查机制

### Claude.ai 产品端
- 查看：Settings > Capabilities > "View and edit memory"
- 新增：直接在对话中告诉 Claude 要记住什么；自动每24小时综合
- 修改：在记忆界面点铅笔图标编辑；或在对话中要求修改
- 删除：删除对话会从记忆综合中移除；可"Reset memory"永久删除所有记忆
- 暂停："Pause memory"保留现有记忆但不使用也不新增

### API Memory Tool（6种命令）
- view：查看目录内容或文件内容（支持行范围）
- create：创建新文件
- str_replace：替换文件中的文本
- insert：在指定行插入文本
- delete：删除文件或目录（递归）
- rename：重命名/移动文件或目录

---

## 六、Anthropic 对 Memory 的设计哲学

1. **用户控制优先**：所有 memory 功能可开关，Incognito 模式排除特定对话
2. **透明可见**：用户可随时查看/编辑 Claude 的记忆内容
3. **工作聚焦**：聚焦角色/项目/技术偏好/编码风格，不主动记忆无关个人信息
4. **客户端架构**：API Memory Tool 刻意设计为客户端工具，开发者完全掌控存储
5. **Just-in-Time Context**：按需检索而非预加载，对抗 context rot
6. **可组合分层**：Memory + Compaction + Context Editing 三者可组合使用
7. **数据可迁移**：支持导入/导出，提供标准化迁移流程
8. **隐私合规**：遵循数据保留策略，删除对话时摘要也被删除

---

## 七、MCP 中的 Memory

MCP 规范本身没有专门的 memory specification。MCP 定义通用原语（Resources/Prompts/Tools），memory 可通过这些原语实现，但不是规范层面的内容。

---

## 八、时间线
- 2025年9月18日：Enterprise 计划上线 memory
- 2025年10月23日：Pro 和 Max 计划上线 memory
- Memory Tool API（memory_20250818）：已 GA
