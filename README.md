# 集成攻防专家系统

本仓库实现了一个用于空战场景的集成专家系统，结合攻击与防御策略，实现对多架友机的协同决策。核心模块现统一收敛到 `airbattle` 包，既包含原有的动态博弈与强化学习实现，也封装了攻击/防御专家系统，便于在单一命名空间下协同验证不同策略组合。

## 功能特性

- **集成式决策框架**：`airbattle/expert/expertsystem.py` 提供统一入口，根据威胁阈值在攻击与防御系统之间切换，并汇总各机决策、解释与历史记录。
- **攻击专家系统**：支持优化模式与行为树模式，可利用帕累托前沿解或态势行为树生成动作，同时给出决策解释。
- **防御专家系统**：提供差分进化 (DE) 优化与行为树两种模式，包含单位制转换、威胁跟踪及导弹规避策略。
- **行为树引擎**：可根据态势阈值动态构建行为树，为每架友机生成打击或支援动作，并输出可读性解释。
- **工具与威胁评估**：附带经纬度与NED坐标互转、航向角转换、归一化等实用函数，以及多模型威胁评估接口。

## 目录结构

```
airbattle/
├── __init__.py                       # 统一对外导出 DroneCombatSystem、AircraftManeuvering 等核心类
├── DroneCombatSystem.py              # 动态贝叶斯策略主控模块
├── AircraftManeuvering.py            # 机动执行模块
├── Hierarchical.py                   # 分层强化学习与专家融合实现
├── BattlefieldEnvironment/           # 环境建模与 GNN 信息融合
├── StrategyPredictor/                # 策略推理器
├── ThreatAnalyzer/                   # 威胁评估模块
├── demo.py / main.py / test.py       # 示例脚本
└── expert/                           # 攻防专家系统子包
    ├── expertsystem.py               # 集成攻防协同系统
    ├── attack_expertsystem.py        # 攻击专家系统
    ├── defend_expertsystem.py        # 防御专家系统
    ├── behavior_tree/                # 行为树引擎与节点
    ├── optimize/                     # 差分进化、NSGA 等优化模块
    ├── evalthreat/                   # 威胁评估模型
    ├── generators.py                 # 行为树节点辅助生成器
    ├── utils.py                      # 坐标/航向转换等工具方法
    └── demo_expertsystem.py          # 集成系统演示脚本
```

## 环境准备

- Python 3.9 及以上版本
- 建议使用虚拟环境（`python -m venv .venv && source .venv/bin/activate`）
- 依赖库：
  - `numpy`
  - `PyYAML`
  - `torch`

> 根据实际接入的威胁模型或优化模块，可能还需要额外依赖（如项目外部的 `common.registry` 注册机制）。

安装依赖例：

```bash
pip install numpy PyYAML torch
```

## 快速开始

1. 克隆仓库并进入项目根目录：

   ```bash
   git clone <repo-url>
   cd <repo-name>/guozhong
   ```

2. 运行演示脚本体验攻防协同流程：

   ```bash
   python -m airbattle.expert.demo_expertsystem
   ```

   - 若存在 `configs/expert.yaml` 与 `configs/defend_expert.yaml`，脚本会使用配置构建完整系统。
   - 当配置缺失或构建失败时，将自动回退至内置参数的简化系统，仍可展示基本流程。

3. 根据自身环境，将实时态势数据转换为系统所需的结构体，并调用 `airbattle.expert.ExpertSystem.process(state)` 获取各机动作及解释。

## 强化学习与元学习文档

- `docs/project_compendium_cn.md`：整合系统架构、分层强化学习、元学习迁移方案与自适应博弈分析。
- `reinforcement_learning_overview.md`：详细拆解分层强化学习策略、专家融合逻辑及奖励反馈。
- `meta_transfer_blueprint.md`：规划元学习任务缓冲、条件化策略与快速再训练机制。
- `adaptive_game_assessment.md`：评估现有自适应博弈能力并给出补强建议。
- `integration_plan.md`：描述专家系统与动态贝叶斯链路的对接策略。

## 关键模块说明

### airbattle/expert/expertsystem.py

- 负责组合攻击与防御系统，依据导弹锁定状态和威胁阈值选择策略。
- 提供 `set_attack_system`、`set_defend_system`、`set_thresholds` 等接口，便于动态调整模式或替换子系统。
- `process(state)` 方法输出所有友机动作、解释及决策摘要。

### airbattle/expert/attack_expertsystem.py

- `ExpertSystem` 支持 `optimization` 与 `behavior_tree` 两种运行模式。
- 在优化模式下，利用优化模块返回帕累托解集，并根据策略选择最终动作。
- 在行为树模式中，借助 `BehaviorTreeEngine` 输出包含攻击/支援标志的动作及解释。

### airbattle/expert/defend_expertsystem.py

- `DefendExpertSystem` 提供差分进化优化 (`mode="de"`) 与行为树 (`mode="bt"`) 两种规避策略。
- 内置多种单位转换、威胁度历史记录与导弹轨迹追踪工具，确保与底层模型接口一致。

### airbattle/expert/behavior_tree/

- `BehaviorTreeEngine` 根据当前态势构建行为树，并对每架友机分配打击或支援动作。
- `nodes.py` 定义条件节点、动作节点及公共上下文，支持共享目标锁定、解释生成等功能。

### airbattle/expert/utils.py

- 提供经纬度 <-> NED 坐标转换、航向角转换及归一化方法，适用于将外部数据映射到系统输入/输出格式。

## 运行测试

当前仓库未包含自动化测试，可通过运行演示脚本验证主要流程：

```bash
python -m airbattle.expert.demo_expertsystem
```

如需集成至更大系统，请结合自身环境补充单元测试或集成测试用例。

## 项目打包

- 运行 `python package_project.py` 将在 `dist/guozhong_project.zip` 中生成包含代码与文档的压缩包。
- 可通过 `python package_project.py --output <自定义路径>` 指定输出位置。

root@9cb2db2c4b52:/workspace/guozhong#
