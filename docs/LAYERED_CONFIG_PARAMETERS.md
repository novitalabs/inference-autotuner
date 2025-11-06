# 分层配置参数体现示例

本文档通过具体示例说明不同配置层如何体现在最终的参数配置中。

## 配置层的应用顺序

```
1. Base Layers (基础层)
   ↓
2. Deployment Mode Layers (部署模式层)
   ↓
3. Runtime Layers (运行时层)
   ↓
4. Profile Layers (场景预设层)
   ↓
5. User Overrides (用户覆盖层)
```

## 实际示例：创建一个低延迟 Docker 任务

### 输入参数

```python
TaskContext(
    model_name="llama-3-2-1b-instruct",
    base_runtime="sglang",
    deployment_mode="docker",
    profiles=["low-latency"],
    optimization_strategy="grid_search",
    optimization_objective="minimize_latency",  # 会被 profile 覆盖
    # ... 其他参数
)
```

### 各层的配置内容

#### 1️⃣ Base Layers (基础层)

**Layer: base-model**
```python
{
    "task_name": "llama-3-2-1b-instruct_grid_search",
    "model": {
        "id_or_path": "llama-3-2-1b-instruct",
        "namespace": "autotuner"
    },
    "base_runtime": "sglang"
}
```

**Layer: base-optimization**
```python
{
    "optimization": {
        "strategy": "grid_search",
        "objective": "minimize_latency",
        "max_iterations": 10,              # 默认值
        "timeout_per_iteration": 600
    },
    "benchmark": {
        "task": "text-to-text",
        "model_name": "llama-3-2-1b-instruct",
        "traffic_scenarios": ["D(200,200)"],
        "num_concurrency": [1, 4, 8]      # 默认值
    }
}
```

**合并结果 (Base):**
```json
{
    "task_name": "llama-3-2-1b-instruct_grid_search",
    "model": {
        "id_or_path": "llama-3-2-1b-instruct",
        "namespace": "autotuner"
    },
    "base_runtime": "sglang",
    "optimization": {
        "strategy": "grid_search",
        "objective": "minimize_latency",
        "max_iterations": 10,
        "timeout_per_iteration": 600
    },
    "benchmark": {
        "task": "text-to-text",
        "model_name": "llama-3-2-1b-instruct",
        "traffic_scenarios": ["D(200,200)"],
        "num_concurrency": [1, 4, 8]
    }
}
```

---

#### 2️⃣ Deployment Mode Layer (部署模式层)

**Layer: docker-defaults** (因为 deployment_mode="docker")
```python
{
    "runtime_image_tag": "v0.5.2-cu126",
    "parameters": {
        "tp-size": [1, 2, 4],              # Docker 默认 tp-size
        "mem-fraction-static": [0.85, 0.9] # Docker 默认内存配置
    }
}
```

**合并结果 (Base + Docker):**
```json
{
    "task_name": "llama-3-2-1b-instruct_grid_search",
    "model": {...},
    "base_runtime": "sglang",
    "runtime_image_tag": "v0.5.2-cu126",    // ← 新增
    "optimization": {...},
    "benchmark": {...},
    "parameters": {                          // ← 新增
        "tp-size": [1, 2, 4],
        "mem-fraction-static": [0.85, 0.9]
    }
}
```

---

#### 3️⃣ Runtime Layer (运行时层)

**Layer: sglang-defaults** (因为 base_runtime="sglang")
```python
{
    "parameters": {
        "schedule-policy": ["lpm"],
        "enable-torch-compile": [True, False]
    }
}
```

**深度合并到 parameters:**
```json
{
    ...,
    "parameters": {
        "tp-size": [1, 2, 4],               // 保持
        "mem-fraction-static": [0.85, 0.9], // 保持
        "schedule-policy": ["lpm"],         // ← 新增 SGLang 参数
        "enable-torch-compile": [True, False] // ← 新增 SGLang 参数
    }
}
```

---

#### 4️⃣ Profile Layer (场景预设层)

**Profile: low-latency** (用户选择的 profile)
```python
LOW_LATENCY_LAYERS = [
    ConfigLayer(
        name="low-latency-params",
        data={
            "optimization": {
                "objective": "minimize_latency",
                "max_iterations": 15           # 覆盖默认的 10
            },
            "parameters": {
                "mem-fraction-static": [0.7, 0.8],  # 覆盖之前的 [0.85, 0.9]
                "tp-size": [1, 2]              # 覆盖之前的 [1, 2, 4]
            },
            "benchmark": {
                "num_concurrency": [1, 4, 8]   # 重新定义并发级别
            }
        }
    )
]
```

**深度合并结果 (覆盖策略):**
```json
{
    "task_name": "llama-3-2-1b-instruct_grid_search",
    "model": {...},
    "base_runtime": "sglang",
    "runtime_image_tag": "v0.5.2-cu126",
    "optimization": {
        "strategy": "grid_search",
        "objective": "minimize_latency",    // 保持
        "max_iterations": 15,               // ✅ 覆盖: 10 → 15
        "timeout_per_iteration": 600
    },
    "benchmark": {
        "task": "text-to-text",
        "model_name": "llama-3-2-1b-instruct",
        "traffic_scenarios": ["D(200,200)"],
        "num_concurrency": [1, 4, 8]        // ✅ 保持 (重新定义相同值)
    },
    "parameters": {
        "tp-size": [1, 2],                  // ✅ 覆盖: [1,2,4] → [1,2]
        "mem-fraction-static": [0.7, 0.8],  // ✅ 覆盖: [0.85,0.9] → [0.7,0.8]
        "schedule-policy": ["lpm"],         // ✅ 保持 SGLang 参数
        "enable-torch-compile": [True, False] // ✅ 保持 SGLang 参数
    }
}
```

---

#### 5️⃣ User Overrides (用户覆盖层)

如果用户在创建任务时提供了自定义覆盖：

```python
user_overrides = {
    "parameters": {
        "mem-fraction-static": [0.75]  # 用户想测试特定值
    },
    "optimization": {
        "max_iterations": 20            # 用户想要更多迭代
    }
}
```

**最终合并结果:**
```json
{
    "task_name": "llama-3-2-1b-instruct_grid_search",
    "model": {...},
    "base_runtime": "sglang",
    "runtime_image_tag": "v0.5.2-cu126",
    "optimization": {
        "strategy": "grid_search",
        "objective": "minimize_latency",
        "max_iterations": 20,               // ✅ 用户覆盖: 15 → 20
        "timeout_per_iteration": 600
    },
    "benchmark": {...},
    "parameters": {
        "tp-size": [1, 2],
        "mem-fraction-static": [0.75],      // ✅ 用户覆盖: [0.7,0.8] → [0.75]
        "schedule-policy": ["lpm"],
        "enable-torch-compile": [True, False]
    }
}
```

---

## 参数覆盖规则总结

### 深度合并 (Deep Merge) 策略

```python
def _deep_merge(target: dict, source: dict, allow_new: bool = True):
    """
    递归深度合并两个字典

    规则:
    - 如果 key 不存在: 添加 (如果 allow_new=True)
    - 如果 key 存在且都是 dict: 递归合并
    - 如果 key 存在但不是 dict: 覆盖 (source 优先)
    """
```

### 实际覆盖效果

| 参数类型 | Base | Docker | Runtime | Profile | User | 最终值 | 说明 |
|---------|------|--------|---------|---------|------|--------|------|
| **max_iterations** | 10 | - | - | 15 | 20 | **20** | 后面的层覆盖前面的 |
| **tp-size** | - | [1,2,4] | - | [1,2] | - | **[1,2]** | Profile 覆盖 Docker |
| **mem-fraction-static** | - | [0.85,0.9] | - | [0.7,0.8] | [0.75] | **[0.75]** | User 最终覆盖 |
| **schedule-policy** | - | - | ["lpm"] | - | - | **["lpm"]** | Runtime 层添加 |
| **enable-torch-compile** | - | - | [T,F] | - | - | **[T,F]** | Runtime 层添加 |
| **runtime_image_tag** | - | v0.5.2 | - | - | - | **v0.5.2** | Docker 层添加 |

---

## 实际应用场景

### 场景 1: 快速测试 (quick-test profile)

```python
profiles=["quick-test"]
```

**最终参数:**
- `max_iterations`: **2** (极少迭代)
- `tp-size`: **[1]** (单卡)
- `mem-fraction-static`: **[0.85]** (单一配置)
- `num_concurrency`: **[1]** (最低并发)

**总实验数:** 1 × 1 × 1 = **1 个实验** (几分钟完成)

---

### 场景 2: 平衡探索 (balanced profile)

```python
profiles=["balanced"]
```

**最终参数:**
- `max_iterations`: **15**
- `tp-size`: **[1, 2, 4]** (多种并行度)
- `mem-fraction-static`: **[0.8, 0.85, 0.9]** (多种内存配置)
- `num_concurrency`: **[4, 8, 16]**

**总实验数:** 3 × 3 × 3 = **27 个实验** (全面探索)

---

### 场景 3: 组合多个 profiles

```python
profiles=["low-latency", "cost-optimization"]
```

**合并逻辑:**
1. 先应用 `low-latency` 的所有层
2. 再应用 `cost-optimization` 的所有层 (覆盖冲突参数)

**最终效果:** 低延迟配置 + 成本约束

---

## 如何在代码中查看配置层

### 1. 查看应用的层列表

```python
from src.config.factory import TaskConfigFactory
from src.config.layers import TaskContext

ctx = TaskContext(
    model_name="llama-3-2-1b-instruct",
    base_runtime="sglang",
    deployment_mode="docker",
    profiles=["low-latency"]
)

config, applied_layers = TaskConfigFactory.create(ctx)

print("Applied layers:", applied_layers)
# 输出:
# [
#   "base-model",
#   "base-optimization",
#   "docker-defaults",
#   "sglang-defaults",
#   "profile:low-latency:low-latency-params"
# ]
```

### 2. 查看最终配置

```python
import json
print(json.dumps(config, indent=2))
```

### 3. 通过 API 查看

```bash
# 创建任务时查看应用的层
curl -X POST http://localhost:8000/api/tasks/from-context \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "llama-3-2-1b-instruct",
    "base_runtime": "sglang",
    "deployment_mode": "docker",
    "profiles": ["low-latency"]
  }' | jq '.applied_layers'

# 输出: ["base-model", "base-optimization", "docker-defaults", "sglang-defaults", "profile:low-latency:low-latency-params"]
```

---

## 调试技巧

### 查看每层的贡献

在 `factory.py` 中启用详细日志:

```python
logger.setLevel(logging.DEBUG)
```

输出示例:
```
DEBUG: Applying layer: base-model
DEBUG:   Added: task_name, model, base_runtime
DEBUG: Applying layer: docker-defaults
DEBUG:   Added: runtime_image_tag, parameters.tp-size, parameters.mem-fraction-static
DEBUG: Applying layer: sglang-defaults
DEBUG:   Merged: parameters.schedule-policy, parameters.enable-torch-compile
DEBUG: Applying layer: profile:low-latency:low-latency-params
DEBUG:   Overrode: optimization.max_iterations (10 → 15)
DEBUG:   Overrode: parameters.tp-size ([1,2,4] → [1,2])
DEBUG:   Overrode: parameters.mem-fraction-static ([0.85,0.9] → [0.7,0.8])
```

---

## 总结

分层配置的核心优势：

1. **模块化**: 每层关注特定配置方面
2. **可复用**: Profile 可在不同任务间共享
3. **优先级清晰**: 后面的层覆盖前面的层
4. **灵活**: 用户可以在任何层级进行自定义覆盖
5. **可追溯**: `applied_layers` 列表记录了所有应用的层

这种设计让配置既有合理的默认值，又允许精细的自定义控制。
