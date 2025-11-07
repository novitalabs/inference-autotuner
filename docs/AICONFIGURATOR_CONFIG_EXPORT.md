# 如何获取 Aiconfigurator 配置文件

本文档说明如何从 NVIDIA aiconfigurator 获取配置文件，以便导入到 inference-autotuner 进行实际验证。

## 方法 1: 使用 aiconfigurator CLI 生成配置文件

### 1.1 基本用法（Default Mode）

```bash
aiconfigurator cli default \
  --model QWEN3_32B \
  --total_gpus 8 \
  --system h200_sxm \
  --save_dir ./results
```

**输出文件结构**：
```
results/
├── agg/
│   ├── config.yaml          ← 可以导入到 inference-autotuner
│   ├── best_config_topn.csv
│   └── pareto.csv
└── disagg/
    ├── config.yaml          ← 可以导入到 inference-autotuner
    ├── best_config_topn.csv
    └── pareto.csv
```

### 1.2 高级用法（Exp Mode - 使用 YAML 配置）

```bash
aiconfigurator cli exp \
  --yaml_path my_config.yaml \
  --save_dir ./results
```

**输入 YAML 示例** (`my_config.yaml`):
```yaml
model_name: "LLAMA_70B"
backend_name: "trtllm"
backend_version: "1.0.0rc3"
isl: 4000              # Input sequence length
osl: 1000              # Output sequence length
ttft: 1000.0           # TTFT SLA in ms
tpot: 40.0             # TPOT SLA in ms

agg:
  config:
    worker_config:
      tp: 8
      pp: 1
      gemm_quant_mode: "fp8"
      kvcache_quant_mode: "fp8"
```

输出的 `config.yaml` 文件会包含 aiconfigurator 的静态性能预测结果。

## 方法 2: 使用 aiconfigurator Web UI (Gradio)

### 2.1 启动 Web UI

```bash
cd third_party/aiconfigurator
aiconfigurator webapp
```

浏览器访问：http://localhost:7860

### 2.2 在 Web UI 中操作

1. **配置参数**：
   - 选择 Model（如 QWEN3_32B）
   - 选择 System（如 h200_sxm）
   - 设置 GPU 数量
   - 配置 TP/PP/DP 参数
   - 设置量化选项

2. **运行静态估算**：
   - 点击"Run Static Analysis"按钮
   - 查看 Pareto frontier 可视化结果
   - 显示预测的 throughput、TTFT、TPOT 等指标

3. **导出配置** ⚠️：
   - **当前限制**：Gradio UI 暂时**没有直接的"下载"或"导出"按钮**
   - **解决方案**：需要从命令行的 `--save_dir` 输出获取文件
   - **未来改进**：可以扩展 Gradio UI 添加导出功能

### 2.3 Gradio UI 的局限性

根据 aiconfigurator 文档，Gradio UI 的功能包括：
- ✅ 交互式参数调整
- ✅ 实时 Pareto frontier 可视化
- ✅ 配置比较
- ⚠️ **结果导出功能不完整**

## 方法 3: 使用 aiconfigurator Python SDK

如果你需要自动化生成配置文件，可以使用 Python API：

```python
from aiconfigurator.sdk import task
from aiconfigurator.sdk.pareto_analysis import agg_pareto

# 创建运行时配置
ctx = task.TaskConfigContext(
    model_name="QWEN3_32B",
    total_gpus=8,
    system_name="h200_sxm",
    backend_name="trtllm",
    isl=4000,
    osl=1000,
    ttft=300.0,  # ms
    tpot=15.0,   # ms
)

# 生成配置
config, layers = task.TaskConfigFactory.create(ctx)

# 运行 Pareto 分析
results = agg_pareto(
    model_name="QWEN3_32B",
    runtime_config=config.runtime_config,
    model_config=config.model_config,
    parallel_config_list=[...],
)

# 保存配置到 YAML
import yaml
with open('output_config.yaml', 'w') as f:
    yaml.dump({
        'model_name': config.model_config.model_name,
        'serving_mode': 'agg',
        'worker_config': {
            'tp': config.parallel_config.tp,
            'pp': config.parallel_config.pp,
            'dp': config.parallel_config.dp,
            'gemm_quant_mode': config.model_config.gemm_quant_mode,
            'kvcache_quant_mode': config.model_config.kvcache_quant_mode,
        },
        'predicted_metrics': {
            'throughput': results['throughput'],
            'ttft': results['ttft'],
            'tpot': results['tpot'],
            'latency_p90': results['latency_p90'],
        }
    }, f)
```

## 配置文件格式要求

inference-autotuner 期望的 YAML 格式：

```yaml
model_name: "LLAMA_1B"                    # 模型名称
serving_mode: "agg"                        # agg 或 disagg
worker_config:
  tp: 1                                    # Tensor parallelism
  pp: 1                                    # Pipeline parallelism (可选)
  dp: 1                                    # Data parallelism (可选)
  gemm_quant_mode: "float16"               # GEMM 量化模式
  kvcache_quant_mode: "float16"            # KV Cache 量化模式
predicted_metrics:                         # aiconfigurator 的预测结果
  throughput: 450.0                        # tokens/s/gpu
  ttft: 120.0                              # Time to First Token (ms)
  tpot: 15.0                               # Time Per Output Token (ms)
  latency_p90: 5.2                         # P90 Latency (seconds)
```

## 推荐工作流程

### 当前最佳实践：

1. **使用 CLI 生成配置**：
   ```bash
   aiconfigurator cli default \
     --model llama-3-2-1b-instruct \
     --total_gpus 1 \
     --system h200_sxm \
     --save_dir ./aiconfig_results
   ```

2. **找到生成的配置文件**：
   ```bash
   # Aggregated mode 配置
   ./aiconfig_results/agg/config.yaml

   # 或 Disaggregated mode 配置
   ./aiconfig_results/disagg/config.yaml
   ```

3. **在 inference-autotuner Web UI 中导入**：
   - 打开 http://localhost:3000
   - 点击 "Create New Task"
   - 在 "Import from Aiconfigurator" 区域上传 `config.yaml`
   - 表单会自动填充配置值
   - 检查并调整参数
   - 创建任务并运行

4. **查看验证结果**：
   - 任务完成后，进入 Experiments 页面
   - 点击实验详情
   - 点击 "Verify Predictions" 按钮
   - 查看预测值 vs 实际值的对比

## 常见问题

### Q: Gradio UI 能否导出配置文件？
**A**: 目前 aiconfigurator 的 Gradio UI **没有提供直接的导出/下载功能**。你需要：
- 使用 CLI 的 `--save_dir` 参数保存结果
- 或者使用 Python SDK 编程生成配置

### Q: 如何获取包含预测指标的配置？
**A**: 确保使用 `--save_dir` 参数运行 CLI，生成的 `config.yaml` 会包含 `predicted_metrics` 字段。

### Q: 是否需要手动添加预测指标？
**A**: 不需要。aiconfigurator 的输出文件已经包含性能预测。如果缺少，可能是：
- 使用的 aiconfigurator 版本不支持
- 配置模式不正确
- 需要手动补充测试数据

### Q: 能否改进 Gradio UI 添加导出功能？
**A**: 可以！这是一个潜在的改进方向：
```python
# 在 aiconfigurator/webapp/main.py 中添加
def export_config_button():
    """导出当前配置为 YAML 文件"""
    config_yaml = generate_config_yaml(current_state)
    return gr.File.update(value=config_yaml, visible=True)
```

## 示例配置文件

参考 `examples/verification/example_aiconfig.yaml`:

```yaml
model_name: "LLAMA_1B"
serving_mode: "agg"
worker_config:
  tp: 1
  pp: 1
  dp: 1
  gemm_quant_mode: "float16"
  kvcache_quant_mode: "float16"
predicted_metrics:
  throughput: 450.0  # tokens/s/gpu
  ttft: 120.0        # ms
  tpot: 15.0         # ms
  latency_p90: 5.2   # seconds
```

## 相关文档

- aiconfigurator 项目文档: `docs/aiconfigurator/ANALYSIS.md`
- 验证计划: `docs/AICONFIGURATOR_VERIFICATION_PLAN.md`
- Frontend 集成文档: `docs/FRONTEND_INTEGRATION.md`
