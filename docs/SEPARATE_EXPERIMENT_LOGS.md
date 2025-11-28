# Separate Experiment Logs

## 问题背景

在之前的实现中，所有实验共享同一个任务日志文件 (`task_{id}.log`)。这导致了严重的日志混乱问题：

### 问题表现
当实验超时时：
- ARQ Worker抛出 `asyncio.TimeoutError` 并开始下一个实验
- 但超时实验的 `subprocess.run()` 继续在后台运行（僵尸进程）
- 当僵尸进程完成时，其输出被写入当前实验的日志
- 结果：实验N的日志出现在实验N+1的日志文件中

### 技术原因
1. `asyncio.wait_for()` 不会终止 `ThreadPoolExecutor` 中的线程
2. `subprocess.run()` 的超时可能比ARQ Worker超时更长
3. Python的 `print()` 写入共享的日志文件
4. 日志重定向 (`sys.stdout = StreamToLogger()`) 是全局的

## 解决方案

### 实现方式：每个实验使用独立日志文件

```
logs/
├── task_9.log                    # 任务级别的汇总日志
├── task_9_exp_1.log              # 实验1的专属日志
├── task_9_exp_2.log              # 实验2的专属日志
├── task_9_exp_3.log              # 实验3的专属日志
└── task_9_exp_4.log              # 实验4的专属日志
```

### 核心变更

#### 1. 新增实验日志设置函数 (`src/web/workers/autotuner_worker.py`)

```python
def setup_experiment_logging(task_id: int, experiment_id: int):
    """Setup logging for a specific experiment.

    创建实验专属日志文件，同时保持任务级别日志的汇总。
    """
    log_dir = Path.home() / ".local/share/inference-autotuner/logs"
    experiment_log_file = log_dir / f"task_{task_id}_exp_{experiment_id}.log"
    task_log_file = log_dir / f"task_{task_id}.log"

    logger = logging.getLogger(f"task_{task_id}_exp_{experiment_id}")

    # 写入实验专属日志（覆盖模式）
    exp_file_handler = logging.FileHandler(experiment_log_file, mode="w")

    # 同时写入任务汇总日志（追加模式）
    task_file_handler = logging.FileHandler(task_log_file, mode="a")

    # ... 配置handlers和formatter

    # 重定向stdout/stderr到新logger
    sys.stdout = StreamToLogger(logger, logging.INFO)
    sys.stderr = StreamToLogger(logger, logging.ERROR)

    return logger
```

#### 2. 在实验开始时切换日志

```python
# 每个实验开始前切换到专属日志
logger = setup_experiment_logging(task_id, iteration)
logger.info(f"[Experiment {iteration}] Logging to experiment-specific file")
```

#### 3. 新增实验日志API (`src/web/routes/tasks.py`)

```python
@router.get("/{task_id}/experiments/{experiment_id}/logs")
async def get_experiment_logs(task_id: int, experiment_id: int, follow: bool = False):
    """获取实验专属日志"""
    log_file = get_experiment_log_file(task_id, experiment_id)
    # ... 返回日志内容或streaming response
```

## 优势

### 1. 隔离僵尸进程输出
- 每个实验的僵尸进程只会污染自己的日志文件
- 不会影响后续实验的日志

### 2. 清晰的日志结构
- 任务级别日志：完整的汇总视图
- 实验级别日志：每个实验的详细输出
- 便于调试和追踪特定实验

### 3. 并行实验支持
- 如果将来支持并行运行实验，每个实验有独立日志文件
- 避免日志交织混乱

### 4. 精确的日志清理
- 可以单独清理特定实验的日志
- 保留任务汇总日志

## 使用方式

### API访问

```bash
# 获取任务级别的汇总日志
curl http://localhost:8000/api/tasks/9/logs

# 获取特定实验的日志
curl http://localhost:8000/api/tasks/9/experiments/3/logs

# 实时流式日志（SSE）
curl http://localhost:8000/api/tasks/9/experiments/3/logs?follow=true
```

### 文件系统访问

```bash
# 查看任务汇总日志
tail -f ~/.local/share/inference-autotuner/logs/task_9.log

# 查看特定实验日志
tail -f ~/.local/share/inference-autotuner/logs/task_9_exp_3.log

# 列出所有实验日志
ls -lh ~/.local/share/inference-autotuner/logs/task_9_exp_*.log
```

## 日志内容

### 任务级别日志 (`task_9.log`)
包含所有实验的输出（汇总视图）：
```
[2025-11-28 10:28:18] [INFO] [ARQ Worker] Running experiment 3 with params: {...}
[2025-11-28 10:28:18] [INFO] [Experiment 3] Logging to experiment-specific file
[2025-11-28 10:28:18] [INFO] [Experiment 3] Status: DEPLOYING
...
[2025-11-28 10:38:18] [INFO] [ARQ Worker] Running experiment 4 with params: {...}
[2025-11-28 10:38:18] [INFO] [Experiment 4] Logging to experiment-specific file
[2025-11-28 10:38:18] [INFO] [Experiment 4] Status: DEPLOYING
...
```

### 实验级别日志 (`task_9_exp_3.log`)
只包含实验3的输出（包括僵尸进程的延迟输出）：
```
[2025-11-28 10:28:18] [INFO] [Experiment 3] Logging to experiment-specific file
[2025-11-28 10:28:18] [INFO] [Experiment 3] Status: DEPLOYING
[2025-11-28 10:31:40] [INFO] [Experiment 3] Status: BENCHMARKING
...
[2025-11-28 10:45:55] [INFO] [Benchmark] Completed in 854.4s
[2025-11-28 10:45:55] [INFO] [Benchmark] Exit code: 1
```

**关键区别**：如果实验3的subprocess在实验4开始后才完成，其输出只会出现在 `task_9_exp_3.log` 中，不会污染 `task_9_exp_4.log`。

## 注意事项

1. **文件数量增加**：每个实验产生一个日志文件
   - 50个实验的任务会产生51个日志文件（1个任务日志 + 50个实验日志）
   - 需要定期清理旧日志

2. **磁盘空间**：实验日志同时写入任务日志，日志量约为原来的2倍
   - 实验专属日志：独立文件
   - 任务汇总日志：包含所有实验
   - 建议定期归档或清理

3. **日志同步**：stdout/stderr重定向是全局的
   - 每次切换实验时需要重新设置logger
   - 确保在实验开始时调用 `setup_experiment_logging()`

## 未来改进

### 1. 日志轮转和归档
- 实现日志自动归档（例如：完成的实验日志压缩为 `.gz`）
- 定期清理超过N天的日志

### 2. 日志查询优化
- 支持按时间范围、关键词搜索日志
- 前端UI支持查看实验专属日志

### 3. 进程终止改进
- 实现主动终止超时的subprocess
- 彻底解决僵尸进程问题（见方案2：主动终止subprocess）

## 相关文档

- 日志混乱问题根因分析 - 详细的技术分析（见agentlog.md）
- [故障排查指南](./TROUBLESHOOTING.md) - 其他超时和日志相关问题
