# Boolean Parameter Handling

## Overview

The autotuner supports boolean parameters for CLI flags that act as switches. Boolean parameters are handled specially:
- **`true`**: The parameter flag is added to the command (e.g., `--enable-mixed-chunk`)
- **`false`**: The parameter is omitted from the command entirely

This is the correct behavior for command-line flags that don't take values.

## Implementation

### Docker Controller

The Docker controller (`src/controllers/docker_controller.py`) handles boolean parameters when building container commands:

```python
for param_name, param_value in parameters.items():
    cli_param = f"--{param_name}" if not param_name.startswith("--") else param_name

    if isinstance(param_value, bool):
        if param_value:  # Only add flag if True
            command_str += f" {cli_param}"
        # If False, skip this parameter entirely
    else:
        command_str += f" {cli_param} {param_value}"
```

### Frontend Parsing

The frontend (`frontend/src/components/PresetEditModal.tsx`) automatically parses boolean strings:

```typescript
const parseParameterValue = (valueStr: string): any[] => {
  const parts = valueStr.split(',').map(s => s.trim()).filter(Boolean);

  // Check if all parts are booleans
  const allBooleans = parts.every(part => part === 'true' || part === 'false');
  if (allBooleans) {
    return parts.map(part => part === 'true');
  }
  // ... handle numbers and strings
}
```

## Usage

### In the Web UI

1. Open the preset editor
2. Add a parameter (e.g., `enable-mixed-chunk`)
3. In the Values field, enter boolean values:
   - `true` - to enable the flag
   - `false` - to disable the flag
   - `true, false` - to test both configurations

### Example Preset

```json
{
  "name": "Boolean Test",
  "runtime": "sglang",
  "parameters": {
    "tensor-parallel-size": [1, 2],
    "enable-mixed-chunk": [true],
    "disable-cuda-graph": [false],
    "enable-lora": [true, false]
  }
}
```

This will generate 4 parameter combinations:
1. `tp=1, enable-lora=true` → Command includes `--enable-mixed-chunk --enable-lora`
2. `tp=1, enable-lora=false` → Command includes `--enable-mixed-chunk` only
3. `tp=2, enable-lora=true` → Command includes `--enable-mixed-chunk --enable-lora`
4. `tp=2, enable-lora=false` → Command includes `--enable-mixed-chunk` only

Note: `disable-cuda-graph: [false]` is never added to any command.

## Common Use Cases

### Testing Feature Flags

Test whether a feature improves performance:

```json
{
  "enable-prefix-caching": [true, false],
  "enable-chunked-prefill": [true, false]
}
```

This creates 4 experiments testing all combinations.

### Disabling Default Features

Some flags disable default behavior:

```json
{
  "disable-cuda-graph": [false],  // Keep CUDA graphs enabled
  "disable-radix-cache": [true]   // Disable radix cache
}
```

### Always-On Features

Enable a feature in all experiments:

```json
{
  "enable-mixed-chunk": [true],  // Always enabled
  "enable-lora": [true]           // Always enabled
}
```

## Common Boolean Parameters

### SGLang
- `enable-mixed-chunk` - Enable mixed chunk prefill
- `enable-lora` - Enable LoRA adapters
- `disable-cuda-graph` - Disable CUDA graphs
- `disable-radix-cache` - Disable radix cache
- `enable-torch-compile` - Enable torch.compile
- `disable-overlap-schedule` - Disable overlapped scheduling
- `enable-hierarchical-cache` - Enable hierarchical cache

### vLLM
- `enable-chunked-prefill` - Enable chunked prefill
- `enable-prefix-caching` - Enable prefix caching
- `enable-lora` - Enable LoRA adapters
- `enforce-eager` - Disable CUDA graphs (eager mode)
- `disable-custom-all-reduce` - Disable custom all-reduce
- `disable-sliding-window` - Disable sliding window attention
- `multi-step-stream-outputs` - Enable multi-step streaming
- `async-scheduling` - Enable async scheduling

## Tips

1. **Use `false` to skip parameters**: If you want to ensure a flag is never added, use `[false]` as the value.

2. **Test both modes**: Use `[true, false]` to compare performance with and without a feature.

3. **Default behavior**: Most CLI flags default to `false` (disabled) when not present. Check the runtime documentation to confirm.

4. **Negative flags**: Parameters like `disable-cuda-graph` work in reverse - `true` means disable, `false` means keep enabled (by omitting the flag).

## Limitations

### OME Controller

The OME controller currently uses Jinja2 templates and doesn't have dynamic boolean handling. If you need boolean parameters with OME, you'll need to:
1. Update the template (`src/templates/inference_service.yaml.j2`)
2. Add conditional logic for each boolean parameter

Example template addition:
```jinja2
{% if enable_mixed_chunk %}
  - --enable-mixed-chunk
{% endif %}
```

## Testing

Run the example script to see boolean parameters in action:

```bash
python3 examples/boolean_parameters_example.py
```

Or run the unit test:

```bash
python3 test_boolean_params.py
```
