# ClusterBaseModel and ClusterServingRuntime Auto-Creation Implementation

**Date:** 2025-11-18
**Status:** Backend Complete, Frontend Pending

## Overview

Implemented preset-based ClusterBaseModel and ClusterServingRuntime configuration for OME tasks with automatic Kubernetes resource creation. Resources persist after tasks complete (no automatic cleanup).

---

## ✅ Completed: Backend Implementation

### Phase 1: Preset System & Database Schema

**1. Preset Definitions Created**
- **File:** `src/config/clusterbasemodel_presets.py` (220 lines)
  - 7 model presets: llama-3-2-1b, llama-3-2-3b, llama-3-1-70b, llama-3-3-70b, mistral-7b, mixtral-8x7b, deepseek-v3
  - Functions: `get_preset()`, `list_presets()`, `validate_custom_config()`, `merge_preset_with_overrides()`

- **File:** `src/config/clusterservingruntime_presets.py` (572 lines)
  - 5 runtime presets: sglang-llama-small, sglang-llama-large, vllm-llama-small, vllm-llama-large, sglang-mixtral-moe
  - Functions: `get_preset()`, `list_presets()`, `get_presets_by_runtime()`, `validate_custom_config()`, `merge_preset_with_overrides()`

**2. Database Schema Updated**
- **File:** `src/web/db/models.py`
  - Added 4 columns to `Task` model:
    - `clusterbasemodel_config: JSON` - Stores preset or custom spec
    - `clusterservingruntime_config: JSON` - Stores preset or custom spec
    - `created_clusterbasemodel: String` - Name of CBM if auto-created
    - `created_clusterservingruntime: String` - Name of CSR if auto-created

**3. API Schemas Updated**
- **File:** `src/web/schemas/__init__.py`
  - Updated `TaskCreate` to accept `clusterbasemodel_config` and `clusterservingruntime_config`
  - Updated `TaskResponse` to include `created_clusterbasemodel` and `created_clusterservingruntime`

---

### Phase 2: K8s Templates & Controller Logic

**4. Jinja2 Templates Created**
- **File:** `src/templates/clusterbasemodel.yaml.j2` (38 lines)
  - Renders ClusterBaseModel YAML from spec
  - Supports storage config, labels, annotations, nodeSelector

- **File:** `src/templates/clusterservingruntime.yaml.j2` (246 lines)
  - Renders ClusterServingRuntime YAML from spec
  - Supports engineConfig, routerConfig, probes, resources, tolerations

**5. OMEController Enhanced**
- **File:** `src/controllers/ome_controller.py` (+197 lines)
  - New methods:
    - `ensure_clusterbasemodel(name, spec, labels, annotations)` - Check/create CBM
    - `_create_clusterbasemodel(...)` - Create CBM from template
    - `list_clusterbasemodels()` - List all CBMs in cluster
    - `ensure_clusterservingruntime(name, spec, labels, annotations)` - Check/create CSR
    - `_create_clusterservingruntime(...)` - Create CSR from template
    - `list_clusterservingruntimes()` - List all CSRs in cluster
  - Idempotent resource creation (checks existence before creating)

**6. Orchestrator Updated**
- **File:** `src/orchestrator.py` (+127 lines)
  - Added resource creation before InferenceService deployment (Step 0a/0b)
  - New helper methods:
    - `_ensure_clusterbasemodel(config, fallback_name)` - Handle preset/custom config
    - `_ensure_clusterservingruntime(config, fallback_name)` - Handle preset/custom config
  - Supports preset selection with overrides
  - Deep merge for custom configurations
  - Tracks created resources in experiment results

---

### Phase 3: API Routes

**7. OME Resources API Created**
- **File:** `src/web/routes/ome_resources.py` (95 lines)
  - Endpoints:
    - `GET /api/ome/clusterbasemodels` - List CBMs in cluster
    - `GET /api/ome/clusterbasemodel-presets` - List available presets
    - `GET /api/ome/clusterservingruntimes` - List CSRs in cluster
    - `GET /api/ome/clusterservingruntime-presets` - List all presets
    - `GET /api/ome/clusterservingruntime-presets/{runtime_type}` - Filter by runtime

- **File:** `src/web/app.py` (modified)
  - Registered `ome_resources.router` in FastAPI app

---

### Phase 4: Worker Integration

**8. Worker Updated**
- **File:** `src/web/workers/autotuner_worker.py` (+14 lines)
  - Persists created resource names to task after first experiment
  - Saves `task.created_clusterbasemodel` and `task.created_clusterservingruntime`
  - Logs resource creation for tracking

---

## Task Configuration Format

### Example: Using Presets

```json
{
  "task_name": "my-optimization-task",
  "model": {
    "id_or_path": "llama-3-2-1b-instruct",
    "namespace": "autotuner"
  },
  "base_runtime": "sglang",
  "deployment_mode": "ome",
  "clusterbasemodel_config": {
    "preset": "llama-3-2-1b-instruct"
  },
  "clusterservingruntime_config": {
    "preset": "sglang-llama-small"
  },
  "parameters": {
    "tp-size": [1, 2],
    "mem-fraction-static": [0.85, 0.9]
  },
  "optimization": {
    "strategy": "grid_search",
    "objective": "minimize_latency",
    "max_iterations": 10
  },
  "benchmark": {
    "task": "text-to-text",
    "model_tokenizer": "meta-llama/Llama-3.2-1B-Instruct",
    "traffic_scenarios": ["D(100,100)"],
    "num_concurrency": [1, 4]
  }
}
```

### Example: Preset with Overrides

```json
{
  "clusterbasemodel_config": {
    "preset": "llama-3-2-1b-instruct",
    "overrides": {
      "storage": {
        "path": "/custom/path/to/model"
      }
    },
    "labels": {
      "environment": "production"
    }
  },
  "clusterservingruntime_config": {
    "preset": "sglang-llama-small",
    "overrides": {
      "spec": {
        "engineConfig": {
          "runner": {
            "resources": {
              "requests": {
                "memory": "40Gi"
              }
            }
          }
        }
      }
    }
  }
}
```

### Example: Custom Configuration

```json
{
  "clusterbasemodel_config": {
    "name": "my-custom-model",
    "spec": {
      "displayName": "custom.my-model",
      "vendor": "custom",
      "version": "1.0.0",
      "storage": {
        "storageUri": "hf://my-org/my-model",
        "path": "/models/my-model",
        "key": "hf-token"
      }
    }
  }
}
```

---

## Workflow

### Resource Creation Flow

```
1. Task created with clusterbasemodel_config and clusterservingruntime_config
   ↓
2. Task started (ARQ worker)
   ↓
3. First experiment begins
   ↓
4. Orchestrator checks clusterbasemodel_config
   ├─ If preset specified → Load preset from clusterbasemodel_presets
   ├─ Apply overrides if provided
   └─ Call OMEController.ensure_clusterbasemodel()
       ├─ Check if ClusterBaseModel exists in cluster
       ├─ If not → Create from template
       └─ Return (name, created_flag)
   ↓
5. Orchestrator checks clusterservingruntime_config
   ├─ If preset specified → Load preset from clusterservingruntime_presets
   ├─ Apply overrides if provided
   └─ Call OMEController.ensure_clusterservingruntime()
       ├─ Check if ClusterServingRuntime exists in cluster
       ├─ If not → Create from template
       └─ Return (name, created_flag)
   ↓
6. Deploy InferenceService using created/existing resources
   ↓
7. Run benchmark and calculate metrics
   ↓
8. Save created resource names to task.created_clusterbasemodel/clusterservingruntime
   ↓
9. Subsequent experiments reuse existing resources
```

### Key Behaviors

- **Idempotency**: Resources checked before creation; no duplicates
- **Persistence**: Resources remain in cluster after task completion
- **Reusability**: Subsequent tasks can use same resources by name
- **Fallback**: If creation fails, orchestrator uses provided model_name/runtime_name
- **Tracking**: Task records which resources it created for audit purposes

---

## Code Statistics

### New Files Created (5)
- `src/config/clusterbasemodel_presets.py` (220 lines)
- `src/config/clusterservingruntime_presets.py` (572 lines)
- `src/templates/clusterbasemodel.yaml.j2` (38 lines)
- `src/templates/clusterservingruntime.yaml.j2` (246 lines)
- `src/web/routes/ome_resources.py` (95 lines)

**Total New Code:** ~1,171 lines

### Files Modified (6)
- `src/web/db/models.py` (+4 columns)
- `src/web/schemas/__init__.py` (+6 fields)
- `src/controllers/ome_controller.py` (+197 lines)
- `src/orchestrator.py` (+127 lines)
- `src/web/app.py` (+2 lines)
- `src/web/workers/autotuner_worker.py` (+14 lines)

**Total Modified:** ~350 lines

**Grand Total:** ~1,521 lines of backend code

---

## ⏳ Pending: Frontend Implementation

### Remaining Tasks

1. **ClusterBaseModelForm.tsx** - Form component with:
   - Preset selector dropdown
   - Custom configuration editor
   - Override fields for storage path, vendor, version
   - Validation and preview

2. **ClusterServingRuntimeForm.tsx** - Form component with:
   - Preset selector filtered by runtime type
   - Runtime-specific configuration options
   - Resource limits editor (CPU, memory, GPU)
   - Image override field

3. **ResourceBrowser.tsx** - Browser component with:
   - List existing ClusterBaseModels in cluster
   - List existing ClusterServingRuntimes in cluster
   - Select from existing resources
   - Display resource details (vendor, version, image, resources)

4. **NewTask.tsx Integration** - Add form sections:
   - "ClusterBaseModel Configuration" accordion/tab
   - "ClusterServingRuntime Configuration" accordion/tab
   - Show only when `deployment_mode === "ome"`
   - Wire up to task creation API

5. **Tasks.tsx Display** - Show created resources:
   - Display `created_clusterbasemodel` name with icon
   - Display `created_clusterservingruntime` name with icon
   - Add badges for auto-created vs. existing resources

6. **API Client Services** - Add TypeScript functions:
   - `fetchClusterBaseModels()`
   - `fetchClusterBaseModelPresets()`
   - `fetchClusterServingRuntimes()`
   - `fetchClusterServingRuntimePresets()`
   - `fetchClusterServingRuntimePresetsByRuntime(runtime_type)`

**Estimated Frontend Work:** 4-6 hours (800-1,200 lines of React/TypeScript code)

---

## Testing Strategy

### Backend Testing (Ready Now)

**1. Test Preset Loading**
```python
from config import clusterbasemodel_presets
presets = clusterbasemodel_presets.list_presets()
print(f"Found {len(presets)} ClusterBaseModel presets")
```

**2. Test Resource Creation (OME mode)**
```bash
# Create task with preset config
curl -X POST http://localhost:8000/api/tasks/ \
  -H "Content-Type: application/json" \
  -d @examples/ome_with_presets_task.json

# Start task
curl -X POST http://localhost:8000/api/tasks/{id}/start

# Check created resources
kubectl get clusterbasemodels
kubectl get clusterservingruntimes
```

**3. Test API Endpoints**
```bash
# List presets
curl http://localhost:8000/api/ome/clusterbasemodel-presets
curl http://localhost:8000/api/ome/clusterservingruntime-presets

# List cluster resources (requires K8s)
curl http://localhost:8000/api/ome/clusterbasemodels
curl http://localhost:8000/api/ome/clusterservingruntimes
```

### Frontend Testing (After UI Implementation)

1. Navigate to "New Task" page
2. Select deployment mode: OME
3. Configure ClusterBaseModel (preset or custom)
4. Configure ClusterServingRuntime (preset or custom)
5. Submit task and verify resource creation
6. Check Tasks page shows created resource names

---

## Database Migration

**Required Migration:**
```sql
ALTER TABLE tasks ADD COLUMN clusterbasemodel_config JSON;
ALTER TABLE tasks ADD COLUMN clusterservingruntime_config JSON;
ALTER TABLE tasks ADD COLUMN created_clusterbasemodel VARCHAR;
ALTER TABLE tasks ADD COLUMN created_clusterservingruntime VARCHAR;
```

**Migration handled automatically** by SQLAlchemy when starting the server (development mode).

---

## API Documentation

Access Swagger UI at: `http://localhost:8000/docs`

New endpoints under `/api/ome`:
- `/clusterbasemodels` - List cluster resources
- `/clusterbasemodel-presets` - List available presets
- `/clusterservingruntimes` - List cluster resources
- `/clusterservingruntime-presets` - List available presets
- `/clusterservingruntime-presets/{runtime_type}` - Filter by runtime

---

## Architecture Decisions

### 1. Preset-Based Configuration
**Decision:** Use predefined presets instead of full YAML in task JSON
**Rationale:**
- Simplifies user experience (select from dropdown vs. write YAML)
- Enforces best practices for common configurations
- Reduces error rate (validated presets)
- Still allows custom configurations for power users

### 2. No Automatic Cleanup
**Decision:** Created resources persist after task completion
**Rationale:**
- ClusterBaseModels/ClusterServingRuntimes are expensive to create (download models)
- Reusability across multiple tasks
- Cluster-wide resources (not namespaced)
- Manual cleanup gives admins control

### 3. Idempotent Resource Creation
**Decision:** Check existence before creating resources
**Rationale:**
- Multiple tasks can reference same resources
- Avoids conflicts when resources already exist
- Safe to retry failed tasks

### 4. Deep Merge for Overrides
**Decision:** Support overriding specific fields in presets
**Rationale:**
- Balance between preset convenience and customization
- Users can tweak memory limits without writing full spec
- Preserves preset defaults for unspecified fields

---

## Next Steps

1. **Implement Frontend Components** (~4-6 hours)
   - Forms for CBM and CSR configuration
   - Resource browser component
   - Integration with NewTask page
   - Display in Tasks page

2. **Create Example Task Configurations**
   - `examples/ome_with_presets_task.json`
   - `examples/ome_with_custom_resources_task.json`
   - `examples/ome_with_overrides_task.json`

3. **Update Documentation**
   - `docs/OME_INSTALLATION.md` - Add preset usage
   - `docs/TASK_CONFIGURATION.md` - Document new fields
   - `README.md` - Update features list

4. **Testing**
   - End-to-end test with actual OME cluster
   - Verify resource creation and reuse
   - Test preset overrides and custom configs

5. **Update agentlog.md**
   - Document milestone completion
   - Include implementation summary

---

## Success Criteria

✅ **Backend Complete:**
- [x] Preset definitions for 7 models, 5 runtimes
- [x] Database schema extended with 4 columns
- [x] K8s templates for CBM and CSR
- [x] OMEController methods for resource management
- [x] Orchestrator integration for resource creation
- [x] API routes for listing presets and resources
- [x] Worker persistence of created resource names

⏳ **Frontend Pending:**
- [ ] ClusterBaseModelForm component
- [ ] ClusterServingRuntimeForm component
- [ ] ResourceBrowser component
- [ ] NewTask page integration
- [ ] Tasks page display updates

---

## References

- **Design Doc:** Initial planning in research phase
- **OME CRDs:** `third_party/ome/config/models/` and `third_party/ome/config/runtimes/`
- **Related Features:** Quantization config (Milestone 3), Parallel config (Milestone 3)

---

**Implementation completed:** November 18, 2025
**Next milestone:** Frontend implementation + end-to-end testing
