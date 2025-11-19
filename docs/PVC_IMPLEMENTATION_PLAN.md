# PVC Implementation Plan for OME (Minimal for Autotuner)

## 🎯 Goal

Enable the inference autotuner to run OME tasks with models stored in PVCs instead of hostPath, without implementing the full OEP-0004 specification.

## 📋 Requirements Analysis

### What Autotuner Needs

1. ✅ **Create InferenceService** with a model reference
2. ✅ **InferenceService pods mount PVC** to access model files
3. ✅ **Models already exist in PVC** (pre-populated)
4. ✅ **Metadata already known** (from task configuration)
5. ❌ **Don't need**: Automatic metadata extraction
6. ❌ **Don't need**: `pvc://` URI parsing in BaseModel controller
7. ❌ **Don't need**: Cross-namespace PVC support

### Current Workflow

```
Autotuner → OMEController.deploy_inference_service()
    ↓
Create InferenceService YAML (references ClusterBaseModel)
    ↓
OME InferenceService Controller creates Deployment
    ↓
Deployment uses hostPath volume (from BaseModel.spec.storage.path)
    ↓
Pod runs on node with model files
```

### Desired Workflow with PVC

```
Autotuner → OMEController.deploy_inference_service()
    ↓
Create InferenceService YAML (references ClusterBaseModel with PVC annotation)
    ↓
OME InferenceService Controller detects PVC annotation
    ↓
Deployment uses PVC volume instead of hostPath
    ↓
Pod runs on any node (PVC accessible from all nodes)
```

## 🏗️ Minimal Implementation Design

### Approach: Annotation-Based PVC Support

Use **Kubernetes annotations** to tell the InferenceService controller to use PVC instead of hostPath.

**Benefits**:
- ✅ No changes to CRD schemas
- ✅ No BaseModel controller changes needed
- ✅ Simple to implement (only modify InferenceService controller)
- ✅ Backward compatible (existing InferenceServices continue to work)
- ✅ Easy to test and rollback

**Limitations**:
- ⚠️ Requires manual PVC creation and model population
- ⚠️ Metadata must be specified manually in ClusterBaseModel
- ⚠️ PVC must be in same namespace as InferenceService

### Design Details

#### 1. Annotation Schema

Add annotations to InferenceService to specify PVC storage:

```yaml
apiVersion: ome.io/v1beta1
kind: InferenceService
metadata:
  name: test-isvc
  namespace: autotuner
  annotations:
    # Enable PVC storage
    ome.io/storage-type: "pvc"
    # PVC name to mount
    ome.io/pvc-name: "model-storage-pvc"
    # SubPath within PVC (optional)
    ome.io/pvc-subpath: "meta/llama-3-2-1b-instruct"
    # Mount path in container (optional, defaults to /raid/models/<model-name>)
    ome.io/mount-path: "/raid/models/meta/llama-3-2-1b-instruct"
spec:
  model:
    name: llama-3-2-1b-instruct
  runtime:
    name: sglang-llama-small
```

#### 2. ClusterBaseModel Configuration

Use existing ClusterBaseModel but **without real storage path**:

```yaml
apiVersion: ome.io/v1beta1
kind: ClusterBaseModel
metadata:
  name: llama-3-2-1b-instruct
  annotations:
    ome.io/storage-type: "pvc"  # Mark as PVC storage
spec:
  displayName: meta.llama-3.2-1b-instruct
  disabled: false
  modelFormat:
    name: safetensors
  storage:
    # Dummy storageUri (won't be downloaded by model-agent)
    storageUri: "pvc://model-storage-pvc/meta/llama-3-2-1b-instruct"
  # Manually specify metadata (no automatic extraction)
  modelArchitecture: LlamaForCausalLM
  maxTokens: 131072
  modelConfiguration:
    architecture: LlamaForCausalLM
    context_length: 131072
    model_type: llama
    parameter_count: "1.24B"
  vendor: meta
```

## 🔧 Code Changes Required

### File 1: InferenceService Controller - Volume Mounting

**File**: `third_party/ome/pkg/controller/v1beta1/inferenceservice/components/base.go`

**Function**: `UpdatePodSpecVolumes()` (line 262)

**Changes**:
```go
func UpdatePodSpecVolumes(b *BaseComponentFields, isvc *v1beta1.InferenceService, podSpec *corev1.PodSpec, objectMeta *metav1.ObjectMeta) {
	// NEW: Check for PVC storage annotation
	if storageType, ok := objectMeta.Annotations["ome.io/storage-type"]; ok && storageType == "pvc" {
		pvcName := objectMeta.Annotations["ome.io/pvc-name"]
		if pvcName == "" {
			b.Log.Error(nil, "PVC storage requested but ome.io/pvc-name annotation is missing")
			return
		}

		subPath := objectMeta.Annotations["ome.io/pvc-subpath"]
		mountPath := objectMeta.Annotations["ome.io/mount-path"]
		if mountPath == "" && b.BaseModel != nil && b.BaseModel.Storage != nil && b.BaseModel.Storage.Path != nil {
			mountPath = *b.BaseModel.Storage.Path
		}
		if mountPath == "" {
			mountPath = constants.ModelDefaultMountPath
		}

		modelVolume := corev1.Volume{
			Name: b.BaseModelMeta.Name,
			VolumeSource: corev1.VolumeSource{
				PersistentVolumeClaim: &corev1.PersistentVolumeClaimVolumeSource{
					ClaimName: pvcName,
					ReadOnly:  true,
				},
			},
		}
		podSpec.Volumes = append(podSpec.Volumes, modelVolume)

		b.Log.Info("Using PVC storage for model",
			"pvcName", pvcName,
			"subPath", subPath,
			"mountPath", mountPath)
		return
	}

	// EXISTING: Add model volume if base model is specified (hostPath)
	if b.BaseModel != nil && b.BaseModel.Storage != nil && b.BaseModel.Storage.Path != nil && b.BaseModelMeta != nil {
		modelVolume := corev1.Volume{
			Name: b.BaseModelMeta.Name,
			VolumeSource: corev1.VolumeSource{
				HostPath: &corev1.HostPathVolumeSource{
					Path: *b.BaseModel.Storage.Path,
				},
			},
		}
		podSpec.Volumes = append(podSpec.Volumes, modelVolume)
	}

	// ... rest of existing code ...
}
```

**Lines of code**: ~40 lines added

### File 2: InferenceService Controller - Volume Mounts

**File**: `third_party/ome/pkg/controller/v1beta1/inferenceservice/components/base.go`

**Function**: `UpdateVolumeMounts()` (line 83)

**Changes**:
```go
func UpdateVolumeMounts(b *BaseComponentFields, isvc *v1beta1.InferenceService, container *corev1.Container, objectMeta *metav1.ObjectMeta) {
	if container == nil {
		b.Log.Error(errors.New("container is nil"), "UpdateVolumeMounts: container is nil")
		return
	}

	// Add model volume mount if base model is specified and it's necessary
	if b.BaseModel != nil && b.BaseModel.Storage != nil && b.BaseModel.Storage.Path != nil && b.BaseModelMeta != nil {
		if isvcutils.IsOriginalModelVolumeMountNecessary(objectMeta.Annotations) {
			// NEW: Handle PVC subPath
			subPath := ""
			if storageType, ok := objectMeta.Annotations["ome.io/storage-type"]; ok && storageType == "pvc" {
				subPath = objectMeta.Annotations["ome.io/pvc-subpath"]
			}

			vm := corev1.VolumeMount{
				Name:      b.BaseModelMeta.Name,
				MountPath: *b.BaseModel.Storage.Path,
				ReadOnly:  true,
				SubPath:   subPath,  // NEW: Add subPath support
			}
			isvcutils.AppendVolumeMount(container, &vm)
		}
	}

	// ... rest of existing code ...
}
```

**Lines of code**: ~10 lines added

### File 3: InferenceService Controller - Node Selector

**File**: `third_party/ome/pkg/controller/v1beta1/inferenceservice/components/base.go`

**Function**: `UpdatePodSpecNodeSelector()` (check if exists)

**Changes**:
```go
func UpdatePodSpecNodeSelector(b *BaseComponentFields, isvc *v1beta1.InferenceService, podSpec *corev1.PodSpec, objectMeta *metav1.ObjectMeta) {
	// NEW: Skip node selector for PVC storage
	if storageType, ok := objectMeta.Annotations["ome.io/storage-type"]; ok && storageType == "pvc" {
		b.Log.Info("Using PVC storage, skipping node selector", "inferenceService", isvc.Name)
		return
	}

	// EXISTING: node selector logic for hostPath storage
	// ... existing code ...
}
```

**Lines of code**: ~5 lines added

### File 4: Model Agent - Skip PVC Storage

**File**: `third_party/ome/pkg/modelagent/gopher.go`

**Status**: ✅ **Already implemented** (lines 376-380, 479-483)

No changes needed!

## 📝 Autotuner Integration

### Changes to Autotuner Code

#### File 1: Task Configuration Schema

Add optional PVC configuration to task JSON:

```json
{
  "task_name": "llama-test",
  "model": {
    "id_or_path": "llama-3-2-1b-instruct",
    "namespace": "autotuner"
  },
  "storage": {
    "type": "pvc",
    "pvc_name": "model-storage-pvc",
    "pvc_subpath": "meta/llama-3-2-1b-instruct",
    "mount_path": "/raid/models/meta/llama-3-2-1b-instruct"
  },
  "base_runtime": "sglang",
  "parameters": { ... },
  "benchmark": { ... }
}
```

#### File 2: InferenceService Template

**File**: `src/templates/inference_service.yaml.j2`

**Changes**:
```yaml
apiVersion: ome.io/v1beta1
kind: InferenceService
metadata:
  name: {{ isvc_name }}
  namespace: {{ namespace }}
  labels:
    autotuner.io/task: "{{ task_name }}"
    autotuner.io/experiment-id: "{{ experiment_id }}"
  annotations:
    {% if storage and storage.type == 'pvc' %}
    ome.io/storage-type: "pvc"
    ome.io/pvc-name: "{{ storage.pvc_name }}"
    {% if storage.pvc_subpath %}
    ome.io/pvc-subpath: "{{ storage.pvc_subpath }}"
    {% endif %}
    {% if storage.mount_path %}
    ome.io/mount-path: "{{ storage.mount_path }}"
    {% endif %}
    {% endif %}
spec:
  model:
    name: {{ model_name }}
  runtime:
    name: {{ runtime_name }}
    apiGroup: ome.io
    kind: ClusterServingRuntime
  engine:
    minReplicas: 1
    maxReplicas: 1
    runner:
      name: ome-container
      resources:
        limits:
          nvidia.com/gpu: {{ params.get('tpsize', params.get('tp_size', params.get('tp-size', 1))) }}
        requests:
          nvidia.com/gpu: {{ params.get('tpsize', params.get('tp_size', params.get('tp-size', 1))) }}
```

#### File 3: OME Controller

**File**: `src/controllers/ome_controller.py`

**Function**: `deploy_inference_service()`

Pass storage configuration to template:
```python
def deploy_inference_service(
    self,
    isvc_name: str,
    namespace: str,
    model_name: str,
    runtime_name: str,
    parameters: Dict[str, Any],
    task_name: str = "",
    experiment_id: str = "",
    storage: Optional[Dict[str, Any]] = None,  # NEW parameter
) -> bool:
    # ... existing code ...

    isvc_manifest = self.isvc_template.render(
        isvc_name=isvc_name,
        namespace=namespace,
        model_name=model_name,
        runtime_name=runtime_name,
        params=parameters,
        task_name=task_name,
        experiment_id=experiment_id,
        storage=storage,  # NEW: Pass storage config
    )

    # ... rest of existing code ...
```

## 📊 Implementation Estimate

### Code Changes Summary

| Component | File | Lines Added | Difficulty |
|-----------|------|-------------|-----------|
| InferenceService Volume | base.go | ~40 | 🟢 Easy |
| InferenceService Mount | base.go | ~10 | 🟢 Easy |
| InferenceService NodeSelector | base.go | ~5 | 🟢 Easy |
| Model Agent | gopher.go | 0 | ✅ Done |
| Autotuner Template | inference_service.yaml.j2 | ~10 | 🟢 Easy |
| Autotuner Controller | ome_controller.py | ~5 | 🟢 Easy |
| **Total** | | **~70 lines** | **🟢 Easy** |

### Time Estimate

**Total: 1-2 days** (single developer)

- OME code changes: **4-6 hours**
  - Modify base.go: 3-4 hours
  - Test and debug: 1-2 hours

- Autotuner code changes: **2-3 hours**
  - Update templates: 1 hour
  - Update controller: 1 hour
  - Test integration: 1 hour

- Documentation: **1 hour**

## 🧪 Testing Plan

### Phase 1: Unit Testing (Manual)

1. **Create PVC and populate models**:
   ```bash
   kubectl apply -f model-storage-pv.yaml
   # Verify models exist in PVC
   ```

2. **Create ClusterBaseModel with PVC annotation**:
   ```yaml
   metadata:
     annotations:
       ome.io/storage-type: "pvc"
   ```

3. **Create InferenceService with PVC annotations**:
   ```yaml
   metadata:
     annotations:
       ome.io/storage-type: "pvc"
       ome.io/pvc-name: "model-storage-pvc"
       ome.io/pvc-subpath: "meta/llama-3-2-1b-instruct"
   ```

4. **Verify pod uses PVC volume**:
   ```bash
   kubectl get pod -n autotuner <pod-name> -o yaml | grep -A 10 volumes
   # Should see PersistentVolumeClaim, not HostPath
   ```

5. **Verify pod starts successfully**:
   ```bash
   kubectl logs -n autotuner <pod-name>
   # Should see model loading successfully
   ```

### Phase 2: Autotuner Integration Testing

1. **Create test task JSON with PVC storage**
2. **Run autotuner**: `python src/run_autotuner.py test-pvc-task.json --mode ome --direct`
3. **Verify experiments run successfully**
4. **Compare with hostPath mode** (same task, different storage)

### Phase 3: Regression Testing

1. **Test existing hostPath tasks still work**
2. **Test mixed workloads** (some PVC, some hostPath)

## 🚀 Rollout Plan

### Step 1: Develop and Test OME Changes (Day 1)

```bash
cd /root/work/inference-autotuner/third_party/ome
# Make changes to pkg/controller/v1beta1/inferenceservice/components/base.go
# Rebuild OME controller
make manager
```

### Step 2: Deploy Updated OME Controller (Day 1)

```bash
# Update controller image
kubectl set image deployment/ome-controller-manager \
  -n ome manager=<new-image>

# Or restart with local binary for testing
kubectl delete pod -n ome -l app=ome-controller-manager
```

### Step 3: Update Autotuner Code (Day 1-2)

```bash
cd /root/work/inference-autotuner
# Update templates and controller
# Test with example PVC task
```

### Step 4: Create Example Task (Day 2)

```bash
# Create example PVC task configuration
cat > examples/pvc_task.json << EOF
{
  "task_name": "pvc-test",
  "model": {
    "id_or_path": "llama-3-2-1b-instruct",
    "namespace": "autotuner"
  },
  "storage": {
    "type": "pvc",
    "pvc_name": "model-storage-pvc",
    "pvc_subpath": "meta/llama-3-2-1b-instruct"
  },
  "base_runtime": "sglang",
  "parameters": {
    "tp-size": [1],
    "mem-fraction-static": [0.9]
  },
  "benchmark": {
    "task": "text-to-text",
    "traffic_scenarios": ["D(100,100)"],
    "num_concurrency": [1]
  }
}
EOF

# Run test
python src/run_autotuner.py examples/pvc_task.json --mode ome --direct
```

## 📋 Acceptance Criteria

- ✅ InferenceService with PVC annotations creates pods with PVC volumes
- ✅ InferenceService without PVC annotations still uses hostPath (backward compatible)
- ✅ Pods start successfully and load models from PVC
- ✅ Autotuner can run experiments using PVC storage
- ✅ No regression in existing hostPath functionality
- ✅ Documentation updated with PVC usage examples

## 🔮 Future Enhancements (Postponed)

These are **out of scope** for minimal implementation:

1. ❌ Full `pvc://` URI support in BaseModel controller
2. ❌ Automatic metadata extraction jobs
3. ❌ Cross-namespace PVC access
4. ❌ PVC provisioning automation
5. ❌ Dynamic PVC creation
6. ❌ Multiple PVC support per InferenceService
7. ❌ PVC storage class validation
8. ❌ ReadWriteOnce vs ReadWriteMany detection

These can be added later if needed, following the full OEP-0004 specification.

## 📚 References

- OEP-0004 Document: `third_party/ome/oeps/0004-pvc-storage-support/README.md`
- Current PVC Investigation: `docs/PVC_STORAGE.md`
- OME InferenceService Components: `third_party/ome/pkg/controller/v1beta1/inferenceservice/components/`
