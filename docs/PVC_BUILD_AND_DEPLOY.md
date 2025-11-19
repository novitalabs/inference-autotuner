# PVC Storage Implementation - Build and Deployment Guide

## Build Status: ✅ COMPLETE

The OME controller has been successfully built with PVC storage support implemented.

### Build Summary

**Date:** November 19, 2025
**Build Time:** ~3 minutes (after dependencies installed)
**Binary Size:** 94MB
**Location:** `/root/work/inference-autotuner/third_party/ome/bin/manager`

### Dependencies Installed

1. **Go 1.24.1** - Downloaded and installed to `/usr/local/go`
2. **Rust 1.91.1** + **Cargo 1.91.1** - Installed via rustup
3. **libssl-dev** - OpenSSL development libraries for Rust dependencies

### Build Steps Completed

```bash
# 1. Install Go 1.24.1
cd /tmp
wget https://go.dev/dl/go1.24.1.linux-amd64.tar.gz
sudo tar -C /usr/local -xzf go1.24.1.tar.gz

# 2. Install Rust/Cargo 1.91.1
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

# 3. Install OpenSSL dev libraries
sudo apt-get install -y libssl-dev pkg-config

# 4. Build OME manager
export PATH=/home/claude/.cargo/bin:/usr/local/go/bin:$PATH
export HTTP_PROXY=http://127.0.0.1:1081
export HTTPS_PROXY=http://127.0.0.1:1081
export GOPROXY=https://goproxy.io,direct
cd /root/work/inference-autotuner/third_party/ome
make ome-manager
```

### Code Changes Summary

#### Files Modified:
1. **base.go** (~55 lines added)
   - PVC volume creation logic in `UpdatePodSpecVolumes()`
   - PVC subPath support in `UpdateVolumeMounts()`
   - Node selector skip logic in `UpdatePodSpecNodeSelector()`

2. **engine.go** (2 lines modified)
   - Updated `UpdatePodSpecNodeSelector()` calls

3. **decoder.go** (2 lines modified)
   - Updated `UpdatePodSpecNodeSelector()` calls

4. **inference_service.yaml.j2** (~10 lines added)
   - Added annotation support for PVC configuration

5. **ome_controller.py** (~20 lines added/modified)
   - Added `storage` parameter to `deploy_inference_service()`

6. **orchestrator.py** (~7 lines added/modified)
   - Extract and pass storage config from task

7. **pvc_task.json** (~35 lines)
   - Example task configuration using PVC storage

**Total Changes:** ~140 lines of code across 7 files

### PVC Implementation Approach

Uses **annotation-based** configuration instead of full OEP-0004 spec:

```yaml
annotations:
  ome.io/storage-type: "pvc"
  ome.io/pvc-name: "model-storage-pvc"
  ome.io/pvc-subpath: "meta/llama-3-2-1b-instruct"
  ome.io/mount-path: "/raid/models/meta/llama-3-2-1b-instruct"
```

### Next Steps

Now that the build is complete, we need to:

1. **Deploy Updated Controller** to Kubernetes cluster
2. **Verify PVC** exists and is accessible
3. **Test Manual InferenceService** with PVC annotations
4. **Test Autotuner** end-to-end with `examples/pvc_task.json`
5. **Verify Backward Compatibility** with existing hostPath tasks

## Deployment Instructions

### Option 1: Build with GitHub Actions (Recommended)

Use GitHub Actions to build and push the Docker image automatically.

**See detailed instructions**: [`docs/GITHUB_ACTIONS_BUILD.md`](./GITHUB_ACTIONS_BUILD.md)

**Quick steps**:
1. Push code to GitHub repository
2. Go to Actions tab → "Build OME Controller with PVC Support"
3. Click "Run workflow" with desired registry and tag
4. Wait for build to complete (~5-10 minutes)
5. Deploy: `kubectl set image deployment/ome-controller-manager -n ome manager=<your-image>`

### Option 1b: Build and Push Docker Image Locally

```bash
# Build Docker image with new manager binary
cd /root/work/inference-autotuner/third_party/ome
make ome-image REGISTRY=<your-registry> TAG=pvc-support

# Push to registry
docker push <your-registry>/ome-manager:pvc-support

# Update deployment
kubectl set image deployment/ome-controller-manager \
  -n ome \
  manager=<your-registry>/ome-manager:pvc-support

# Restart controller
kubectl rollout restart deployment/ome-controller-manager -n ome
```

### Option 2: Direct Binary Replacement (Dev/Test Only)

```bash
# Copy binary to running pod
kubectl cp /root/work/inference-autotuner/third_party/ome/bin/manager \
  ome/$(kubectl get pod -n ome -l control-plane=controller-manager -o jsonpath='{.items[0].metadata.name}'):/manager

# Restart pod
kubectl delete pod -n ome -l control-plane=controller-manager
```

### Option 3: Local Development Testing

```bash
# Run manager locally (requires kubeconfig access)
export PATH=/home/claude/.cargo/bin:/usr/local/go/bin:$PATH
cd /root/work/inference-autotuner/third_party/ome
./bin/manager --kubeconfig ~/.kube/config
```

## Testing Plan

### Test 1: Verify PVC

```bash
# Check if PVC exists
kubectl get pvc -n autotuner model-storage-pvc

# Verify PVC is bound
kubectl describe pvc -n autotuner model-storage-pvc
```

### Test 2: Manual InferenceService

```bash
# Apply test InferenceService
kubectl apply -f /tmp/test-pvc-isvc.yaml

# Wait for pod creation
kubectl get pods -n autotuner -w

# Verify pod mounts PVC
kubectl get pod -n autotuner -l serving.kserve.io/inferenceservice=test-pvc-manual \
  -o yaml | grep -A 20 volumes

# Check pod logs
kubectl logs -n autotuner -l serving.kserve.io/inferenceservice=test-pvc-manual --follow
```

### Test 3: Autotuner Integration

```bash
# Run PVC task
python src/run_autotuner.py examples/pvc_task.json --mode ome --direct

# Expected: Task completes successfully with PVC storage
```

### Test 4: Backward Compatibility

```bash
# Run existing hostPath task
python src/run_autotuner.py examples/simple_task.json --mode ome --direct

# Expected: Task completes successfully with hostPath storage
```

## Troubleshooting

### Issue: Binary Not Found
**Solution:** Verify build completed: `ls -lh /root/work/inference-autotuner/third_party/ome/bin/manager`

### Issue: Pod Not Starting
**Check:**
1. PVC exists and is bound
2. PVC subpath exists within the volume
3. Pod has permissions to access PVC

### Issue: Model Not Found
**Check:**
1. PVC is mounted correctly: `kubectl exec -n autotuner <pod> -- ls /raid/models`
2. Subpath is correct
3. Mount path matches model configuration

## Build Environment Details

### System Info
- **OS:** Ubuntu 22.04 (Jammy)
- **Kernel:** Linux 5.15.0-143-generic
- **Architecture:** x86_64

### Compiler Versions
- **Go:** 1.24.1 linux/amd64
- **Rust:** 1.91.1 (ed61e7d7e 2025-11-07)
- **Cargo:** 1.91.1 (ea2d97820 2025-10-10)
- **GCC:** Included in Ubuntu base

### Proxy Configuration
- **HTTP Proxy:** http://127.0.0.1:1081
- **HTTPS Proxy:** http://127.0.0.1:1081
- **GOPROXY:** https://goproxy.io,direct

## References

- **Implementation Plan:** `/root/work/inference-autotuner/docs/PVC_IMPLEMENTATION_PLAN.md`
- **Implementation Summary:** `/root/work/inference-autotuner/docs/PVC_IMPLEMENTATION_SUMMARY.md`
- **OEP-0004 Spec:** `/root/work/inference-autotuner/third_party/ome/oeps/0004-pvc-storage-support/README.md`
- **Example Task:** `/root/work/inference-autotuner/examples/pvc_task.json`
- **Test InferenceService:** `/tmp/test-pvc-isvc.yaml`
