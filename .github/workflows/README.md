# GitHub Actions: Build OME Controller Image

## Quick Start

This workflow builds the OME controller Docker image with PVC support using GitHub Actions.

### Prerequisites

1. **Push your code** to a GitHub repository
2. **Enable GitHub Actions** in your repository settings
3. **Ensure workflow permissions** allow package writes (Settings → Actions → Workflow permissions → "Read and write permissions")

### How to Use

#### Option 1: Manual Trigger from GitHub UI

1. Go to your repository on GitHub
2. Click **Actions** tab
3. Select **"Build OME Controller with PVC Support"** workflow
4. Click **"Run workflow"** button
5. Fill in parameters:
   - **registry**: Your container registry (e.g., `ghcr.io/your-username`)
   - **tag**: Image tag (e.g., `pvc-v1.0.0`)
6. Click **"Run workflow"**
7. Monitor build progress (takes ~5-10 minutes)

#### Option 2: Automatic Build on Push

The workflow automatically triggers when you push to `feature/ome` branch:

```bash
git add .github/workflows/build-ome-pvc.yml
git add third_party/ome/
git commit -m "feat: add PVC storage support"
git push origin feature/ome
```

### Deploy Built Image

After the build completes:

```bash
# Update your deployment with the new image
kubectl set image deployment/ome-controller-manager \
  -n ome \
  manager=ghcr.io/your-username/ome-manager:pvc-v1.0.0

# Restart controller
kubectl rollout restart deployment/ome-controller-manager -n ome

# Verify deployment
kubectl rollout status deployment/ome-controller-manager -n ome
```

### Verify PVC Support

Test with a PVC-enabled InferenceService:

```bash
kubectl apply -f - <<EOF
apiVersion: ome.io/v1beta1
kind: InferenceService
metadata:
  name: test-pvc
  namespace: autotuner
  annotations:
    ome.io/storage-type: "pvc"
    ome.io/pvc-name: "model-storage-pvc"
    ome.io/pvc-subpath: "meta/llama-3-2-1b-instruct"
    ome.io/mount-path: "/raid/models/meta/llama-3-2-1b-instruct"
spec:
  model:
    name: llama-3-2-1b-instruct
  runtime:
    name: sglang-llama-small
    apiGroup: ome.io
    kind: ClusterServingRuntime
  engine:
    minReplicas: 1
    maxReplicas: 1
EOF

# Check if PVC is mounted
kubectl get deployment test-pvc-engine -n autotuner -o yaml | grep -A 5 persistentVolumeClaim
```

## Documentation

- **Full Guide**: [`docs/GITHUB_ACTIONS_BUILD.md`](../docs/GITHUB_ACTIONS_BUILD.md)
- **Verification Results**: [`docs/PVC_VERIFICATION_RESULTS.md`](../docs/PVC_VERIFICATION_RESULTS.md)
- **Build Guide**: [`docs/PVC_BUILD_AND_DEPLOY.md`](../docs/PVC_BUILD_AND_DEPLOY.md)

## Troubleshooting

**Build fails with permission error?**
- Go to Settings → Actions → General → Workflow permissions
- Select "Read and write permissions"
- Save and retry

**Need to use a different registry?**
- Update the `registry` parameter when running the workflow
- For non-GitHub registries, you may need to add authentication secrets

**Build takes too long?**
- First build: 10-15 minutes (building dependencies)
- Subsequent builds: 3-5 minutes (using cache)

## What Gets Built

The workflow:
1. ✅ Installs Go 1.24 and Rust/Cargo
2. ✅ Builds the XET library (Rust component)
3. ✅ Builds the OME manager binary with PVC support
4. ✅ Creates Docker image using distroless base
5. ✅ Pushes to container registry
6. ✅ Uses GitHub Actions cache for faster subsequent builds

## Next Steps

After successful deployment:
1. Test with autotuner: `python src/run_autotuner.py examples/pvc_task.json --mode ome --direct`
2. Verify backward compatibility with hostPath tasks
3. Update production deployments
