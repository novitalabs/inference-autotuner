# Building OME Controller with GitHub Actions

## Overview

Since Docker is not available in the local development environment, we can use GitHub Actions to build and push the OME controller Docker image with PVC support.

## Workflow File

**Location**: `.github/workflows/build-ome-pvc.yml`

This workflow:
- Builds the OME manager binary with all PVC changes
- Creates a Docker image using the existing `manager.Dockerfile`
- Pushes the image to a container registry (GitHub Container Registry by default)
- Supports both manual triggers and automatic builds on push

## Prerequisites

### 1. GitHub Repository Access

Ensure your code is pushed to a GitHub repository:

```bash
git remote -v  # Check current remote
# If needed, add remote:
# git remote add origin https://github.com/YOUR_USERNAME/inference-autotuner.git
```

### 2. Container Registry Authentication

The workflow uses `GITHUB_TOKEN` by default to push to GitHub Container Registry (ghcr.io). This token is automatically available in GitHub Actions.

**For private registries**, add the following secrets to your GitHub repository:
- Go to: Settings → Secrets and variables → Actions → New repository secret
- Add credentials for your registry

### 3. Enable GitHub Actions

Ensure GitHub Actions is enabled for your repository:
- Go to: Settings → Actions → General
- Under "Actions permissions", select "Allow all actions and reusable workflows"

## Usage

### Method 1: Manual Trigger (Recommended for Testing)

1. **Go to GitHub Actions tab** in your repository
2. **Select** "Build OME Controller with PVC Support" workflow
3. **Click** "Run workflow" button
4. **Configure** parameters:
   - **registry**: Container registry (default: `ghcr.io/moirai-internal`)
   - **tag**: Image tag (default: `pvc-support`)
5. **Click** "Run workflow"

**Example Manual Trigger:**
- Registry: `ghcr.io/your-username`
- Tag: `pvc-v1.0.0`
- Result: `ghcr.io/your-username/ome-manager:pvc-v1.0.0`

### Method 2: Automatic Build on Push

The workflow automatically triggers when:
- You push changes to the `feature/ome` branch
- Files in `third_party/ome/` are modified
- The workflow file itself is modified

```bash
# Push to trigger build
git add third_party/ome/
git commit -m "feat: add PVC storage support"
git push origin feature/ome
```

## Monitoring Build Progress

1. **Go to Actions tab** in your GitHub repository
2. **Click on** the running workflow
3. **Monitor** build steps:
   - Checkout code
   - Set up Docker Buildx
   - Log in to Container Registry
   - Build and push image (this takes ~5-10 minutes)

## Using the Built Image

Once the build completes, deploy to your Kubernetes cluster:

### Option 1: Update Deployment Image

```bash
# Replace with your actual image
IMAGE="ghcr.io/moirai-internal/ome-manager:pvc-support"

# Update deployment
kubectl set image deployment/ome-controller-manager \
  -n ome \
  manager=${IMAGE}

# Restart deployment
kubectl rollout restart deployment/ome-controller-manager -n ome

# Monitor rollout
kubectl rollout status deployment/ome-controller-manager -n ome
```

### Option 2: Edit Deployment Directly

```bash
kubectl edit deployment ome-controller-manager -n ome
```

Update the image field:
```yaml
spec:
  template:
    spec:
      containers:
      - name: manager
        image: ghcr.io/moirai-internal/ome-manager:pvc-support  # Change this
```

### Option 3: Test Locally First

```bash
# Pull image
docker pull ghcr.io/moirai-internal/ome-manager:pvc-support

# Run locally with kubeconfig
docker run --rm -it \
  -v ~/.kube/config:/config \
  ghcr.io/moirai-internal/ome-manager:pvc-support \
  --kubeconfig /config
```

## Verifying the Deployment

After deploying the new image:

```bash
# Check controller pods
kubectl get pods -n ome

# Check controller version
kubectl logs -n ome deployment/ome-controller-manager | grep GitVersion

# Expected output should show: v0.1.3-86-g2b36d3f-dirty (or similar)

# Test with PVC InferenceService
kubectl apply -f /path/to/test-pvc-isvc.yaml

# Check if PVC is mounted
kubectl get deployment test-pvc-manual-engine -n autotuner -o yaml | grep -A 5 "persistentVolumeClaim"
```

## Troubleshooting

### Build Fails: "No permission to push"

**Solution**: Ensure GitHub Actions has package write permissions
1. Go to: Settings → Actions → General → Workflow permissions
2. Select "Read and write permissions"
3. Save and re-run workflow

### Build Fails: "Cannot access registry"

**Solution**: For private registries, add authentication secrets
1. Settings → Secrets and variables → Actions
2. Add `REGISTRY_USERNAME` and `REGISTRY_PASSWORD` secrets
3. Update workflow to use these secrets in the login step

### Image Size Too Large

The manager image is ~94MB compressed due to:
- Go binary with CGO enabled (for XET library)
- Distroless base image (minimal)

This is expected and efficient.

### Build Takes Too Long

First build may take 10-15 minutes. Subsequent builds use caching and complete in 3-5 minutes.

## Customization

### Using Different Registry

**DockerHub:**
```yaml
registry: docker.io/your-username
tag: pvc-support
```

**AWS ECR:**
```yaml
registry: 123456789.dkr.ecr.us-west-2.amazonaws.com
tag: ome-manager-pvc
```

**Note**: For non-GitHub registries, update the `docker/login-action` step with proper credentials.

### Multi-Architecture Builds

To build for multiple architectures (amd64, arm64):

1. Update workflow platform line:
   ```yaml
   platforms: linux/amd64,linux/arm64
   ```

2. Build time will increase (~2x)

### Build Arguments

Customize build arguments in workflow:
```yaml
build-args: |
  VERSION=${{ steps.meta.outputs.git_tag }}
  GIT_TAG=${{ steps.meta.outputs.git_tag }}
  GIT_COMMIT=${{ steps.meta.outputs.git_commit }}
  HTTP_PROXY=http://your-proxy:1081  # Add if needed
```

## Alternative: Build Locally with GitHub Codespaces

If you have GitHub Codespaces access, you can build Docker images there:

```bash
# In GitHub Codespaces terminal
cd third_party/ome
make ome-image TAG=pvc-support
docker push ghcr.io/your-username/ome-manager:pvc-support
```

## Security Notes

- The workflow uses `GITHUB_TOKEN` which has limited scope (only your repo)
- Images are private by default in GitHub Container Registry
- To make public: Go to package settings → Change visibility

## Next Steps

After successful image build and deployment:

1. ✅ Verify PVC functionality with test InferenceService
2. ✅ Test autotuner integration with `examples/pvc_task.json`
3. ✅ Verify backward compatibility with hostPath tasks
4. ✅ Update production deployments

## References

- **Workflow File**: `.github/workflows/build-ome-pvc.yml`
- **Dockerfile**: `third_party/ome/dockerfiles/manager.Dockerfile`
- **Verification Results**: `docs/PVC_VERIFICATION_RESULTS.md`
- **Build Guide**: `docs/PVC_BUILD_AND_DEPLOY.md`
