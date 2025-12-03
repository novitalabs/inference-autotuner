/**
 * Version checking service for auto-update functionality
 *
 * This service provides:
 * - Current version fetching from backend API
 * - Latest release checking from GitHub API
 * - Semantic version comparison
 * - Update availability detection
 */

export interface VersionInfo {
  version: string;
}

export interface UpdateInfo {
  hasUpdate: boolean;
  currentVersion: string;
  latestVersion?: string;
  releaseNotes?: string;
  downloadUrl?: string;
  publishedAt?: string;
}

interface GitHubRelease {
  tag_name: string;
  name: string;
  body: string;
  html_url: string;
  published_at: string;
}

/**
 * Get current version from backend API
 */
export async function getCurrentVersion(): Promise<VersionInfo> {
  const response = await fetch('/api/system/info');
  if (!response.ok) {
    throw new Error('Failed to fetch version info');
  }
  return response.json();
}

/**
 * Get latest release from GitHub API
 */
async function getLatestRelease(githubRepo: string): Promise<GitHubRelease> {
  const url = `https://api.github.com/repos/${githubRepo}/releases/latest`;
  const response = await fetch(url);

  if (!response.ok) {
    throw new Error(`GitHub API returned ${response.status}`);
  }

  return response.json();
}

/**
 * Compare two semantic versions
 *
 * @param v1 - First version (e.g., "0.1.0" or "v0.1.0")
 * @param v2 - Second version (e.g., "0.2.0" or "v0.2.0")
 * @returns 1 if v1 > v2, -1 if v1 < v2, 0 if equal
 */
function compareVersions(v1: string, v2: string): number {
  // Remove 'v' prefix if present
  const v1Clean = v1.replace(/^v/, '');
  const v2Clean = v2.replace(/^v/, '');

  const v1Parts = v1Clean.split('.').map(Number);
  const v2Parts = v2Clean.split('.').map(Number);

  const maxLength = Math.max(v1Parts.length, v2Parts.length);

  for (let i = 0; i < maxLength; i++) {
    const part1 = v1Parts[i] || 0;
    const part2 = v2Parts[i] || 0;

    if (part1 > part2) return 1;
    if (part1 < part2) return -1;
  }

  return 0;
}

/**
 * Check if a new version is available
 *
 * @param githubRepo - GitHub repository in format "owner/repo"
 * @returns Update information including availability status
 */
export async function checkForUpdates(githubRepo: string): Promise<UpdateInfo> {
  try {
    // Get current version from backend
    const versionInfo = await getCurrentVersion();
    const currentVersion = versionInfo.version;

    // Get latest release from GitHub
    const latestRelease = await getLatestRelease(githubRepo);
    const latestVersion = latestRelease.tag_name.replace(/^v/, '');

    // Compare versions
    const comparison = compareVersions(latestVersion, currentVersion);

    if (comparison > 0) {
      // New version available
      return {
        hasUpdate: true,
        currentVersion,
        latestVersion,
        releaseNotes: latestRelease.body || 'No release notes available',
        downloadUrl: latestRelease.html_url,
        publishedAt: latestRelease.published_at
      };
    } else {
      // Already on latest version (or ahead)
      return {
        hasUpdate: false,
        currentVersion
      };
    }
  } catch (error) {
    // On any error, return no update available (graceful degradation)
    console.error('Error checking for updates:', error);
    return {
      hasUpdate: false,
      currentVersion: 'unknown'
    };
  }
}
