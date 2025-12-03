/// <reference types="vite/client" />

interface ImportMetaEnv {
	readonly VITE_API_URL: string;
	readonly VITE_GITHUB_REPO: string;
}

interface ImportMeta {
	readonly env: ImportMetaEnv;
}

// Global constants injected by Vite at build time
declare const __BUILD_TIME__: string;
