// TypeScript port of quantization_mapper.py
// Maps quantization configuration to runtime-specific CLI arguments

import { QuantizationConfig } from '../types/api';

// Resolved config with only string types (arrays resolved to first value)
interface ResolvedQuantConfig {
	gemm_dtype: string;
	kvcache_dtype: string;
	attention_dtype: string;
	moe_dtype: string;
}

// Quantization presets matching backend implementation
export const QUANTIZATION_PRESETS: Record<string, Omit<QuantizationConfig, 'preset' | 'presets'>> = {
	'default': {
		gemm_dtype: 'auto',
		kvcache_dtype: 'auto',
		attention_dtype: 'auto',
		moe_dtype: 'auto'
	},
	'kv-cache-fp8': {
		gemm_dtype: 'auto',
		kvcache_dtype: 'fp8_e5m2',
		attention_dtype: 'auto',
		moe_dtype: 'auto'
	},
	'dynamic-fp8': {
		gemm_dtype: 'fp8',
		kvcache_dtype: 'fp8_e5m2',
		attention_dtype: 'fp8',
		moe_dtype: 'auto'
	},
	'bf16-stable': {
		gemm_dtype: 'bfloat16',
		kvcache_dtype: 'fp8_e5m2',
		attention_dtype: 'auto',
		moe_dtype: 'auto'
	},
	'aggressive-moe': {
		gemm_dtype: 'auto',
		kvcache_dtype: 'fp8_e5m2',
		attention_dtype: 'fp8',
		moe_dtype: 'w4afp8'
	}
};

// Expand preset to full config
export function expandPreset(presetName: string): ResolvedQuantConfig {
	const preset = QUANTIZATION_PRESETS[presetName];
	if (!preset) {
		return {
			gemm_dtype: 'auto',
			kvcache_dtype: 'auto',
			attention_dtype: 'auto',
			moe_dtype: 'auto'
		};
	}

	// Convert string | string[] to string
	const resolveValue = (val: string | string[] | undefined): string => {
		if (!val) return 'auto';
		if (Array.isArray(val)) return val[0] || 'auto';
		return val;
	};

	return {
		gemm_dtype: resolveValue(preset.gemm_dtype),
		kvcache_dtype: resolveValue(preset.kvcache_dtype),
		attention_dtype: resolveValue(preset.attention_dtype),
		moe_dtype: resolveValue(preset.moe_dtype)
	};
}

// Resolve config to flat structure (takes first value if array)
export function resolveQuantConfig(config: QuantizationConfig): ResolvedQuantConfig {
	// Multi-preset mode - use first preset for display
	if (config.presets && config.presets.length > 0) {
		return expandPreset(config.presets[0]);
	}

	// Single preset mode
	if (config.preset) {
		return expandPreset(config.preset);
	}

	// Custom mode or empty - take first value if array
	const resolveField = (value: string | string[] | undefined): string => {
		if (!value) return 'auto';
		if (Array.isArray(value)) return value[0] || 'auto';
		return value;
	};

	return {
		gemm_dtype: resolveField(config.gemm_dtype),
		kvcache_dtype: resolveField(config.kvcache_dtype),
		attention_dtype: resolveField(config.attention_dtype),
		moe_dtype: resolveField(config.moe_dtype)
	};
}

// Map to vLLM arguments
export function mapToVllmArgs(config: ResolvedQuantConfig): Record<string, string> {
	const args: Record<string, string> = {};

	// GEMM dtype -> quantization method
	if (config.gemm_dtype === 'fp8') {
		args['--quantization'] = 'fp8';
		args['--dtype'] = 'auto';
	} else if (config.gemm_dtype === 'int8') {
		args['--quantization'] = 'int8';
		args['--dtype'] = 'auto';
	} else if (config.gemm_dtype && config.gemm_dtype !== 'auto') {
		args['--dtype'] = config.gemm_dtype;
	}

	// KV cache dtype
	if (config.kvcache_dtype && config.kvcache_dtype !== 'auto') {
		args['--kv-cache-dtype'] = config.kvcache_dtype;
	}

	// Note: vLLM doesn't support separate attention dtype
	// Note: vLLM has limited MoE dtype control

	return args;
}

// Map to SGLang arguments
export function mapToSglangArgs(config: ResolvedQuantConfig): Record<string, string> {
	const args: Record<string, string> = {};

	// MoE dtype can override GEMM quantization
	if (config.moe_dtype === 'w4afp8') {
		args['--quantization'] = 'w4afp8';
		args['--moe-runner-backend'] = 'flashinfer_cutlass';
	} else if (config.moe_dtype === 'mxfp4') {
		args['--quantization'] = 'mxfp4';
		args['--moe-runner-backend'] = 'flashinfer_mxfp4';
	} else {
		// GEMM dtype
		if (config.gemm_dtype === 'fp8') {
			args['--quantization'] = 'fp8';
			args['--dtype'] = 'auto';
		} else if (config.gemm_dtype === 'int8') {
			args['--quantization'] = 'int8';
			args['--dtype'] = 'auto';
		} else if (config.gemm_dtype && config.gemm_dtype !== 'auto') {
			args['--dtype'] = config.gemm_dtype;
		}
	}

	// KV cache dtype
	if (config.kvcache_dtype && config.kvcache_dtype !== 'auto') {
		args['--kv-cache-dtype'] = config.kvcache_dtype;
	}

	// Attention backend for FP8
	if (config.attention_dtype && ['fp8', 'fp8_e5m2', 'fp8_e4m3'].includes(config.attention_dtype)) {
		args['--attention-backend'] = 'flashinfer';
	}

	return args;
}

// Map to TensorRT-LLM arguments
export function mapToTensorrtLlmArgs(config: ResolvedQuantConfig): Record<string, string> {
	const args: Record<string, string> = {};

	// GEMM dtype
	if (config.gemm_dtype === 'fp8') {
		args['--quant-algo'] = 'FP8';
	} else if (config.gemm_dtype === 'int8') {
		args['--quant-algo'] = 'W8A8_SQ_PER_CHANNEL';
	} else if (config.gemm_dtype && config.gemm_dtype !== 'auto') {
		args['--dtype'] = config.gemm_dtype;
	}

	// KV cache dtype
	if (config.kvcache_dtype && config.kvcache_dtype !== 'auto') {
		if (config.kvcache_dtype.includes('fp8')) {
			args['--kv-cache-quant-algo'] = 'FP8';
		} else if (config.kvcache_dtype === 'int8') {
			args['--kv-cache-quant-algo'] = 'INT8';
		} else if (config.kvcache_dtype === 'int4') {
			args['--kv-cache-quant-algo'] = 'INT4';
		}
	}

	// Attention dtype (FMHA quantization)
	if (config.attention_dtype && ['fp8', 'fp8_e5m2', 'fp8_e4m3'].includes(config.attention_dtype)) {
		args['--fmha-quant-algo'] = 'FP8';
	} else if (config.attention_dtype === 'fp8_block') {
		args['--fmha-quant-algo'] = 'FP8_BLOCK';
	}

	// Note: TensorRT-LLM doesn't support separate MoE dtype

	return args;
}

// Get runtime arguments for specific engine
export function getRuntimeArgs(
	runtime: string,
	config: QuantizationConfig
): Record<string, string> {
	const resolvedConfig = resolveQuantConfig(config);

	const runtimeLower = runtime.toLowerCase();

	if (runtimeLower === 'vllm') {
		return mapToVllmArgs(resolvedConfig);
	} else if (runtimeLower === 'sglang') {
		return mapToSglangArgs(resolvedConfig);
	} else if (runtimeLower === 'tensorrt-llm' || runtimeLower === 'tensorrt_llm') {
		return mapToTensorrtLlmArgs(resolvedConfig);
	}

	return {};
}

// Generate all combinations from arrays
function generateCombinations(config: QuantizationConfig): ResolvedQuantConfig[] {
	// Get arrays for each field, defaulting to ['auto'] if empty or undefined
	const getValues = (field: string | string[] | undefined): string[] => {
		if (Array.isArray(field)) {
			return field.length > 0 ? field : ['auto'];
		}
		return [field || 'auto'];
	};

	const gemmValues = getValues(config.gemm_dtype);
	const kvcacheValues = getValues(config.kvcache_dtype);
	const attentionValues = getValues(config.attention_dtype);
	const moeValues = getValues(config.moe_dtype);

	const combinations: ResolvedQuantConfig[] = [];

	// Generate all combinations using nested loops
	for (const gemm of gemmValues) {
		for (const kvcache of kvcacheValues) {
			for (const attention of attentionValues) {
				for (const moe of moeValues) {
					combinations.push({
						gemm_dtype: gemm,
						kvcache_dtype: kvcache,
						attention_dtype: attention,
						moe_dtype: moe
					});
				}
			}
		}
	}

	return combinations;
}

// Get all runtime argument combinations for a config
export function getAllRuntimeArgCombinations(
	runtime: string,
	config: QuantizationConfig,
	maxCombinations: number = 10
): { combinations: Record<string, string>[]; total: number; truncated: boolean } {
	// Handle preset/multi-preset modes
	if (config.presets && config.presets.length > 0) {
		// For multi-preset, each preset is a separate combination
		const combinations = config.presets.map(preset => {
			const resolvedConfig = expandPreset(preset);
			return mapConfigToArgs(runtime, resolvedConfig);
		});

		// Deduplicate combinations
		const uniqueCombinations = deduplicateCombinations(combinations);
		const limitedCombinations = uniqueCombinations.slice(0, maxCombinations);

		return {
			combinations: limitedCombinations,
			total: uniqueCombinations.length,
			truncated: uniqueCombinations.length > maxCombinations
		};
	}

	if (config.preset) {
		// Single preset - just one combination
		const resolvedConfig = expandPreset(config.preset);
		return {
			combinations: [mapConfigToArgs(runtime, resolvedConfig)],
			total: 1,
			truncated: false
		};
	}

	// Custom mode - generate all combinations
	const allCombinations = generateCombinations(config);
	const argCombinations = allCombinations.map(resolved =>
		mapConfigToArgs(runtime, resolved)
	);

	// Deduplicate combinations
	const uniqueCombinations = deduplicateCombinations(argCombinations);
	const limitedCombinations = uniqueCombinations.slice(0, maxCombinations);

	return {
		combinations: limitedCombinations,
		total: uniqueCombinations.length,
		truncated: uniqueCombinations.length > maxCombinations
	};
}

// Deduplicate combinations based on their argument content
function deduplicateCombinations(combinations: Record<string, string>[]): Record<string, string>[] {
	const seen = new Set<string>();
	const unique: Record<string, string>[] = [];

	for (const combo of combinations) {
		// Create a canonical string representation by sorting keys
		const sortedEntries = Object.entries(combo).sort(([keyA], [keyB]) => keyA.localeCompare(keyB));
		const signature = JSON.stringify(sortedEntries);

		if (!seen.has(signature)) {
			seen.add(signature);
			unique.push(combo);
		}
	}

	return unique;
}

// Helper to map a resolved config to args based on runtime
function mapConfigToArgs(runtime: string, config: ResolvedQuantConfig): Record<string, string> {
	const runtimeLower = runtime.toLowerCase();

	if (runtimeLower === 'vllm') {
		return mapToVllmArgs(config);
	} else if (runtimeLower === 'sglang') {
		return mapToSglangArgs(config);
	} else if (runtimeLower === 'tensorrt-llm' || runtimeLower === 'tensorrt_llm') {
		return mapToTensorrtLlmArgs(config);
	}

	return {};
}

// Format arguments for display
export function formatArgsForDisplay(args: Record<string, string>): string {
	if (Object.keys(args).length === 0) {
		return 'No quantization arguments (using defaults)';
	}

	return Object.entries(args)
		.map(([key, value]) => `${key} ${value}`)
		.join(' ');
}

// Format multiple combinations for display
export function formatMultipleCombinationsForDisplay(
	combinations: Record<string, string>[],
	total: number,
	truncated: boolean
): string {
	if (combinations.length === 0) {
		return 'No quantization arguments (using defaults)';
	}

	if (combinations.length === 1) {
		return formatArgsForDisplay(combinations[0]);
	}

	const lines = combinations.map((args, index) => {
		const formattedArgs = formatArgsForDisplay(args);
		return `[${index + 1}] ${formattedArgs}`;
	});

	if (truncated) {
		lines.push(`... and ${total - combinations.length} more combinations`);
	}

	return lines.join('\n');
}
