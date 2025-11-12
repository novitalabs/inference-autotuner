import { describe, it, expect } from 'vitest';
import { getAllRuntimeArgCombinations, QUANTIZATION_PRESETS } from './quantizationMapper';

describe('getAllRuntimeArgCombinations', () => {
	describe('Preset mode', () => {
		it('should return single combination for single preset', () => {
			const result = getAllRuntimeArgCombinations('vllm', { presets: ['default'] });

			expect(result.total).toBe(1);
			expect(result.truncated).toBe(false);
			expect(result.combinations.length).toBe(1);
		});

		it('should return correct vLLM args for kv-cache-fp8 preset', () => {
			const result = getAllRuntimeArgCombinations('vllm', { presets: ['kv-cache-fp8'] });

			expect(result.combinations[0]).toEqual({
				'--kv-cache-dtype': 'fp8_e5m2'
			});
		});

		it('should return correct vLLM args for dynamic-fp8 preset', () => {
			const result = getAllRuntimeArgCombinations('vllm', { presets: ['dynamic-fp8'] });

			expect(result.combinations[0]).toEqual({
				'--quantization': 'fp8',
				'--dtype': 'auto',
				'--kv-cache-dtype': 'fp8_e5m2'
			});
		});

		it('should return correct SGLang args for dynamic-fp8 preset', () => {
			const result = getAllRuntimeArgCombinations('sglang', { presets: ['dynamic-fp8'] });

			expect(result.combinations[0]).toEqual({
				'--quantization': 'fp8',
				'--dtype': 'auto',
				'--kv-cache-dtype': 'fp8_e5m2',
				'--attention-backend': 'flashinfer'
			});
		});

		it('should return correct SGLang args for aggressive-moe preset', () => {
			const result = getAllRuntimeArgCombinations('sglang', { presets: ['aggressive-moe'] });

			expect(result.combinations[0]).toEqual({
				'--quantization': 'w4afp8',
				'--moe-runner-backend': 'flashinfer_cutlass',
				'--kv-cache-dtype': 'fp8_e5m2',
				'--attention-backend': 'flashinfer'
			});
		});

		it('should handle multiple presets', () => {
			const result = getAllRuntimeArgCombinations('vllm', {
				presets: ['default', 'kv-cache-fp8', 'dynamic-fp8']
			});

			expect(result.total).toBe(3);
			expect(result.truncated).toBe(false);
			expect(result.combinations.length).toBe(3);
		});

		it('should deduplicate identical preset combinations', () => {
			const result = getAllRuntimeArgCombinations('vllm', {
				presets: ['default', 'default', 'kv-cache-fp8']
			});

			// default preset appears twice but should be deduplicated
			expect(result.total).toBe(2);
			expect(result.combinations.length).toBe(2);
		});
	});

	describe('Custom mode - single values', () => {
		it('should return single combination with single values', () => {
			const result = getAllRuntimeArgCombinations('vllm', {
				gemm_dtype: 'fp8',
				kvcache_dtype: 'fp8_e5m2',
				attention_dtype: 'auto',
				moe_dtype: 'auto'
			});

			expect(result.total).toBe(1);
			expect(result.truncated).toBe(false);
			expect(result.combinations[0]).toEqual({
				'--quantization': 'fp8',
				'--dtype': 'auto',
				'--kv-cache-dtype': 'fp8_e5m2'
			});
		});

		it('should handle auto values correctly', () => {
			const result = getAllRuntimeArgCombinations('vllm', {
				gemm_dtype: 'auto',
				kvcache_dtype: 'auto',
				attention_dtype: 'auto',
				moe_dtype: 'auto'
			});

			expect(result.total).toBe(1);
			expect(result.combinations[0]).toEqual({});
		});

		it('should handle bfloat16 gemm_dtype', () => {
			const result = getAllRuntimeArgCombinations('vllm', {
				gemm_dtype: 'bfloat16',
				kvcache_dtype: 'auto',
				attention_dtype: 'auto',
				moe_dtype: 'auto'
			});

			expect(result.combinations[0]).toEqual({
				'--dtype': 'bfloat16'
			});
		});
	});

	describe('Custom mode - multiple values (combinations)', () => {
		it('should generate all combinations for 2x2 values', () => {
			const result = getAllRuntimeArgCombinations('vllm', {
				gemm_dtype: ['auto', 'bfloat16'],
				kvcache_dtype: ['auto', 'fp8_e5m2'],
				attention_dtype: 'auto',
				moe_dtype: 'auto'
			});

			// 2 * 2 * 1 * 1 = 4 combinations
			expect(result.total).toBe(4);
			expect(result.truncated).toBe(false);
			expect(result.combinations.length).toBe(4);
		});

		it('should generate correct combinations for 2x2x2 values', () => {
			const result = getAllRuntimeArgCombinations('vllm', {
				gemm_dtype: ['auto', 'fp8'],
				kvcache_dtype: ['auto', 'fp8_e5m2'],
				attention_dtype: ['auto', 'fp8'],
				moe_dtype: 'auto'
			});

			// 2 * 2 * 2 * 1 = 8 total combinations before deduplication
			// But vLLM doesn't use attention_dtype, so we get:
			// - auto/auto/auto -> {}
			// - auto/auto/fp8 -> {} (duplicate)
			// - auto/fp8_e5m2/auto -> {--kv-cache-dtype: fp8_e5m2}
			// - auto/fp8_e5m2/fp8 -> {--kv-cache-dtype: fp8_e5m2} (duplicate)
			// - fp8/auto/auto -> {--quantization: fp8, --dtype: auto}
			// - fp8/auto/fp8 -> {--quantization: fp8, --dtype: auto} (duplicate)
			// - fp8/fp8_e5m2/auto -> {--quantization: fp8, --dtype: auto, --kv-cache-dtype: fp8_e5m2}
			// - fp8/fp8_e5m2/fp8 -> {--quantization: fp8, --dtype: auto, --kv-cache-dtype: fp8_e5m2} (duplicate)
			// Result: 4 unique combinations
			expect(result.total).toBe(4);
			expect(result.combinations.length).toBe(4);
		});

		it('should deduplicate identical custom combinations', () => {
			// For vLLM, attention_dtype doesn't affect the output
			// So different attention values should produce duplicate results
			const result = getAllRuntimeArgCombinations('vllm', {
				gemm_dtype: 'auto',
				kvcache_dtype: 'auto',
				attention_dtype: ['auto', 'fp8', 'bfloat16'],
				moe_dtype: 'auto'
			});

			// All three attention_dtype values produce the same empty args
			expect(result.total).toBe(1);
			expect(result.combinations.length).toBe(1);
			expect(result.combinations[0]).toEqual({});
		});

		it('should deduplicate when some combinations produce same args', () => {
			const result = getAllRuntimeArgCombinations('vllm', {
				gemm_dtype: ['auto', 'bfloat16'],
				kvcache_dtype: 'auto',
				attention_dtype: ['auto', 'fp8'], // vLLM ignores attention_dtype
				moe_dtype: 'auto'
			});

			// 2 gemm * 1 kvcache * 2 attention * 1 moe = 4 total combinations
			// But vLLM doesn't use attention_dtype, so we get:
			// - auto/auto -> {}
			// - auto/fp8 -> {} (duplicate)
			// - bfloat16/auto -> {--dtype: bfloat16}
			// - bfloat16/fp8 -> {--dtype: bfloat16} (duplicate)
			// Result: 2 unique combinations
			expect(result.total).toBe(2);
		});
	});

	describe('Truncation', () => {
		it('should truncate when combinations exceed maxCombinations', () => {
			const result = getAllRuntimeArgCombinations('sglang', {
				gemm_dtype: ['auto', 'fp8', 'bfloat16'],
				kvcache_dtype: ['auto', 'fp8_e5m2', 'fp8_e4m3', 'int8'],
				attention_dtype: 'auto',
				moe_dtype: 'auto'
			}, 10);

			// 3 * 4 * 1 * 1 = 12 combinations
			expect(result.total).toBe(12);
			expect(result.truncated).toBe(true);
			expect(result.combinations.length).toBe(10); // Limited to 10
		});

		it('should respect custom maxCombinations parameter', () => {
			const result = getAllRuntimeArgCombinations('vllm', {
				gemm_dtype: ['auto', 'fp8', 'bfloat16'],
				kvcache_dtype: ['auto', 'fp8_e5m2'],
				attention_dtype: 'auto',
				moe_dtype: 'auto'
			}, 3);

			// 3 * 2 * 1 * 1 = 6 combinations
			expect(result.total).toBe(6);
			expect(result.truncated).toBe(true);
			expect(result.combinations.length).toBe(3); // Limited to 3
		});
	});

	describe('Different runtimes', () => {
		it('should handle TensorRT-LLM runtime', () => {
			const result = getAllRuntimeArgCombinations('tensorrt-llm', {
				gemm_dtype: 'fp8',
				kvcache_dtype: 'fp8_e5m2',
				attention_dtype: 'fp8',
				moe_dtype: 'auto'
			});

			expect(result.combinations[0]).toEqual({
				'--quant-algo': 'FP8',
				'--kv-cache-quant-algo': 'FP8',
				'--fmha-quant-algo': 'FP8'
			});
		});

		it('should handle TensorRT-LLM with underscore', () => {
			const result = getAllRuntimeArgCombinations('tensorrt_llm', {
				gemm_dtype: 'int8',
				kvcache_dtype: 'int8',
				attention_dtype: 'auto',
				moe_dtype: 'auto'
			});

			expect(result.combinations[0]).toEqual({
				'--quant-algo': 'W8A8_SQ_PER_CHANNEL',
				'--kv-cache-quant-algo': 'INT8'
			});
		});

		it('should handle unknown runtime gracefully', () => {
			const result = getAllRuntimeArgCombinations('unknown-runtime', {
				gemm_dtype: 'fp8',
				kvcache_dtype: 'fp8_e5m2',
				attention_dtype: 'auto',
				moe_dtype: 'auto'
			});

			expect(result.total).toBe(1);
			expect(result.combinations[0]).toEqual({});
		});
	});

	describe('Edge cases', () => {
		it('should handle empty config', () => {
			const result = getAllRuntimeArgCombinations('vllm', {});

			expect(result.total).toBe(1);
			expect(result.truncated).toBe(false);
			expect(result.combinations[0]).toEqual({});
		});

		it('should handle empty presets array', () => {
			const result = getAllRuntimeArgCombinations('vllm', { presets: [] });

			expect(result.total).toBe(1);
			expect(result.combinations[0]).toEqual({});
		});

		it('should handle empty arrays in custom mode', () => {
			// When all fields are unchecked (empty arrays), should default to 'auto'
			const result = getAllRuntimeArgCombinations('vllm', {
				gemm_dtype: [],
				kvcache_dtype: [],
				attention_dtype: [],
				moe_dtype: []
			});

			expect(result.total).toBe(1);
			expect(result.combinations[0]).toEqual({});
		});

		it('should handle partial empty arrays in custom mode', () => {
			// When some fields have values and others are empty arrays
			const result = getAllRuntimeArgCombinations('vllm', {
				gemm_dtype: [],  // Empty - should default to 'auto'
				kvcache_dtype: ['fp8_e5m2'],  // Has value
				attention_dtype: [],  // Empty - should default to 'auto'
				moe_dtype: []  // Empty - should default to 'auto'
			});

			// 1 * 1 * 1 * 1 = 1 combination
			expect(result.total).toBe(1);
			expect(result.combinations[0]).toEqual({
				'--kv-cache-dtype': 'fp8_e5m2'
			});
		});

		it('should handle maxCombinations = 0', () => {
			const result = getAllRuntimeArgCombinations('vllm', {
				gemm_dtype: ['auto', 'fp8'],
				kvcache_dtype: 'auto',
				attention_dtype: 'auto',
				moe_dtype: 'auto'
			}, 0);

			expect(result.total).toBe(2);
			expect(result.truncated).toBe(true);
			expect(result.combinations.length).toBe(0);
		});

		it('should handle single-element arrays as arrays', () => {
			const result = getAllRuntimeArgCombinations('vllm', {
				gemm_dtype: ['fp8'],
				kvcache_dtype: ['fp8_e5m2'],
				attention_dtype: ['auto'],
				moe_dtype: ['auto']
			});

			expect(result.total).toBe(1);
			expect(result.combinations[0]).toEqual({
				'--quantization': 'fp8',
				'--dtype': 'auto',
				'--kv-cache-dtype': 'fp8_e5m2'
			});
		});
	});

	describe('SGLang-specific features', () => {
		it('should set attention-backend for FP8 attention', () => {
			const result = getAllRuntimeArgCombinations('sglang', {
				gemm_dtype: 'auto',
				kvcache_dtype: 'auto',
				attention_dtype: 'fp8',
				moe_dtype: 'auto'
			});

			expect(result.combinations[0]).toHaveProperty('--attention-backend', 'flashinfer');
		});

		it('should handle w4afp8 MoE quantization', () => {
			const result = getAllRuntimeArgCombinations('sglang', {
				gemm_dtype: 'auto',
				kvcache_dtype: 'auto',
				attention_dtype: 'auto',
				moe_dtype: 'w4afp8'
			});

			expect(result.combinations[0]).toEqual({
				'--quantization': 'w4afp8',
				'--moe-runner-backend': 'flashinfer_cutlass'
			});
		});

		it('should handle mxfp4 MoE quantization', () => {
			const result = getAllRuntimeArgCombinations('sglang', {
				gemm_dtype: 'auto',
				kvcache_dtype: 'auto',
				attention_dtype: 'auto',
				moe_dtype: 'mxfp4'
			});

			expect(result.combinations[0]).toEqual({
				'--quantization': 'mxfp4',
				'--moe-runner-backend': 'flashinfer_mxfp4'
			});
		});

		it('should prioritize MoE quantization over GEMM', () => {
			const result = getAllRuntimeArgCombinations('sglang', {
				gemm_dtype: 'fp8',
				kvcache_dtype: 'auto',
				attention_dtype: 'auto',
				moe_dtype: 'w4afp8'
			});

			// MoE quantization should override GEMM
			expect(result.combinations[0]['--quantization']).toBe('w4afp8');
			expect(result.combinations[0]).not.toHaveProperty('--dtype');
		});
	});

	describe('Complex real-world scenarios', () => {
		it('should handle comprehensive tuning scenario', () => {
			const result = getAllRuntimeArgCombinations('sglang', {
				gemm_dtype: ['auto', 'fp8', 'bfloat16'],
				kvcache_dtype: ['auto', 'fp8_e5m2'],
				attention_dtype: ['auto', 'fp8'],
				moe_dtype: 'auto'
			}, 15);

			// 3 * 2 * 2 * 1 = 12 combinations (before deduplication)
			expect(result.total).toBeGreaterThan(0);
			expect(result.total).toBeLessThanOrEqual(12);
			expect(result.truncated).toBe(false);
			expect(result.combinations.length).toBe(result.total);
		});

		it('should deduplicate all presets scenario', () => {
			const allPresets = Object.keys(QUANTIZATION_PRESETS).filter(p => p !== 'none');
			const result = getAllRuntimeArgCombinations('sglang', {
				presets: allPresets
			});

			// Should have one combination per preset
			expect(result.total).toBe(allPresets.length);
			expect(result.combinations.length).toBe(allPresets.length);
		});
	});
});
