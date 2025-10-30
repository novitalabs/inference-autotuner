"""
System preset seeding for parameter presets.
"""

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from web.db.models import ParameterPreset


SYSTEM_PRESETS = [
	{
		"name": "Memory Efficient",
		"description": "Optimized for low memory usage, suitable for small GPUs",
		"category": "memory",
		"is_system": True,
		"parameters": {
			"tp-size": [1],
			"mem-fraction-static": [0.7, 0.75],
			"enable-chunked-prefill": [True]
		},
		"metadata": {
			"author": "system",
			"tags": ["memory", "small-gpu", "conservative"],
			"recommended_for": ["small-models", "limited-vram"]
		}
	},
	{
		"name": "High Throughput",
		"description": "Maximize tokens per second for maximum performance",
		"category": "performance",
		"is_system": True,
		"parameters": {
			"tp-size": [2, 4],
			"mem-fraction-static": [0.9],
			"schedule-policy": ["fcfs"],
			"enable-mixed-chunking": [True]
		},
		"metadata": {
			"author": "system",
			"tags": ["throughput", "performance", "production"],
			"recommended_for": ["large-models", "batch-processing"]
		}
	},
	{
		"name": "Low Latency",
		"description": "Minimize end-to-end latency for interactive applications",
		"category": "performance",
		"is_system": True,
		"parameters": {
			"tp-size": [1, 2],
			"schedule-policy": ["lpm"],
			"mem-fraction-static": [0.85]
		},
		"metadata": {
			"author": "system",
			"tags": ["latency", "interactive", "real-time"],
			"recommended_for": ["chatbots", "interactive-apps"]
		}
	},
	{
		"name": "Balanced",
		"description": "Balanced configuration for general-purpose use",
		"category": "general",
		"is_system": True,
		"parameters": {
			"tp-size": [1, 2],
			"mem-fraction-static": [0.85],
			"schedule-policy": ["fcfs", "lpm"]
		},
		"metadata": {
			"author": "system",
			"tags": ["balanced", "recommended", "general"],
			"recommended_for": ["general-use", "experimentation"]
		}
	},
]


async def seed_system_presets(db: AsyncSession):
	"""Seed database with system presets if they don't exist."""
	for preset_data in SYSTEM_PRESETS:
		# Check if preset already exists
		result = await db.execute(
			select(ParameterPreset).where(ParameterPreset.name == preset_data["name"])
		)
		existing = result.scalar_one_or_none()

		if not existing:
			# Create preset with correct field name
			preset_dict = preset_data.copy()
			preset_dict["preset_metadata"] = preset_dict.pop("metadata")
			preset = ParameterPreset(**preset_dict)
			db.add(preset)
			print(f"  ✅ Seeded system preset: {preset_data['name']}")
		else:
			print(f"  ⏭️  System preset already exists: {preset_data['name']}")

	await db.commit()
	print(f"✅ System presets seeding complete ({len(SYSTEM_PRESETS)} presets)")
