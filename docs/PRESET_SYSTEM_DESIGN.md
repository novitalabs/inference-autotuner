# Parameter Preset System Design

## Overview

A preset system that allows users to save, manage, and apply reusable parameter configurations for inference engine tuning tasks. Supports creating presets, importing/exporting them, and intelligently merging multiple presets.

## Features

1. **Preset Management**: Create, edit, delete, and list parameter presets
2. **Import/Export**: Share presets via JSON files
3. **Multi-Preset Application**: Apply multiple presets to a task with automatic merging
4. **Merge Strategies**: Handle conflicts when applying multiple presets
5. **Preset Templates**: Built-in presets for common scenarios

## Database Schema

### New Table: `parameter_presets`

```sql
CREATE TABLE parameter_presets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name VARCHAR(255) NOT NULL UNIQUE,
    description TEXT,
    category VARCHAR(100),  -- e.g., "performance", "memory", "custom"
    is_system BOOLEAN DEFAULT FALSE,  -- System presets cannot be deleted
    parameters JSON NOT NULL,  -- The parameter configuration
    metadata JSON,  -- Additional metadata (author, tags, etc.)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_preset_category ON parameter_presets(category);
CREATE INDEX idx_preset_system ON parameter_presets(is_system);
```

### SQLAlchemy Model

```python
# src/web/db/models.py

from sqlalchemy import Column, Integer, String, Text, Boolean, JSON, DateTime
from sqlalchemy.sql import func
from .session import Base

class ParameterPreset(Base):
    __tablename__ = "parameter_presets"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False, unique=True, index=True)
    description = Column(Text)
    category = Column(String(100), index=True)
    is_system = Column(Boolean, default=False, index=True)
    parameters = Column(JSON, nullable=False)
    metadata = Column(JSON)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), server_default=func.now())

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "is_system": self.is_system,
            "parameters": self.parameters,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
```

## Parameter Structure

Presets store parameters in the same format as task configurations:

```json
{
  "name": "High Performance",
  "description": "Optimized for maximum throughput",
  "category": "performance",
  "parameters": {
    "tp-size": [2, 4],
    "mem-fraction-static": [0.9],
    "schedule-policy": ["fcfs"]
  },
  "metadata": {
    "author": "system",
    "tags": ["throughput", "gpu-heavy"],
    "recommended_for": ["large-models"]
  }
}
```

## API Endpoints

### Preset CRUD Operations

```python
# src/web/routes/presets.py

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List
import json

router = APIRouter(prefix="/api/presets", tags=["presets"])

# List all presets
@router.get("/")
async def list_presets(
    category: Optional[str] = None,
    db: AsyncSession = Depends(get_db)
):
    """List all parameter presets, optionally filtered by category."""
    pass

# Get preset by ID
@router.get("/{preset_id}")
async def get_preset(preset_id: int, db: AsyncSession = Depends(get_db)):
    """Get a specific preset by ID."""
    pass

# Create preset
@router.post("/")
async def create_preset(preset: PresetCreate, db: AsyncSession = Depends(get_db)):
    """Create a new parameter preset."""
    pass

# Update preset
@router.put("/{preset_id}")
async def update_preset(
    preset_id: int,
    preset: PresetUpdate,
    db: AsyncSession = Depends(get_db)
):
    """Update an existing preset. System presets cannot be modified."""
    pass

# Delete preset
@router.delete("/{preset_id}")
async def delete_preset(preset_id: int, db: AsyncSession = Depends(get_db)):
    """Delete a preset. System presets cannot be deleted."""
    pass

# Import preset from JSON
@router.post("/import")
async def import_preset(file: UploadFile = File(...), db: AsyncSession = Depends(get_db)):
    """Import a preset from a JSON file."""
    pass

# Export preset to JSON
@router.get("/{preset_id}/export")
async def export_preset(preset_id: int, db: AsyncSession = Depends(get_db)):
    """Export a preset as a JSON file."""
    pass

# Merge multiple presets
@router.post("/merge")
async def merge_presets(preset_ids: List[int], db: AsyncSession = Depends(get_db)):
    """Merge multiple presets and return the combined parameters."""
    pass
```

### Pydantic Schemas

```python
# src/web/schemas/preset.py

from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
from datetime import datetime

class PresetBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    category: Optional[str] = None
    parameters: Dict[str, Any] = Field(..., description="Parameter configuration")
    metadata: Optional[Dict[str, Any]] = None

class PresetCreate(PresetBase):
    pass

class PresetUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    category: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None

class PresetResponse(PresetBase):
    id: int
    is_system: bool
    created_at: datetime
    updated_at: Optional[datetime]

    class Config:
        from_attributes = True

class PresetMergeRequest(BaseModel):
    preset_ids: List[int] = Field(..., min_items=1)
    merge_strategy: str = Field(default="union", pattern="^(union|intersection|last_wins)$")

class PresetMergeResponse(BaseModel):
    parameters: Dict[str, Any]
    applied_presets: List[str]
    conflicts: Optional[List[Dict[str, Any]]] = None
```

## Parameter Merging Logic

### Merge Strategies

When applying multiple presets, the system supports different merge strategies:

#### 1. **Union (Default)** - Combine all values
```python
# Input:
# Preset A: {"tp-size": [1, 2], "mem-fraction-static": [0.8]}
# Preset B: {"tp-size": [4], "schedule-policy": ["lpm"]}

# Output:
{
  "tp-size": [1, 2, 4],  # Combined and deduplicated
  "mem-fraction-static": [0.8],
  "schedule-policy": ["lpm"]
}
```

#### 2. **Intersection** - Only keep common parameters
```python
# Input:
# Preset A: {"tp-size": [1, 2], "mem-fraction-static": [0.8]}
# Preset B: {"tp-size": [2, 4], "schedule-policy": ["lpm"]}

# Output:
{
  "tp-size": [2]  # Only values present in both presets
}
```

#### 3. **Last Wins** - Later presets override earlier ones
```python
# Input (applied in order A, B):
# Preset A: {"tp-size": [1, 2], "mem-fraction-static": [0.8]}
# Preset B: {"tp-size": [4], "schedule-policy": ["lpm"]}

# Output:
{
  "tp-size": [4],  # B overrides A
  "mem-fraction-static": [0.8],  # Only in A
  "schedule-policy": ["lpm"]  # Only in B
}
```

### Implementation

```python
# src/utils/preset_merger.py

from typing import Dict, List, Any, Tuple
from enum import Enum

class MergeStrategy(str, Enum):
    UNION = "union"
    INTERSECTION = "intersection"
    LAST_WINS = "last_wins"

class PresetMerger:
    """Handles merging of multiple parameter presets."""

    @staticmethod
    def merge_parameters(
        presets: List[Dict[str, Any]],
        strategy: MergeStrategy = MergeStrategy.UNION
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Merge multiple parameter presets.

        Returns:
            Tuple of (merged_parameters, conflicts)
        """
        if not presets:
            return {}, []

        if len(presets) == 1:
            return presets[0].get("parameters", {}), []

        if strategy == MergeStrategy.UNION:
            return PresetMerger._merge_union(presets)
        elif strategy == MergeStrategy.INTERSECTION:
            return PresetMerger._merge_intersection(presets)
        elif strategy == MergeStrategy.LAST_WINS:
            return PresetMerger._merge_last_wins(presets)

    @staticmethod
    def _merge_union(presets: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Union merge: combine all parameter values."""
        merged = {}
        conflicts = []

        for preset in presets:
            parameters = preset.get("parameters", {})
            for param_name, values in parameters.items():
                if param_name not in merged:
                    merged[param_name] = []

                # Convert single value to list
                if not isinstance(values, list):
                    values = [values]

                # Add new values (deduplicate)
                for value in values:
                    if value not in merged[param_name]:
                        merged[param_name].append(value)

        return merged, conflicts

    @staticmethod
    def _merge_intersection(presets: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Intersection merge: only keep values present in all presets."""
        if not presets:
            return {}, []

        # Get all parameter names from first preset
        first_params = presets[0].get("parameters", {})
        merged = {}
        conflicts = []

        for param_name, first_values in first_params.items():
            if not isinstance(first_values, list):
                first_values = [first_values]

            # Find intersection of values across all presets
            common_values = set(first_values)

            for preset in presets[1:]:
                preset_params = preset.get("parameters", {})
                if param_name in preset_params:
                    preset_values = preset_params[param_name]
                    if not isinstance(preset_values, list):
                        preset_values = [preset_values]
                    common_values = common_values.intersection(preset_values)
                else:
                    # Parameter not in this preset, intersection is empty
                    common_values = set()
                    break

            if common_values:
                merged[param_name] = list(common_values)
            elif param_name in first_params:
                # Track conflict: parameter exists but no common values
                conflicts.append({
                    "parameter": param_name,
                    "reason": "No common values across all presets"
                })

        return merged, conflicts

    @staticmethod
    def _merge_last_wins(presets: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Last wins merge: later presets override earlier ones."""
        merged = {}
        conflicts = []

        for preset in presets:
            parameters = preset.get("parameters", {})
            for param_name, values in parameters.items():
                if param_name in merged:
                    # Track conflict
                    conflicts.append({
                        "parameter": param_name,
                        "overridden_by": preset.get("name", "unknown"),
                        "previous_values": merged[param_name],
                        "new_values": values
                    })

                merged[param_name] = values if isinstance(values, list) else [values]

        return merged, conflicts

    @staticmethod
    def validate_parameters(parameters: Dict[str, Any]) -> List[str]:
        """
        Validate merged parameters.

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        # Check that all parameter values are lists
        for param_name, values in parameters.items():
            if not isinstance(values, list):
                errors.append(f"Parameter '{param_name}' must be a list")
            elif len(values) == 0:
                errors.append(f"Parameter '{param_name}' cannot be empty")

        return errors
```

## Frontend Components

### 1. Preset Management Page

**Location**: `frontend/src/pages/Presets.tsx`

**Features**:
- List all presets with category filtering
- Create/edit preset modal
- Delete confirmation
- Import/export buttons
- Search and sort functionality

```typescript
// frontend/src/pages/Presets.tsx

interface Preset {
  id: number;
  name: string;
  description?: string;
  category?: string;
  is_system: boolean;
  parameters: Record<string, any>;
  metadata?: Record<string, any>;
  created_at: string;
  updated_at?: string;
}

export default function Presets() {
  // State
  const [presets, setPresets] = useState<Preset[]>([]);
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null);
  const [isCreateModalOpen, setIsCreateModalOpen] = useState(false);
  const [editingPreset, setEditingPreset] = useState<Preset | null>(null);

  // API calls using React Query
  const { data: presetsData } = useQuery({
    queryKey: ['presets', selectedCategory],
    queryFn: () => fetchPresets(selectedCategory)
  });

  // UI: Table with columns: Name, Description, Category, Actions (Edit, Delete, Export)
  // UI: Create/Import buttons in header
  // UI: Category filter dropdown
}
```

### 2. Preset Selector Component

**Location**: `frontend/src/components/PresetSelector.tsx`

**Features**:
- Multi-select dropdown for presets
- Applied presets displayed as chips
- Merge strategy selector
- Preview merged parameters
- Conflict warnings

```typescript
// frontend/src/components/PresetSelector.tsx

interface PresetSelectorProps {
  selectedPresets: number[];
  onPresetsChange: (presetIds: number[]) => void;
  onParametersChange: (parameters: Record<string, any>) => void;
  mergeStrategy?: 'union' | 'intersection' | 'last_wins';
}

export default function PresetSelector({
  selectedPresets,
  onPresetsChange,
  onParametersChange,
  mergeStrategy = 'union'
}: PresetSelectorProps) {
  // State
  const [localStrategy, setLocalStrategy] = useState(mergeStrategy);
  const [mergedParams, setMergedParams] = useState<Record<string, any>>({});
  const [conflicts, setConflicts] = useState<any[]>([]);

  // Fetch all presets
  const { data: presets } = useQuery({
    queryKey: ['presets'],
    queryFn: fetchAllPresets
  });

  // Merge presets when selection changes
  const { data: mergeResult } = useQuery({
    queryKey: ['merge', selectedPresets, localStrategy],
    queryFn: () => mergePresets(selectedPresets, localStrategy),
    enabled: selectedPresets.length > 0
  });

  useEffect(() => {
    if (mergeResult) {
      setMergedParams(mergeResult.parameters);
      setConflicts(mergeResult.conflicts || []);
      onParametersChange(mergeResult.parameters);
    }
  }, [mergeResult]);

  // UI components...
}
```

### 3. Updated NewTask/EditTask Pages

**Location**: `frontend/src/pages/NewTask.tsx`

**Integration**:
- Add PresetSelector component above parameter form
- Allow toggling between "Use Presets" and "Manual Entry"
- Show merged parameters as starting point
- Allow fine-tuning after preset application

```typescript
// Addition to NewTask.tsx

export default function NewTask() {
  const [usePresets, setUsePresets] = useState(true);
  const [selectedPresets, setSelectedPresets] = useState<number[]>([]);
  const [parameters, setParameters] = useState<Record<string, any>>({});

  return (
    <div>
      {/* Toggle: Use Presets / Manual Entry */}
      <div className="mb-4">
        <button onClick={() => setUsePresets(!usePresets)}>
          {usePresets ? 'Switch to Manual Entry' : 'Use Presets'}
        </button>
      </div>

      {usePresets && (
        <PresetSelector
          selectedPresets={selectedPresets}
          onPresetsChange={setSelectedPresets}
          onParametersChange={setParameters}
        />
      )}

      {/* Parameter form (pre-filled with merged parameters) */}
      <ParameterForm
        initialValues={parameters}
        onChange={setParameters}
        editable={true}
      />
    </div>
  );
}
```

## Import/Export Format

### Export Format (JSON)

```json
{
  "version": "1.0",
  "preset": {
    "name": "High Performance SGLang",
    "description": "Optimized for high throughput with SGLang runtime",
    "category": "performance",
    "parameters": {
      "tp-size": [2, 4],
      "mem-fraction-static": [0.9],
      "schedule-policy": ["fcfs"],
      "enable-mixed-chunking": [true]
    },
    "metadata": {
      "author": "system",
      "tags": ["sglang", "throughput", "production"],
      "recommended_for": ["llama-3-70b", "llama-3-405b"],
      "runtime": "sglang"
    }
  }
}
```

### Import Validation

```python
# src/utils/preset_validator.py

from typing import Dict, Any, List
import jsonschema

PRESET_SCHEMA = {
    "type": "object",
    "required": ["version", "preset"],
    "properties": {
        "version": {"type": "string"},
        "preset": {
            "type": "object",
            "required": ["name", "parameters"],
            "properties": {
                "name": {"type": "string", "minLength": 1},
                "description": {"type": "string"},
                "category": {"type": "string"},
                "parameters": {"type": "object"},
                "metadata": {"type": "object"}
            }
        }
    }
}

def validate_preset_import(data: Dict[str, Any]) -> List[str]:
    """Validate imported preset data."""
    errors = []

    try:
        jsonschema.validate(data, PRESET_SCHEMA)
    except jsonschema.ValidationError as e:
        errors.append(f"Invalid preset format: {e.message}")

    # Additional validation
    parameters = data.get("preset", {}).get("parameters", {})
    for param_name, values in parameters.items():
        if not isinstance(values, list):
            errors.append(f"Parameter '{param_name}' must be a list")

    return errors
```

## System Presets

### Built-in Presets (Seeded on Installation)

```python
# src/web/db/seed_presets.py

SYSTEM_PRESETS = [
    {
        "name": "Memory Efficient",
        "description": "Optimized for low memory usage",
        "category": "memory",
        "is_system": True,
        "parameters": {
            "tp-size": [1],
            "mem-fraction-static": [0.7, 0.75],
            "enable-chunked-prefill": [True]
        },
        "metadata": {
            "author": "system",
            "tags": ["memory", "small-gpu"]
        }
    },
    {
        "name": "High Throughput",
        "description": "Maximize tokens per second",
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
            "tags": ["throughput", "performance"]
        }
    },
    {
        "name": "Low Latency",
        "description": "Minimize end-to-end latency",
        "category": "performance",
        "is_system": True,
        "parameters": {
            "tp-size": [1, 2],
            "schedule-policy": ["lpm"],
            "mem-fraction-static": [0.85]
        },
        "metadata": {
            "author": "system",
            "tags": ["latency", "interactive"]
        }
    },
    {
        "name": "Balanced",
        "description": "Balanced configuration for general use",
        "category": "general",
        "is_system": True,
        "parameters": {
            "tp-size": [1, 2],
            "mem-fraction-static": [0.85],
            "schedule-policy": ["fcfs", "lpm"]
        },
        "metadata": {
            "author": "system",
            "tags": ["balanced", "recommended"]
        }
    }
]

async def seed_system_presets(db: AsyncSession):
    """Seed database with system presets."""
    for preset_data in SYSTEM_PRESETS:
        preset = ParameterPreset(**preset_data)
        db.add(preset)
    await db.commit()
```

## UI/UX Flow

### Creating a Task with Presets

1. User navigates to "New Task" page
2. User clicks "Use Parameter Presets"
3. Preset selector dropdown appears with all available presets
4. User selects multiple presets (e.g., "High Throughput" + "Memory Efficient")
5. System shows merge strategy dropdown (default: Union)
6. Preview panel shows merged parameters
7. If conflicts exist, warning badge appears with details
8. User can switch to manual mode and fine-tune parameters
9. User proceeds to create task with merged parameters

### Managing Presets

1. User navigates to "Presets" page from sidebar
2. Table shows all presets with categories
3. User clicks "Create Preset" button
4. Modal opens with form:
   - Name (required)
   - Description
   - Category dropdown
   - Parameter JSON editor with syntax highlighting
   - Metadata (optional)
5. User saves preset
6. Preset appears in table and is immediately available for use

### Import/Export

**Export**:
1. User clicks "Export" button next to preset
2. Browser downloads JSON file named `preset-{name}.json`

**Import**:
1. User clicks "Import" button in header
2. File picker opens
3. User selects JSON file
4. System validates format
5. If valid, preset is added with confirmation toast
6. If invalid, error message shows specific issues

## Implementation Checklist

### Backend
- [ ] Create `ParameterPreset` SQLAlchemy model
- [ ] Create database migration for `parameter_presets` table
- [ ] Implement `PresetMerger` utility class
- [ ] Create preset Pydantic schemas
- [ ] Implement preset CRUD routes
- [ ] Implement merge endpoint
- [ ] Implement import/export endpoints
- [ ] Add preset validator
- [ ] Create system preset seeder
- [ ] Add unit tests for merge logic

### Frontend
- [ ] Create `Presets.tsx` page
- [ ] Create `PresetSelector.tsx` component
- [ ] Create preset API service methods
- [ ] Update `NewTask.tsx` with preset integration
- [ ] Update `Tasks.tsx` for editing with presets
- [ ] Add preset import/export UI
- [ ] Add merge strategy selector
- [ ] Add conflict warning UI
- [ ] Update routing to include presets page
- [ ] Add TypeScript interfaces for presets

### Testing
- [ ] Unit tests for merge strategies
- [ ] Integration tests for preset CRUD
- [ ] E2E tests for preset application flow
- [ ] Test import/export functionality
- [ ] Test conflict detection and resolution

### Documentation
- [ ] Update CLAUDE.md with preset system info
- [ ] Add user guide for presets
- [ ] Document preset JSON format
- [ ] Add examples of common presets

## Future Enhancements

1. **Preset Versioning**: Track changes to presets over time
2. **Preset Sharing**: Share presets with other users/teams
3. **Preset Recommendations**: Suggest presets based on model/hardware
4. **Preset Analytics**: Track which presets perform best
5. **Preset Templates**: Pre-filled forms for common parameter types
6. **Preset Validation Rules**: Define constraints for parameter combinations
7. **Preset Dependencies**: Presets that require other presets
8. **Smart Merge**: AI-assisted conflict resolution
