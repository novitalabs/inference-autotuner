# Parameter Preset System - Architecture Diagram

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              FRONTEND (React)                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐          │
│  │ Presets Page     │  │ NewTask Page     │  │ EditTask Page    │          │
│  │                  │  │                  │  │                  │          │
│  │ • List presets   │  │ • Preset toggle  │  │ • Load existing  │          │
│  │ • Create/Edit    │  │ • PresetSelector │  │ • Modify params  │          │
│  │ • Import/Export  │  │ • Param form     │  │ • Update task    │          │
│  └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘          │
│           │                     │                     │                     │
│           └─────────────────────┼─────────────────────┘                     │
│                                 │                                           │
│                    ┌────────────▼────────────┐                              │
│                    │  PresetSelector.tsx     │                              │
│                    │                         │                              │
│                    │  • Multi-select         │                              │
│                    │  • Merge strategy       │                              │
│                    │  • Live preview         │                              │
│                    │  • Conflict warnings    │                              │
│                    └────────────┬────────────┘                              │
│                                 │                                           │
│                    ┌────────────▼────────────┐                              │
│                    │  presetService.ts       │                              │
│                    │                         │                              │
│                    │  • API calls (axios)    │                              │
│                    │  • React Query hooks    │                              │
│                    │  • Type-safe responses  │                              │
│                    └────────────┬────────────┘                              │
└─────────────────────────────────┼──────────────────────────────────────────┘
                                  │
                                  │ HTTP REST API
                                  │
┌─────────────────────────────────▼──────────────────────────────────────────┐
│                            BACKEND (FastAPI)                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      API Routes (/api/presets/)                      │   │
│  │                                                                       │   │
│  │  GET    /                 List all presets (with filtering)          │   │
│  │  GET    /{id}             Get preset by ID                           │   │
│  │  POST   /                 Create new preset                          │   │
│  │  PUT    /{id}             Update preset                              │   │
│  │  DELETE /{id}             Delete preset                              │   │
│  │  POST   /import           Import from JSON file                      │   │
│  │  GET    /{id}/export      Export to JSON file                        │   │
│  │  POST   /merge            Merge multiple presets ★                   │   │
│  └───────────────────────────────┬─────────────────────────────────────┘   │
│                                  │                                          │
│                    ┌─────────────▼──────────────┐                          │
│                    │  PresetMerger (utils)      │                          │
│                    │                            │                          │
│                    │  • Union strategy          │                          │
│                    │  • Intersection strategy   │                          │
│                    │  • Last wins strategy      │                          │
│                    │  • Conflict detection      │                          │
│                    │  • Parameter validation    │                          │
│                    └─────────────┬──────────────┘                          │
│                                  │                                          │
│                    ┌─────────────▼──────────────┐                          │
│                    │  Database (SQLite)         │                          │
│                    │                            │                          │
│                    │  parameter_presets table   │                          │
│                    │  • id, name, description   │                          │
│                    │  • category, is_system     │                          │
│                    │  • parameters (JSON)       │                          │
│                    │  • metadata (JSON)         │                          │
│                    └────────────────────────────┘                          │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Data Flow Diagrams

### 1. Applying Multiple Presets

```
User Action                   Frontend                    Backend                     Database
──────────                   ────────                    ───────                     ────────

Select Preset 1
     │
     ├─────────────────────> Update selection state
     │
Select Preset 2
     │
     ├─────────────────────> Update selection state
     │                               │
Choose "Union"                       │
     │                               │
     ├─────────────────────> POST /api/presets/merge
     │                        preset_ids=[1, 2]    ─────> Fetch preset 1 ────────> SELECT * WHERE id=1
     │                        strategy="union"     ─────> Fetch preset 2 ────────> SELECT * WHERE id=2
     │                                                    │
     │                                                    PresetMerger.merge()
     │                                                    • Combine parameters
     │                                                    • Deduplicate values
     │                                                    • Detect conflicts
     │                                                    │
     │                        <────────────────────────── Return merged result
     │                        {
     │                          "parameters": {...},
     │                          "conflicts": [...]
     │                        }
     │
     ├<─────────────────────> Display merged params
     │                        Show conflicts (if any)
     │
Apply to form
     │
     └─────────────────────> Pre-fill parameter form
```

### 2. Creating a Task with Presets

```
User Journey:

1. Navigate to "New Task" page
2. Enable "Use Parameter Presets" toggle
   └─> PresetSelector component appears

3. Select "High Throughput" preset
   └─> API: GET /api/presets/
   └─> API: POST /api/presets/merge (preset_ids=[3])
   └─> Preview shows: {"tp-size": [2,4], "mem-fraction-static": [0.9], ...}

4. Add "Memory Efficient" preset
   └─> API: POST /api/presets/merge (preset_ids=[3,1], strategy="union")
   └─> Preview updates: {"tp-size": [1,2,4], "mem-fraction-static": [0.7,0.75,0.9], ...}
   └─> ⚠️ Warning: Parameter conflict detected (different memory fractions)

5. Choose merge strategy: "Last Wins"
   └─> API: POST /api/presets/merge (preset_ids=[3,1], strategy="last_wins")
   └─> Preview updates: {"tp-size": [1], "mem-fraction-static": [0.7,0.75], ...}
   └─> "Memory Efficient" values override "High Throughput"

6. Fine-tune parameters manually (optional)
   └─> Edit parameter form with merged values as starting point

7. Fill in other task fields (model, runtime, benchmark config)

8. Submit task
   └─> API: POST /api/tasks/
   └─> Task created with merged parameters
```

### 3. Import/Export Flow

```
EXPORT:
User clicks "Export" on preset
     │
     ├──────────────> GET /api/presets/{id}/export
     │                        │
     │                        Fetch preset from DB
     │                        │
     │                        Generate JSON:
     │                        {
     │                          "version": "1.0",
     │                          "preset": {
     │                            "name": "...",
     │                            "parameters": {...}
     │                          }
     │                        }
     │                        │
     │                        Return as file download
     │                        Content-Type: application/json
     │                        Content-Disposition: attachment;
     │                                             filename="preset-high-perf.json"
     │<───────────────
     └─> Browser downloads file


IMPORT:
User clicks "Import" → selects JSON file
     │
     ├──────────────> POST /api/presets/import
     │                (multipart/form-data)
     │                        │
     │                        Read file contents
     │                        │
     │                        Validate JSON schema:
     │                        • Check "version" field
     │                        • Check "preset" structure
     │                        • Validate parameters
     │                        │
     │                        ├─ VALID ──────────> INSERT INTO parameter_presets
     │                        │                    │
     │                        │<──────────────────
     │                        │
     │                        └─ INVALID ─────────> Return error
     │                                              {
     │                                                "detail": "Invalid format: ..."
     │                                              }
     │<───────────────
     └─> Show success/error message
```

## Component Hierarchy

```
App.tsx
└── Router
    ├── Dashboard.tsx
    ├── Tasks.tsx
    │   └── TaskList
    ├── NewTask.tsx ★
    │   ├── PresetSelector ★ NEW
    │   │   ├── PresetDropdown (multi-select)
    │   │   ├── AppliedPresetsChips
    │   │   ├── MergeStrategySelector
    │   │   ├── ParameterPreview
    │   │   └── ConflictWarnings
    │   ├── TaskForm
    │   │   ├── ModelConfig
    │   │   ├── RuntimeConfig
    │   │   ├── ParameterForm ← Pre-filled with merged params
    │   │   └── BenchmarkConfig
    │   └── SubmitButton
    ├── Experiments.tsx
    ├── Presets.tsx ★ NEW
    │   ├── PresetList
    │   │   ├── CategoryFilter
    │   │   ├── SearchBar
    │   │   └── PresetTable
    │   │       └── PresetRow
    │   │           ├── EditButton → PresetModal
    │   │           ├── DeleteButton → ConfirmDialog
    │   │           └── ExportButton
    │   ├── CreateButton → PresetModal
    │   ├── ImportButton → FileUpload
    │   └── PresetModal ★ NEW
    │       ├── NameInput
    │       ├── DescriptionInput
    │       ├── CategorySelect
    │       ├── ParameterEditor (JSON editor)
    │       └── MetadataEditor
    └── Containers.tsx
```

## Database Schema Visualization

```
┌─────────────────────────────────────────────────────────┐
│              TABLE: parameter_presets                    │
├──────────────┬──────────────────┬──────────────────────┤
│ Column       │ Type             │ Constraints          │
├──────────────┼──────────────────┼──────────────────────┤
│ id           │ INTEGER          │ PRIMARY KEY, AUTO    │
│ name         │ VARCHAR(255)     │ NOT NULL, UNIQUE     │
│ description  │ TEXT             │ NULL                 │
│ category     │ VARCHAR(100)     │ NULL, INDEXED        │
│ is_system    │ BOOLEAN          │ DEFAULT FALSE        │
│ parameters   │ JSON             │ NOT NULL             │
│ metadata     │ JSON             │ NULL                 │
│ created_at   │ TIMESTAMP        │ DEFAULT NOW()        │
│ updated_at   │ TIMESTAMP        │ ON UPDATE NOW()      │
└──────────────┴──────────────────┴──────────────────────┘

Indexes:
• idx_preset_category ON (category)
• idx_preset_system ON (is_system)

Relationships:
• No foreign keys (standalone table)
• Used by tasks indirectly (parameters applied at creation)

Sample Data:
┌────┬──────────────────┬──────────────────┬────────────┬───────────┬──────────────────────────────┐
│ id │ name             │ category         │ is_system  │ parameters                       │
├────┼──────────────────┼──────────────────┼────────────┼──────────────────────────────────┤
│  1 │ Memory Efficient │ memory           │ true       │ {"tp-size": [1], ...}            │
│  2 │ High Throughput  │ performance      │ true       │ {"tp-size": [2,4], ...}          │
│  3 │ Low Latency      │ performance      │ true       │ {"schedule-policy": ["lpm"], ...}│
│  4 │ Balanced         │ general          │ true       │ {"tp-size": [1,2], ...}          │
│ 10 │ My Custom        │ custom           │ false      │ {"tp-size": [8], ...}            │
└────┴──────────────────┴──────────────────┴────────────┴──────────────────────────────────┘
```

## Merge Strategy Comparison

```
┌───────────────────────────────────────────────────────────────────────────────────┐
│                            MERGE STRATEGIES                                        │
├───────────────────────────────────────────────────────────────────────────────────┤
│                                                                                    │
│  Preset A: {                              Preset B: {                             │
│    "tp-size": [1, 2],                       "tp-size": [2, 4],                    │
│    "mem-fraction": [0.8]                    "schedule-policy": ["lpm"]            │
│  }                                        }                                        │
│                                                                                    │
├───────────────────────────────────────────────────────────────────────────────────┤
│                                                                                    │
│  1. UNION (Default)                       Best for: Exploring all combinations    │
│     ────────────────                                                               │
│     Combines all values from both         Use case: Initial tuning, grid search  │
│                                                                                    │
│     Result: {                                                                      │
│       "tp-size": [1, 2, 4],              ← Combined & deduplicated                │
│       "mem-fraction": [0.8],             ← From A only                            │
│       "schedule-policy": ["lpm"]         ← From B only                            │
│     }                                                                              │
│                                                                                    │
│     Conflicts: None                                                                │
│                                                                                    │
├───────────────────────────────────────────────────────────────────────────────────┤
│                                                                                    │
│  2. INTERSECTION                          Best for: Conservative tuning           │
│     ──────────────                                                                 │
│     Only keeps common values              Use case: Finding optimal overlap       │
│                                                                                    │
│     Result: {                                                                      │
│       "tp-size": [2]                     ← Only value in both A and B            │
│     }                                                                              │
│                                                                                    │
│     Conflicts:                                                                     │
│       - "mem-fraction" not in B                                                    │
│       - "schedule-policy" not in A                                                 │
│                                                                                    │
├───────────────────────────────────────────────────────────────────────────────────┤
│                                                                                    │
│  3. LAST WINS                             Best for: Applying base + override      │
│     ───────────                                                                    │
│     Later presets override earlier        Use case: Base config + specialization  │
│                                                                                    │
│     Result (A then B):                                                             │
│     {                                                                              │
│       "tp-size": [2, 4],                 ← B overrides A                          │
│       "mem-fraction": [0.8],             ← From A (not in B)                      │
│       "schedule-policy": ["lpm"]         ← From B (not in A)                      │
│     }                                                                              │
│                                                                                    │
│     Conflicts:                                                                     │
│       - "tp-size" overridden (was [1,2], now [2,4])                               │
│                                                                                    │
└───────────────────────────────────────────────────────────────────────────────────┘
```

## State Management (Frontend)

```
Component State:
┌─────────────────────────────────────────────────────────┐
│ PresetSelector                                           │
│                                                          │
│ Local State:                                             │
│ • selectedPresets: number[]                              │
│ • mergeStrategy: MergeStrategy                           │
│ • showConflicts: boolean                                 │
│                                                          │
│ React Query Cache:                                       │
│ • ['presets'] → Preset[]                                 │
│ • ['merge', ids, strategy] → MergeResult                 │
│                                                          │
│ Props:                                                   │
│ • value: number[]        (controlled)                    │
│ • onChange: (ids) => void                                │
│ • onMergedParamsChange: (params) => void                 │
└─────────────────────────────────────────────────────────┘
         │
         │ onMergedParamsChange(merged.parameters)
         ▼
┌─────────────────────────────────────────────────────────┐
│ NewTask / EditTask                                       │
│                                                          │
│ Form State:                                              │
│ • parameters: ParameterMap ← from presets or manual     │
│ • model: ModelConfig                                     │
│ • runtime: RuntimeConfig                                 │
│ • benchmark: BenchmarkConfig                             │
│                                                          │
│ Submission:                                              │
│ • POST /api/tasks/ with parameters                       │
└─────────────────────────────────────────────────────────┘
```

## API Request/Response Examples

```json
// GET /api/presets/
Response: [
  {
    "id": 1,
    "name": "Memory Efficient",
    "description": "Optimized for low memory usage",
    "category": "memory",
    "is_system": true,
    "parameters": {
      "tp-size": [1],
      "mem-fraction-static": [0.7, 0.75]
    },
    "metadata": {
      "author": "system",
      "tags": ["memory", "small-gpu"]
    },
    "created_at": "2024-10-01T10:00:00Z",
    "updated_at": "2024-10-01T10:00:00Z"
  }
]

// POST /api/presets/merge
Request: {
  "preset_ids": [1, 2],
  "merge_strategy": "union"
}
Response: {
  "parameters": {
    "tp-size": [1, 2, 4],
    "mem-fraction-static": [0.7, 0.75, 0.9],
    "schedule-policy": ["fcfs", "lpm"]
  },
  "applied_presets": ["Memory Efficient", "High Throughput"],
  "conflicts": []
}

// POST /api/presets/
Request: {
  "name": "My Custom Preset",
  "description": "Custom config for large models",
  "category": "custom",
  "parameters": {
    "tp-size": [8],
    "mem-fraction-static": [0.95]
  },
  "metadata": {
    "author": "user123",
    "tags": ["custom", "large-model"]
  }
}
Response: {
  "id": 10,
  "name": "My Custom Preset",
  // ... full preset object
}
```

## File Structure

```
inference-autotuner/
├── src/
│   ├── utils/
│   │   ├── preset_merger.py ★ NEW
│   │   └── preset_validator.py ★ NEW
│   └── web/
│       ├── db/
│       │   ├── models.py (+ ParameterPreset) ★ MODIFIED
│       │   └── seed_presets.py ★ NEW
│       ├── routes/
│       │   └── presets.py ★ NEW
│       └── schemas/
│           └── preset.py ★ NEW
│
├── frontend/
│   └── src/
│       ├── components/
│       │   └── PresetSelector.tsx ★ NEW
│       ├── pages/
│       │   ├── Presets.tsx ★ NEW
│       │   ├── NewTask.tsx (modified) ★ MODIFIED
│       │   └── Tasks.tsx (modified for edit) ★ MODIFIED
│       ├── services/
│       │   └── presetService.ts ★ NEW
│       └── types/
│           └── preset.ts ★ NEW
│
├── docs/
│   ├── PRESET_SYSTEM_DESIGN.md ★ NEW
│   ├── PRESET_IMPLEMENTATION_GUIDE.md ★ NEW
│   └── PRESET_ARCHITECTURE_DIAGRAM.md ★ NEW (this file)
│
└── tests/
    ├── test_preset_merger.py ★ NEW
    └── test_preset_routes.py ★ NEW
```
