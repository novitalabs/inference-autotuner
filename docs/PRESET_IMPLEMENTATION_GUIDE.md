# Parameter Preset System - Implementation Guide

## Quick Start

This guide provides step-by-step instructions for implementing the parameter preset system.

## Phase 1: Backend Foundation (Days 1-2)

### Step 1: Database Model and Migration

1. **Create the database model**:

```bash
# Edit: src/web/db/models.py
# Add ParameterPreset class from design doc
```

2. **Generate migration** (if using Alembic):

```bash
alembic revision --autogenerate -m "Add parameter_presets table"
alembic upgrade head
```

Or for raw SQLite:

```bash
sqlite3 ~/.local/share/inference-autotuner/autotuner.db < migrations/add_presets.sql
```

Migration SQL:
```sql
-- migrations/add_presets.sql
CREATE TABLE parameter_presets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name VARCHAR(255) NOT NULL UNIQUE,
    description TEXT,
    category VARCHAR(100),
    is_system BOOLEAN DEFAULT FALSE,
    parameters JSON NOT NULL,
    metadata JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_preset_category ON parameter_presets(category);
CREATE INDEX idx_preset_system ON parameter_presets(is_system);
```

### Step 2: Implement Merge Logic

1. **Create preset merger utility**:

```bash
# Create: src/utils/preset_merger.py
# Implement PresetMerger class with three strategies
```

2. **Test merge logic**:

```bash
# Create: tests/test_preset_merger.py
pytest tests/test_preset_merger.py -v
```

Example test cases:
```python
def test_union_merge():
    presets = [
        {"name": "A", "parameters": {"tp-size": [1, 2]}},
        {"name": "B", "parameters": {"tp-size": [3], "mem-fraction": [0.9]}}
    ]
    result, conflicts = PresetMerger.merge_parameters(presets, MergeStrategy.UNION)
    assert result == {"tp-size": [1, 2, 3], "mem-fraction": [0.9]}

def test_intersection_merge():
    # Similar test for intersection strategy
    pass

def test_last_wins_merge():
    # Similar test for last_wins strategy
    pass
```

### Step 3: API Routes and Schemas

1. **Create Pydantic schemas**:

```bash
# Create: src/web/schemas/preset.py
# Add PresetCreate, PresetUpdate, PresetResponse, etc.
```

2. **Implement API routes**:

```bash
# Create: src/web/routes/presets.py
# Implement all CRUD endpoints
```

3. **Register routes in main app**:

```python
# Edit: src/web/app.py
from web.routes import presets

app.include_router(presets.router)
```

4. **Test API endpoints**:

```bash
# Start server
./scripts/start_dev.sh

# Test with curl
curl -X POST http://localhost:8000/api/presets/ \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Test Preset",
    "parameters": {"tp-size": [1, 2]}
  }'

curl http://localhost:8000/api/presets/
```

### Step 4: System Presets Seeding

1. **Create seeder script**:

```bash
# Create: src/web/db/seed_presets.py
# Implement seed_system_presets function
```

2. **Add seeding to startup**:

```python
# Edit: src/web/server.py or app.py startup event
@app.on_event("startup")
async def startup_event():
    async with get_db() as db:
        await seed_system_presets(db)
```

3. **Run seeding**:

```bash
python -c "from src.web.db.seed_presets import seed_system_presets; import asyncio; asyncio.run(seed_system_presets())"
```

## Phase 2: Frontend Components (Days 3-4)

### Step 5: API Client Services

1. **Create preset service**:

```typescript
// frontend/src/services/presetService.ts
import axios from 'axios';

const API_BASE = 'http://localhost:8000/api';

export interface Preset {
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

export const presetService = {
  async getAll(category?: string): Promise<Preset[]> {
    const params = category ? { category } : {};
    const response = await axios.get(`${API_BASE}/presets/`, { params });
    return response.data;
  },

  async getById(id: number): Promise<Preset> {
    const response = await axios.get(`${API_BASE}/presets/${id}`);
    return response.data;
  },

  async create(preset: Omit<Preset, 'id' | 'is_system' | 'created_at' | 'updated_at'>): Promise<Preset> {
    const response = await axios.post(`${API_BASE}/presets/`, preset);
    return response.data;
  },

  async update(id: number, preset: Partial<Preset>): Promise<Preset> {
    const response = await axios.put(`${API_BASE}/presets/${id}`, preset);
    return response.data;
  },

  async delete(id: number): Promise<void> {
    await axios.delete(`${API_BASE}/presets/${id}`);
  },

  async merge(presetIds: number[], strategy: string = 'union'): Promise<any> {
    const response = await axios.post(`${API_BASE}/presets/merge`, {
      preset_ids: presetIds,
      merge_strategy: strategy
    });
    return response.data;
  },

  async importPreset(file: File): Promise<Preset> {
    const formData = new FormData();
    formData.append('file', file);
    const response = await axios.post(`${API_BASE}/presets/import`, formData, {
      headers: { 'Content-Type': 'multipart/form-data' }
    });
    return response.data;
  },

  async exportPreset(id: number): Promise<Blob> {
    const response = await axios.get(`${API_BASE}/presets/${id}/export`, {
      responseType: 'blob'
    });
    return response.data;
  }
};
```

2. **Create TypeScript types**:

```typescript
// frontend/src/types/preset.ts
export interface Preset {
  id: number;
  name: string;
  description?: string;
  category?: string;
  is_system: boolean;
  parameters: ParameterMap;
  metadata?: PresetMetadata;
  created_at: string;
  updated_at?: string;
}

export interface ParameterMap {
  [key: string]: any[];
}

export interface PresetMetadata {
  author?: string;
  tags?: string[];
  recommended_for?: string[];
  runtime?: string;
}

export interface MergeResult {
  parameters: ParameterMap;
  applied_presets: string[];
  conflicts?: Conflict[];
}

export interface Conflict {
  parameter: string;
  reason: string;
  [key: string]: any;
}

export type MergeStrategy = 'union' | 'intersection' | 'last_wins';
```

### Step 6: Preset Management Page

1. **Create Presets page component**:

```bash
# Create: frontend/src/pages/Presets.tsx
```

Key features to implement:
- List view with table/grid
- Create/Edit modal
- Delete confirmation dialog
- Category filter
- Search functionality
- Import/Export buttons

```typescript
// Skeleton structure
export default function Presets() {
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null);
  const [isCreateOpen, setIsCreateOpen] = useState(false);
  const [editingPreset, setEditingPreset] = useState<Preset | null>(null);

  const { data: presets, isLoading } = useQuery({
    queryKey: ['presets', selectedCategory],
    queryFn: () => presetService.getAll(selectedCategory)
  });

  return (
    <div className="container mx-auto p-4">
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-2xl font-bold">Parameter Presets</h1>
        <div className="flex gap-2">
          <button onClick={() => setIsCreateOpen(true)}>Create Preset</button>
          <button onClick={handleImport}>Import</button>
        </div>
      </div>

      {/* Category filter */}
      <CategoryFilter value={selectedCategory} onChange={setSelectedCategory} />

      {/* Presets table/grid */}
      <PresetsTable
        presets={presets}
        onEdit={setEditingPreset}
        onDelete={handleDelete}
        onExport={handleExport}
      />

      {/* Create/Edit modal */}
      {(isCreateOpen || editingPreset) && (
        <PresetModal
          preset={editingPreset}
          onClose={() => { setIsCreateOpen(false); setEditingPreset(null); }}
          onSave={handleSave}
        />
      )}
    </div>
  );
}
```

### Step 7: Preset Selector Component

1. **Create PresetSelector component**:

```bash
# Create: frontend/src/components/PresetSelector.tsx
```

Key features:
- Multi-select dropdown
- Applied presets shown as chips
- Merge strategy selector
- Parameter preview
- Conflict warnings

```typescript
interface PresetSelectorProps {
  value: number[];
  onChange: (presetIds: number[]) => void;
  onMergedParamsChange: (params: ParameterMap) => void;
  mergeStrategy?: MergeStrategy;
}

export default function PresetSelector({
  value,
  onChange,
  onMergedParamsChange,
  mergeStrategy = 'union'
}: PresetSelectorProps) {
  const [strategy, setStrategy] = useState(mergeStrategy);

  // Fetch all presets
  const { data: presets } = useQuery({
    queryKey: ['presets'],
    queryFn: () => presetService.getAll()
  });

  // Auto-merge when selection changes
  const { data: mergeResult } = useQuery({
    queryKey: ['merge', value, strategy],
    queryFn: () => presetService.merge(value, strategy),
    enabled: value.length > 0
  });

  useEffect(() => {
    if (mergeResult) {
      onMergedParamsChange(mergeResult.parameters);
    }
  }, [mergeResult, onMergedParamsChange]);

  return (
    <div className="preset-selector border rounded-lg p-4">
      <h3 className="font-semibold mb-2">Apply Parameter Presets</h3>

      {/* Multi-select dropdown */}
      <select
        multiple
        value={value.map(String)}
        onChange={(e) => {
          const selected = Array.from(e.target.selectedOptions).map(o => Number(o.value));
          onChange(selected);
        }}
        className="w-full border rounded p-2 mb-3"
      >
        {presets?.map(preset => (
          <option key={preset.id} value={preset.id}>
            {preset.name} {preset.is_system && '(System)'}
          </option>
        ))}
      </select>

      {/* Applied presets chips */}
      <div className="flex flex-wrap gap-2 mb-3">
        {value.map(id => {
          const preset = presets?.find(p => p.id === id);
          return preset ? (
            <span key={id} className="bg-blue-100 px-3 py-1 rounded-full text-sm flex items-center gap-2">
              {preset.name}
              <button onClick={() => onChange(value.filter(v => v !== id))} className="text-red-500">×</button>
            </span>
          ) : null;
        })}
      </div>

      {/* Merge strategy */}
      {value.length > 1 && (
        <div className="mb-3">
          <label className="block text-sm font-medium mb-1">Merge Strategy</label>
          <select
            value={strategy}
            onChange={(e) => setStrategy(e.target.value as MergeStrategy)}
            className="border rounded px-3 py-2"
          >
            <option value="union">Union (Combine All)</option>
            <option value="intersection">Intersection (Common Only)</option>
            <option value="last_wins">Last Wins (Override)</option>
          </select>
        </div>
      )}

      {/* Conflicts warning */}
      {mergeResult?.conflicts && mergeResult.conflicts.length > 0 && (
        <div className="bg-yellow-50 border border-yellow-200 rounded p-3 mb-3">
          <p className="font-semibold text-yellow-800">⚠️ Merge Conflicts Detected</p>
          <ul className="text-sm text-yellow-700 mt-1">
            {mergeResult.conflicts.map((conflict, idx) => (
              <li key={idx}>• {conflict.parameter}: {conflict.reason}</li>
            ))}
          </ul>
        </div>
      )}

      {/* Parameter preview */}
      {mergeResult && (
        <div className="bg-gray-50 rounded p-3">
          <p className="text-sm font-medium mb-2">Merged Parameters Preview:</p>
          <pre className="text-xs overflow-auto max-h-40">
            {JSON.stringify(mergeResult.parameters, null, 2)}
          </pre>
        </div>
      )}
    </div>
  );
}
```

### Step 8: Integrate with NewTask Page

1. **Update NewTask.tsx**:

```typescript
// frontend/src/pages/NewTask.tsx

export default function NewTask() {
  const [usePresets, setUsePresets] = useState(false);
  const [selectedPresets, setSelectedPresets] = useState<number[]>([]);
  const [parameters, setParameters] = useState<ParameterMap>({});

  return (
    <div className="container mx-auto p-4">
      <h1 className="text-2xl font-bold mb-6">Create New Task</h1>

      {/* Toggle between presets and manual */}
      <div className="mb-6">
        <label className="flex items-center gap-2">
          <input
            type="checkbox"
            checked={usePresets}
            onChange={(e) => setUsePresets(e.target.checked)}
          />
          <span>Use Parameter Presets</span>
        </label>
      </div>

      {/* Preset selector */}
      {usePresets && (
        <PresetSelector
          value={selectedPresets}
          onChange={setSelectedPresets}
          onMergedParamsChange={setParameters}
        />
      )}

      {/* Existing parameter form (pre-filled if presets used) */}
      <ParameterForm
        parameters={parameters}
        onChange={setParameters}
        editable={true}
      />

      {/* Rest of the form... */}
    </div>
  );
}
```

### Step 9: Routing and Navigation

1. **Add route**:

```typescript
// frontend/src/App.tsx
import Presets from './pages/Presets';

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/tasks" element={<Tasks />} />
        <Route path="/tasks/new" element={<NewTask />} />
        <Route path="/experiments" element={<Experiments />} />
        <Route path="/presets" element={<Presets />} /> {/* NEW */}
        <Route path="/containers" element={<Containers />} />
      </Routes>
    </Router>
  );
}
```

2. **Add navigation link**:

```typescript
// frontend/src/components/Layout.tsx
<nav>
  <Link to="/">Dashboard</Link>
  <Link to="/tasks">Tasks</Link>
  <Link to="/experiments">Experiments</Link>
  <Link to="/presets">Presets</Link> {/* NEW */}
  <Link to="/containers">Containers</Link>
</nav>
```

## Phase 3: Testing and Polish (Day 5)

### Step 10: Testing

1. **Backend tests**:

```bash
pytest tests/test_preset_merger.py -v
pytest tests/test_preset_routes.py -v
```

2. **Frontend tests** (optional):

```bash
cd frontend
npm run test
```

3. **Manual E2E testing**:

- [ ] Create a new preset via UI
- [ ] Edit an existing preset
- [ ] Delete a preset (verify system presets cannot be deleted)
- [ ] Export a preset and verify JSON format
- [ ] Import a preset from JSON file
- [ ] Apply single preset to a task
- [ ] Apply multiple presets with union strategy
- [ ] Apply multiple presets with intersection strategy
- [ ] Apply multiple presets with last_wins strategy
- [ ] Verify conflicts are shown when applicable
- [ ] Create and run a task with preset-generated parameters
- [ ] Verify merged parameters are saved correctly

### Step 11: Documentation

1. **Update CLAUDE.md**:

```bash
# Add section about preset system
# Document new API endpoints
# Update development workflow
```

2. **Create user guide**:

```bash
# Create: docs/USER_GUIDE_PRESETS.md
# Include screenshots and examples
```

3. **Update README**:

```bash
# Add preset system to features list
# Update screenshots if applicable
```

## Verification Checklist

### Backend
- [ ] Database table created and indexed
- [ ] PresetMerger passes all unit tests
- [ ] All CRUD endpoints working
- [ ] Merge endpoint returns correct results
- [ ] Import validates JSON format
- [ ] Export generates valid JSON
- [ ] System presets seeded on startup
- [ ] System presets cannot be deleted/modified

### Frontend
- [ ] Presets page shows all presets
- [ ] Create preset modal works
- [ ] Edit preset modal works
- [ ] Delete confirmation dialog works
- [ ] Import file picker works
- [ ] Export downloads JSON file
- [ ] PresetSelector component renders
- [ ] Multi-select works correctly
- [ ] Merge strategy selector works
- [ ] Parameter preview updates on change
- [ ] Conflicts are displayed
- [ ] NewTask integrates preset selector
- [ ] Parameters pre-fill from presets

### Integration
- [ ] Frontend can fetch presets from API
- [ ] Create preset saves to database
- [ ] Update preset modifies database record
- [ ] Delete preset removes from database
- [ ] Import adds preset to database
- [ ] Export retrieves preset from database
- [ ] Merge calls backend and returns result
- [ ] Task creation uses merged parameters

## Troubleshooting

### Common Issues

**Issue**: Merge endpoint returns empty parameters
- **Fix**: Check that preset IDs exist in database
- **Fix**: Verify parameters field is not null/empty

**Issue**: Import fails with validation error
- **Fix**: Ensure JSON matches schema (version and preset fields)
- **Fix**: Check that parameters are lists, not single values

**Issue**: Frontend doesn't show presets
- **Fix**: Verify backend is running on port 8000
- **Fix**: Check browser console for CORS errors
- **Fix**: Ensure React Query is configured correctly

**Issue**: Merge conflicts not showing
- **Fix**: Verify merge strategy is not "union" (union has no conflicts)
- **Fix**: Check conflicts array in API response

**Issue**: System presets can be deleted
- **Fix**: Add check in delete endpoint: `if preset.is_system: raise HTTPException`

## Performance Considerations

1. **Database Indexing**: Ensure indexes on `category` and `is_system` columns
2. **Caching**: Consider caching frequently accessed presets
3. **Query Optimization**: Use pagination for large preset lists
4. **Frontend**: Use React Query caching to minimize API calls
5. **Merge Performance**: For large parameter sets, optimize merge logic

## Next Steps After Implementation

1. **User Feedback**: Gather feedback on merge strategies
2. **Analytics**: Track most-used presets
3. **Templates**: Add wizard for creating common preset types
4. **Validation**: Add parameter validation rules
5. **Sharing**: Allow users to share presets with team
6. **Versioning**: Track changes to presets over time
