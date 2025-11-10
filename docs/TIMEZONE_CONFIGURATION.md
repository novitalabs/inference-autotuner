# Timezone Configuration

The inference autotuner supports configurable timezone display for all timestamps in both backend and frontend.

## Configuration

### Backend Configuration

Set the timezone via environment variable in `.env`:

```bash
# Timezone for displaying timestamps (e.g., 'UTC', 'Asia/Shanghai', 'America/New_York')
TIMEZONE=Asia/Shanghai
```

Valid timezone values follow the [IANA Time Zone Database](https://en.wikipedia.org/wiki/List_of_tz_database_time_zones) format.

**Common timezones:**
- `UTC` - Coordinated Universal Time (default)
- `Asia/Shanghai` - China Standard Time (UTC+8)
- `America/New_York` - Eastern Time
- `America/Los_Angeles` - Pacific Time
- `Europe/London` - British Time
- `Asia/Tokyo` - Japan Standard Time

### How It Works

1. **Backend** (`src/web/config.py`):
   - Reads `TIMEZONE` from environment variable
   - Defaults to `UTC` if not set
   - Exposes timezone via `/api/info` endpoint

2. **Frontend** (`frontend/src/contexts/TimezoneContext.tsx`):
   - Fetches timezone from backend on app load
   - Provides timezone-aware formatting functions via React context
   - All datetime displays use the configured timezone

3. **API Response**:
   ```json
   {
     "app_name": "LLM Inference Autotuner API",
     "version": "0.1.0",
     "deployment_mode": "docker",
     "available_runtimes": ["sglang", "vllm"],
     "timezone": "Asia/Shanghai"
   }
   ```

## Usage in Components

Import and use the timezone context in React components:

```typescript
import { useTimezone } from '../contexts/TimezoneContext';

function MyComponent() {
  const { formatTime, formatDate, formatDateTime, timezone } = useTimezone();

  return (
    <div>
      <p>Current timezone: {timezone}</p>
      <p>Time: {formatTime(new Date())}</p>
      <p>Date: {formatDate(new Date())}</p>
      <p>DateTime: {formatDateTime(new Date())}</p>
    </div>
  );
}
```

### Formatting Functions

- **`formatTime(date)`**: Returns time in HH:mm format (24-hour)
- **`formatDate(date)`**: Returns date in MM/DD/YYYY format
- **`formatDateTime(date)`**: Returns full datetime in MM/DD/YYYY, HH:mm:ss format
- **`timezone`**: Current timezone string

All functions accept either `Date` object or ISO string.

## Components Using Timezone

Currently implemented in:
- **Dashboard Timeline**: Experiment timeline X-axis labels use `formatTime()`
- Can be extended to other components showing timestamps

## Benefits

1. **Consistent Display**: All users see times in the same configured timezone
2. **Global Teams**: Teams across different locations can use a common timezone
3. **No Client Timezone Confusion**: Avoids ambiguity from different client local times
4. **Easy Configuration**: Single environment variable controls all display

## Example

With `TIMEZONE=Asia/Shanghai`:
- Backend stores timestamps in UTC in database (always)
- Frontend fetches timezone setting: `"Asia/Shanghai"`
- Timeline shows: `14:30`, `15:00`, `15:30` (Beijing time)
- Without config (defaults to UTC): Shows times in UTC instead

## Technical Details

### Storage vs Display

- **Database**: Always stores timestamps in UTC (SQLite TIMESTAMP)
- **Display**: Converts to configured timezone for presentation
- **API**: Returns ISO 8601 strings with timezone info

### Browser Compatibility

Uses standard JavaScript `Intl.DateTimeFormat` API with `timeZone` option:
- Supported in all modern browsers
- Handles daylight saving time automatically
- No external dependencies required

### Performance

- Timezone fetched once on app load
- Cached in React context
- No per-render overhead
- Formatting is lightweight browser API call
