# Database Migration - ClusterBaseModel & ClusterServingRuntime

**Date:** 2025-11-18
**Migration:** Add 4 new columns to `tasks` table

## Issue

After implementing ClusterBaseModel and ClusterServingRuntime features, the database schema was out of sync with the SQLAlchemy models, causing this error:

```
sqlalchemy.exc.OperationalError: (sqlite3.OperationalError) no such column: tasks.clusterbasemodel_config
```

## Solution

Created and ran migration script: `scripts/migrate_cbm_csr.py`

## Columns Added

| Column Name | Type | Description |
|-------------|------|-------------|
| `clusterbasemodel_config` | JSON | Stores preset name or custom ClusterBaseModel spec |
| `clusterservingruntime_config` | JSON | Stores preset name or custom ClusterServingRuntime spec |
| `created_clusterbasemodel` | VARCHAR | Name of ClusterBaseModel if auto-created by this task |
| `created_clusterservingruntime` | VARCHAR | Name of ClusterServingRuntime if auto-created by this task |

## How to Run Migration

If you encounter the "no such column" error on a different database:

```bash
cd /root/work/inference-autotuner
python scripts/migrate_cbm_csr.py
```

**Output:**
```
Migrating database: /home/user/.local/share/inference-autotuner/autotuner.db
  ✓ Added column 'clusterbasemodel_config' (JSON)
  ✓ Added column 'clusterservingruntime_config' (JSON)
  ✓ Added column 'created_clusterbasemodel' (VARCHAR)
  ✓ Added column 'created_clusterservingruntime' (VARCHAR)

✅ Migration completed successfully!
```

## Verification

After migration, verify the columns exist:

```bash
# Check API response includes new fields
curl -s 'http://localhost:8000/api/tasks/1' | grep clusterbasemodel

# Should output:
# "clusterbasemodel_config": null,
# "created_clusterbasemodel": null,
# "clusterservingruntime_config": null,
# "created_clusterservingruntime": null,
```

## Notes

- **Idempotent:** Script checks if columns exist before adding them
- **Safe:** Uses ALTER TABLE (doesn't drop or modify existing data)
- **Automatic:** Future installations will have columns from initial DB creation
- **Nullable:** All new columns are nullable (won't break existing tasks)

## For New Installations

New installations automatically get the correct schema via SQLAlchemy's `init_db()` function. No migration needed.

## Migration Status

✅ **Completed:** November 18, 2025
✅ **Verified:** All API endpoints working correctly
✅ **Backward Compatible:** Existing tasks unaffected
