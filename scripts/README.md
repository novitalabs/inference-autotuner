# Database Management Scripts

## reset_db.py

A comprehensive database management tool for resetting and managing the Inference Autotuner database.

### Features

- **Drop and recreate tables** - Clear all data while keeping the database file
- **Delete database file** - Completely remove the database file
- **Reset task status** - Reset tasks to PENDING status
- **List tasks** - View all tasks in the database

### Usage

```bash
# Activate virtual environment first
source env/bin/activate

# List all tasks
python scripts/reset_db.py --list-tasks

# Reset specific task to PENDING
python scripts/reset_db.py --task-id 1

# Reset all tasks to PENDING
python scripts/reset_db.py --reset-tasks

# Drop all tables and recreate (clears all data)
python scripts/reset_db.py --drop-tables

# Delete database file completely
python scripts/reset_db.py --delete-db
```

### Examples

#### Check current tasks
```bash
$ python scripts/reset_db.py --list-tasks

📋 Found 1 task(s):

  ID: 1
  Name: docker-simple-tune
  Status: running
  Created: 2025-10-24 07:51:50.809251
  Started: 2025-10-27 07:39:35.804044
```

#### Reset a task to PENDING
```bash
$ python scripts/reset_db.py --task-id 1
✅ Task #1 'docker-simple-tune': running → PENDING
```

#### Reset all tasks
```bash
$ python scripts/reset_db.py --reset-tasks
🔄 Resetting 2 task(s) to PENDING status...
  • Task #1 'docker-simple-tune': running → PENDING
  • Task #2 'vllm-optimization': completed → PENDING
✅ All tasks reset to PENDING
```

#### Drop and recreate all tables
```bash
$ python scripts/reset_db.py --drop-tables
🔄 Dropping all tables...
✅ All tables dropped
🔄 Creating tables...
✅ Tables created successfully
```

### Database Location

The database is stored at:
```
~/.local/share/inference-autotuner/autotuner.db
```

### Notes

- The script requires the virtual environment to be activated
- SQLAlchemy logging is enabled, so you'll see SQL queries in the output
- Resetting tasks only affects status and timestamps, not configuration
- When the database file is deleted, it will be automatically recreated when the server starts
