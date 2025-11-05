-- Add slo_config and metadata columns to tasks table
-- Migration: add_slo_and_metadata_to_tasks
-- Date: 2025-11-05
-- SQLite compatible version (no IF NOT EXISTS, no COMMENT ON)

-- Add slo_config column (JSON, nullable)
ALTER TABLE tasks ADD COLUMN slo_config JSON;

-- Add metadata column (JSON, nullable)
ALTER TABLE tasks ADD COLUMN metadata JSON;
