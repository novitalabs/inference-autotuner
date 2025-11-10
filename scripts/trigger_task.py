#!/usr/bin/env python3
"""
Trigger Task 7 via ARQ directly.
"""
import asyncio
from src.web.workers.client import enqueue_autotuning_task

async def main():
    job_id = await enqueue_autotuning_task(7)
    print(f"Task 7 enqueued: job_id={job_id}")

if __name__ == "__main__":
    asyncio.run(main())
