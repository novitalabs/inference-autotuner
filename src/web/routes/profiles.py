"""
Configuration profiles API endpoints.
"""

from fastapi import APIRouter, HTTPException, status
from typing import List, Dict, Any

from config.factory import TaskConfigFactory

router = APIRouter()


@router.get("/", response_model=List[Dict[str, Any]])
async def list_profiles():
    """
    List all available configuration profiles.

    Returns a list of profiles with their metadata including:
    - name: Profile identifier
    - description: Human-readable description
    - use_case: Recommended use cases
    - tags: Search tags
    - recommended_for: Recommended scenarios
    - layers_count: Number of configuration layers
    """
    profiles = TaskConfigFactory.list_profiles()
    return profiles


@router.get("/{profile_name}", response_model=Dict[str, Any])
async def get_profile(profile_name: str):
    """
    Get detailed information about a specific profile.

    Args:
        profile_name: Profile identifier

    Returns:
        Profile information including metadata and layer names
    """
    profile_info = TaskConfigFactory.get_profile_info(profile_name)

    if not profile_info:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Profile '{profile_name}' not found"
        )

    return profile_info
