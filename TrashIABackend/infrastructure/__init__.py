"""
Infrastructure layer - External service clients and adapters.
"""
from infrastructure.supabase_client import SupabaseClient, get_supabase_client

__all__ = [
    "SupabaseClient",
    "get_supabase_client",
]
