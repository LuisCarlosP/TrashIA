"""
Supabase client singleton for database operations.
"""
import logging
from typing import Optional

from supabase import create_client, Client

from config.settings import SUPABASE_URL, SUPABASE_SERVICE_KEY

logger = logging.getLogger(__name__)


class SupabaseClient:
    """
    Singleton client for Supabase database operations.
    Uses the service role key for full database access.
    """
    _instance: Optional[Client] = None
    _initialized: bool = False
    
    @classmethod
    def get_client(cls) -> Client:
        """
        Get or create the Supabase client instance.
        
        Returns:
            Client: Supabase client instance
            
        Raises:
            RuntimeError: If Supabase configuration is missing
        """
        if cls._instance is None:
            if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
                raise RuntimeError(
                    "Supabase configuration missing. "
                    "Please set SUPABASE_URL and SUPABASE_SERVICE_KEY environment variables."
                )
            
            try:
                cls._instance = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
                cls._initialized = True
                logger.info("Supabase client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Supabase client: {e}")
                raise RuntimeError(f"Failed to connect to Supabase: {e}")
        
        return cls._instance
    
    @classmethod
    def is_initialized(cls) -> bool:
        """Check if the client has been initialized."""
        return cls._initialized
    
    @classmethod
    def reset(cls) -> None:
        """Reset the client instance (useful for testing)."""
        cls._instance = None
        cls._initialized = False


def get_supabase_client() -> Client:
    """
    Dependency injection function for FastAPI.
    
    Returns:
        Client: Supabase client instance
    """
    return SupabaseClient.get_client()
