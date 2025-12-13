from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, List, Optional, overload

class BindingBase(ABC):
    """Base interface class for database operations"""
    
    @abstractmethod
    def load_data(self, load_data_output_file: Optional[str] = None) -> None:
        """Load initial data into the database"""
        pass
    
    @abstractmethod
    def insert(self) -> Tuple[float, float, Dict[str, Any]]:
        """Insert a new record into the database
        Returns:
            Tuple[float, float, Dict[str, Any]]: (start_time, end_time, operation_details)
        """
        pass
    
    @abstractmethod
    def read(self) -> Tuple[float, float, Dict[str, Any]]:
        """Read a record from the database
        Returns:
            Tuple[float, float, Dict[str, Any]]: (start_time, end_time, operation_details)
        """
        pass
    
    @abstractmethod
    def update(self) -> Tuple[float, float, Dict[str, Any]]:
        """Update a record in the database
        Returns:
            Tuple[float, float, Dict[str, Any]]: (start_time, end_time, operation_details)
        """
        pass
    
    @abstractmethod
    def scan(self) -> Tuple[float, float, Dict[str, Any]]:
        """Scan records from the database
        Returns:
            Tuple[float, float, Dict[str, Any]]: (start_time, end_time, operation_details)
        """
        pass
    
    @abstractmethod
    def multiget(self, batch_size: int) -> Tuple[float, float, Dict[str, Any]]:
        """Get multiple records from the database
        Args:
            batch_size: Number of records to get
        Returns:
            Tuple[float, float, Dict[str, Any]]: (start_time, end_time, operation_details)
        """
        pass
    
    @abstractmethod
    def multiset(self, batch_size: int) -> Tuple[float, float, Dict[str, Any]]:
        """Update multiple records in the database
        Args:
            batch_size: Number of records to update
        Returns:
            Tuple[float, float, Dict[str, Any]]: (start_time, end_time, operation_details)
        """
        pass 