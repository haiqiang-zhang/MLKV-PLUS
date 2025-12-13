from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, List, Optional, overload
from .BindingBase import BindingBase

class DistBindingBase(BindingBase):
    """Base interface class for distributed bindings"""
    
    @abstractmethod
    def rank(self):
        """Get the rank of the current process"""
        pass
    
    @abstractmethod
    def num_ranks(self):
        """Get the number of ranks"""
        pass
    
    @abstractmethod
    def broadcast(self, tensor, root=0):
        """Broadcast a tensor from the root rank to all ranks"""
        pass
    
    @abstractmethod
    def barrier(self):
        """Barrier all ranks"""
        pass
    
    @abstractmethod
    def allreduce(self, tensor, op):
        """Allreduce a tensor"""
        pass