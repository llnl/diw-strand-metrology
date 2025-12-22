"""Exception classes for transform configuration errors."""

from typing import Dict, Any


class TransformConfigError(Exception):
    """Base exception for transform configuration errors."""
    
    def __init__(self, message: str, context: Dict[str, Any] = None):
        super().__init__(message)
        self.context = context or {}


class YAMLParsingError(TransformConfigError):
    """Exception raised when YAML configuration cannot be parsed."""
    pass


class TransformConstructionError(TransformConfigError):
    """Exception raised when transform construction fails."""
    pass


class ParameterValidationError(TransformConfigError):
    """Exception raised when transform parameters are invalid."""
    pass