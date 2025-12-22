import albumentations as A
import yaml
import inspect
from typing import Tuple, Dict, Type, Any

from .exceptions import TransformConfigError, ParameterValidationError, TransformConstructionError


class TransformRegistry:
    """Registry for dynamic discovery and mapping of Albumentations transforms."""
    
    def __init__(self):
        self.transforms: Dict[str, Type] = {}
        self.aliases: Dict[str, str] = {}  # For case-insensitive lookup
        self._discover_transforms()
    
    def _discover_transforms(self) -> None:
        """Auto-discover available Albumentations transforms through introspection."""
        for name, obj in inspect.getmembers(A):
            if (inspect.isclass(obj) and 
                hasattr(A, 'BasicTransform') and
                issubclass(obj, A.BasicTransform) and 
                obj != A.BasicTransform):
                # Register both exact name and lowercase alias
                self.transforms[name] = obj
                self.aliases[name.lower()] = name
    
    def get_transform_class(self, name: str) -> Type:
        """Get transform class by name (case-insensitive)."""
        # Try exact match first
        if name in self.transforms:
            return self.transforms[name]
        
        # Try case-insensitive match
        lower_name = name.lower()
        if lower_name in self.aliases:
            return self.transforms[self.aliases[lower_name]]
        
        # Provide helpful error with suggestions
        available_names = list(self.transforms.keys())
        similar_names = [n for n in available_names if lower_name in n.lower() or n.lower() in lower_name]
        
        error_msg = f"Unknown transform '{name}'"
        if similar_names:
            error_msg += f". Did you mean one of: {similar_names[:5]}?"
        error_msg += f" Available transforms: {sorted(available_names)}"
        
        raise TransformConfigError(error_msg)
    
    def list_available_transforms(self) -> list[str]:
        """Return list of all available transform names."""
        return sorted(self.transforms.keys())


def build_transforms(config_file: str) -> Tuple[A.Compose, A.Compose]:
    """
    Builds the train and validation/test transform sets from the given config_file.
    The config_file is yaml format and should contain a train and val section with 
    each section being a list of augmentation dicts.

    train:
        - name: RandomCrop
          height: 1200
          width: 1200
          pad_if_needed: true
        - name: RandomRotation
        - name: Normalize
        - name: ToTensorV2
    val:
        - name: RandomCrop
          height: 1200
          width: 1200
          pad_if_needed: true
        - name: Normalize
        - name: ToTensorV2

    Args:
        config_file: Path to YAML configuration file
        
    Returns:
        Tuple of (train_transforms, val_transforms)
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file has invalid YAML syntax
        TransformConfigError: If transform specifications are invalid
    """
    try:
        with open(config_file, 'r') as fp:
            transform_config = yaml.safe_load(fp)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_file}")
    except yaml.YAMLError as e:
        raise TransformConfigError(f"Invalid YAML syntax in {config_file}: {e}")
    
    if transform_config is None:
        transform_config = {}
    
    # Handle missing sections gracefully with empty defaults
    train_transforms = _build_transform(transform_config.get("train", []))
    val_transforms = _build_transform(transform_config.get("val", []))

    return train_transforms, val_transforms


# Global registry instance
_transform_registry = None


def get_transform_registry() -> TransformRegistry:
    """Get the global transform registry instance."""
    global _transform_registry
    if _transform_registry is None:
        _transform_registry = TransformRegistry()
    return _transform_registry


def _validate_parameters(transform_class: Type, kwargs: Dict[str, Any]) -> None:
    """Validate parameters against transform signature."""
    try:
        sig = inspect.signature(transform_class.__init__)
        valid_params = set(sig.parameters.keys()) - {'self'}
        invalid_params = set(kwargs.keys()) - valid_params
        
        if invalid_params:
            # Provide helpful suggestions for common typos
            suggestions = []
            for invalid_param in invalid_params:
                for valid_param in valid_params:
                    if (invalid_param.lower() in valid_param.lower() or 
                        valid_param.lower() in invalid_param.lower()):
                        suggestions.append(f"'{invalid_param}' -> '{valid_param}'")
            
            error_msg = f"Invalid parameters for {transform_class.__name__}: {list(invalid_params)}"
            if suggestions:
                error_msg += f". Did you mean: {suggestions}?"
            error_msg += f" Valid parameters: {sorted(valid_params)}"
            
            raise ParameterValidationError(error_msg)
    except Exception as e:
        if isinstance(e, ParameterValidationError):
            raise
        raise TransformConfigError(f"Failed to validate parameters for {transform_class.__name__}: {e}")


def _build_transform(transform_list: list[dict]) -> A.Compose:
    """
    Builds a set of transforms from a list of dicts defining the set of 
    Albumentations transforms to use.

    An example list of transforms will look like:
    [
        {'name': 'RandomCrop', 'height': 1200, 'width': 1200, 'pad_if_needed': True},
        {'name': 'RandomRotation'},
        {'name': 'Normalize'},
        {'name': 'ToTensorV2'}
    ]

    Args:
        transform_list (list[dict]): List of dictionaries defining transforms

    Returns:
        A composed albumentations transform

    """
    if not transform_list:
        return A.Compose([])
    
    registry = get_transform_registry()
    compose_list = []
    
    for i, transform_dict in enumerate(transform_list):
        name = None
        try:
            # Make a copy to avoid modifying the original
            transform_spec = transform_dict.copy()
            
            if 'name' not in transform_spec:
                raise TransformConfigError(f"Transform at index {i} missing required 'name' field")
            
            name = transform_spec.pop('name')
            transform_class = registry.get_transform_class(name)
            
            # Validate parameters before construction
            _validate_parameters(transform_class, transform_spec)
            
            # Construct the transform
            transform_instance = transform_class(**transform_spec)
            compose_list.append(transform_instance)
            
        except Exception as e:
            if isinstance(e, TransformConfigError):
                raise
            transform_name = name if name else f"transform at index {i}"
            raise TransformConstructionError(f"Failed to construct {transform_name}: {e}")
    
    return A.Compose(compose_list)