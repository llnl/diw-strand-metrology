import albumentations as A
import logging
import os
import yaml
import inspect
from pathlib import Path
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

def _resolve_config_path(config_file: str) -> str:
    """Resolve a config filename to a full path.

    If *config_file* is a bare filename (no directory component) and does not
    exist at the current working directory, attempt to locate it inside the
    bundled ``llnl_ml/configs/`` package directory.

    Args:
        config_file: A path or bare filename such as ``"default_transforms.yaml"``.

    Returns:
        The resolved path as a string.  If the file cannot be found in the
        configs directory the original value is returned unchanged (so that
        downstream code raises the appropriate ``FileNotFoundError``).
    """
    logger = logging.getLogger(__name__)

    # Only attempt resolution when the caller passed a bare filename
    if os.path.basename(config_file) != config_file:
        return config_file

    # Already exists relative to cwd – nothing to resolve
    if os.path.exists(config_file):
        return config_file

    # Try importlib.resources first (works with installed packages)
    try:
        from importlib.resources import files as _files

        config_resource = _files("llnl_ml.configs") / config_file
        if hasattr(config_resource, "is_file") and config_resource.is_file():
            resolved = str(config_resource)
            logger.info(f"Resolved config '{config_file}' from package: {resolved}")
            return resolved
    except Exception as exc:
        logger.debug(f"importlib.resources lookup failed: {exc}")

    # Fallback: resolve relative to this source file
    fallback = Path(__file__).parent.parent / "configs" / config_file
    if fallback.exists():
        resolved = str(fallback)
        logger.info(f"Resolved config '{config_file}' via fallback path: {resolved}")
        return resolved

    return config_file

def build_transforms(config_file: str) -> Tuple[A.Compose, A.Compose]:
    """
    Builds the train and validation/test transform sets from the given config_file.
    The config_file is yaml format and should contain a train and val section with
    each section being a list of augmentation dicts.

    If config_file is a bare filename (no directory separators), it will be resolved
    against the bundled configs directory (llnl_ml/configs/).

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
        config_file: Path to YAML configuration file, or a bare filename to
                     resolve from the bundled configs directory.

    Returns:
        Tuple of (train_transforms, val_transforms)

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file has invalid YAML syntax
        TransformConfigError: If transform specifications are invalid
    """
    config_file = _resolve_config_path(config_file)

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