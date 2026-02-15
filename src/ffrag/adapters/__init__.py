from .base import BaseAdapter, AdapterBuildResult
from .rules import GenericMappingAdapter, MappingSpec
from .registry import register_adapter, get_adapter, list_adapters
from .validation import GraphContractValidator, GraphValidationIssue, GraphValidationResult

# Built-in adapter registrations.
register_adapter(
    "generic",
    lambda records, spec, **kwargs: GenericMappingAdapter(records=records, spec=spec, **kwargs),
)

__all__ = [
    "BaseAdapter",
    "AdapterBuildResult",
    "MappingSpec",
    "GenericMappingAdapter",
    "register_adapter",
    "get_adapter",
    "list_adapters",
    "GraphContractValidator",
    "GraphValidationIssue",
    "GraphValidationResult",
]
