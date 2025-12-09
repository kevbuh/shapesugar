import torch
import torch.nn as nn
from contextlib import contextmanager
from typing import Any, Iterable, Tuple, Dict, List, Optional, Union

def _shape_of(x: Any):
    """Return a readable shape description for tensors / nested structures."""
    if torch.is_tensor(x):
        return list(x.shape)
    if isinstance(x, (list, tuple)):
        return [_shape_of(i) for i in x]
    if isinstance(x, dict):
        return {k: _shape_of(v) for k, v in x.items()}
    return type(x).__name__

def _short_module_name(m: nn.Module):
    return m.__class__.__name__

def _default_filter(m: nn.Module):
    # Reduce noise: keep layers where shape changes are meaningful.
    return isinstance(m, (
        nn.Conv1d, nn.Conv2d, nn.Conv3d,
        nn.Linear,
        nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
        nn.LayerNorm,
        nn.Embedding,
        nn.MultiheadAttention,
        nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d,
        nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d,
        nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool3d,
        nn.Upsample,
    ))

class ShapeTracer:
    """
    Attach forward hooks to a model and print input/output shapes.
    
    Usage:
        tracer = ShapeTracer(model)
        with tracer:
            _ = model(x)
    """
    def __init__(
        self,
        model: nn.Module,
        module_filter=_default_filter,
        print_fn=print,
        show_leaf_only: bool = True,
        max_depth: Optional[int] = None,
    ):
        self.model = model
        self.module_filter = module_filter
        self.print_fn = print_fn
        self.show_leaf_only = show_leaf_only
        self.max_depth = max_depth
        
        self.handles: List[torch.utils.hooks.RemovableHandle] = []
        self._name_map = dict(model.named_modules())
        self._depth_map = self._build_depth_map()

    def _build_depth_map(self):
        depth = {}
        for name, module in self.model.named_modules():
            if name == "":
                depth[module] = 0
            else:
                depth[module] = name.count(".") + 1
        return depth

    def _should_show(self, module: nn.Module):
        if self.max_depth is not None and self._depth_map.get(module, 0) > self.max_depth:
            return False
        
        # Leaf-only means don't print containers like Sequential
        if self.show_leaf_only:
            has_children = any(True for _ in module.children())
            if has_children:
                return False
        
        return self.module_filter(module)

    def _hook(self, module: nn.Module, inputs: Tuple[Any, ...], output: Any):
        if not self._should_show(module):
            return
        
        name = None
        for n, m in self.model.named_modules():
            if m is module:
                name = n
                break
        
        depth = self._depth_map.get(module, 0)
        indent = "  " * depth
        mod_name = _short_module_name(module)
        label = f"{name}" if name else mod_name

        in_shape = _shape_of(inputs[0]) if len(inputs) == 1 else _shape_of(inputs)
        out_shape = _shape_of(output)

        self.print_fn(f"{indent}{label} [{mod_name}]")
        self.print_fn(f"{indent}  in : {in_shape}")
        self.print_fn(f"{indent}  out: {out_shape}")

    def __enter__(self):
        # Register hooks
        for module in self.model.modules():
            # Avoid hooking the root twice & skip if filter will never pass
            if module is self.model:
                continue
            handle = module.register_forward_hook(self._hook)
            self.handles.append(handle)
        return self

    def __exit__(self, exc_type, exc, tb):
        for h in self.handles:
            h.remove()
        self.handles.clear()
        return False  # don't suppress exceptions