# EZDIM

Ever broadcasted to the wrong dimension? It is clear that ML libraries need to surface tensor dimensions to the type level, but currently there is no easy way to do so. The ML communituy deserves a good solution, one that doesn't require weird libraries. We need something convenient to use. Ideally, we should be able to hover over variables and see the shape. This will make debugging much easier. 

PyTorch has named tensors, but almost nobody uses them. they have been labeled as "abandonware" by some PyTorch contributors.

I'm not the only one with these frustrations, see reactions from others in the following [thread](https://x.com/syntacrobat/status/1984470895000199447):

```
> Mom, can we have a type system?
> No, we have a type system at home
The type system at home:
# (B, C, T)
```

```
literally how do ML people survive without lifting tensor dimensions into the type system? isn't that like the number one thing youd immediately want
```

```
This shouldn't even need dependent types tbh, just a regular static typesystem and marking specific dimensions as dynamic.
```

Python [PEP 0646: Variadic Generics](https://peps.python.org/pep-0646/) proposed a solution to this. It is now known as [TypeVarTuple](https://typing.python.org/en/latest/spec/generics.html#typevartuple) with an example shown below:

```
from typing import NewType

class Array[*Shape]: ...

Height = NewType('Height', int)
Width = NewType('Width', int)
x: Array[Height, Width] = Array()

Time = NewType('Time', int)
Batch = NewType('Batch', int)
y: Array[Batch, Height, Width] = Array()
z: Array[Time, Batch, Height, Width] = Array()
```

# MISC

#### PyTorch Shape Tracer

Allows you to see the input output dimensions for each layer. Defined in ```ShapeTracer.py```.

```python
import torch
import torch.nn as nn
from ShapeTracer import ShapeTracer

class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.block = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 56 * 56, 10),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.block(x)
        x = self.head(x)
        return x

model = TinyCNN()
x = torch.randn(4, 3, 224, 224)

tracer = ShapeTracer(model)
with tracer:
    y = model(x)
```

with output of 

```
stem.0 [Conv2d]
    in : [4, 3, 224, 224]
    out: [4, 16, 224, 224]
stem.2 [MaxPool2d]
    in : [4, 16, 224, 224]
    out: [4, 16, 112, 112]
block.0 [Conv2d]
    in : [4, 16, 112, 112]
    out: [4, 32, 112, 112]
block.2 [MaxPool2d]
    in : [4, 32, 112, 112]
    out: [4, 32, 56, 56]
head.1 [Linear]
    in : [4, 100352]
    out: [4, 10]
```

### shapelang

What would a machine learning kind of like notation or language look like to where we just focus on shapes of transformations?

We can imagine a kind of "shape calculus" or tensor shape language, much like how people use tensor contraction notation in physics or einsum-style notation in ML.

Like Halide, focus on the interface. Algorithm decoupled from the underlying computation. Input and output.

Basic Convention:
- Use square brackets for shape: [B, D] (e.g. batch, feature dim)
- Arrow -> for transformation
- Named axes when useful: [batch B, dim D]
- Use * or ... for variable/unspecified dimensions

| Operation | Shape Notation | Meaning |
|---|---|---|
| Input vector | `[D]` | A 1D tensor with `D` features |
| Batch of vectors | `[B, D]` | Batch size `B`, each with `D` features |
| Linear layer | `[B, D] -> [B, H]` | Projects `D` dims to `H` dims |
| Activation (ReLU) | `[B, H] -> [B, H]` | Element-wise, shape unchanged |
| Softmax over classes | `[B, C] -> [B, C]` | Softmax over last axis (classes) |
| Flatten | `[B, C, H, W] -> [B, C*H*W]` | Collapse spatial dims |
| CNN conv layer | `[B, C, H, W] -> [B, C', H', W']` | Convolution output with new channel/spatial |
| Attention (QK^T) | `[B, H, L, D] x [B, H, D, L] -> [B, H, L, L]` | Dot-product attention scores |
| Masking or broadcasting | `[B, 1, L] + [B, H, L] -> [B, H, L]` | Broadcast mask across heads |


### Inspo
- Decoupling Algorithms from the Organization of Computation for High Performance Image Processing

### Misc
- https://einops.rocks/pytorch-examples.html
- https://github.com/google/trax

### TODO
- add pytorch/jax/tinygrad backends
- wider support of common operations
- unit tests
- more beautiful syntax -->