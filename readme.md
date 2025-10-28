# shapelang

What would a machine learning kind of like notation or language look like to where we just focus on shapes of transformations?

We can imagine a kind of "shape calculus" or tensor shape language, much like how people use tensor contraction notation in physics or einsum-style notation in ML.

Like Halide, focus on the interface. Algorithm decoupled from the underlying computation. Input and output.

### Basic Convention:
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
- more beautiful syntax