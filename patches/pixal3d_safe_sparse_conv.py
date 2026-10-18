"""Bounded explicit sparse convolution without Windows Triton compilation."""

import torch

MAX_IM2COL_BYTES = 64 * 1024**2


def explicit_submanifold_conv(feats, neighbors, weight, bias=None):
    out_channels, kd, kh, kw, in_channels = weight.shape
    volume = kd * kh * kw
    chunk_size = max(1, MAX_IM2COL_BYTES // (volume * in_channels * feats.element_size()))
    weight = weight.reshape(out_channels, volume * in_channels).t()
    output = feats.new_empty((feats.shape[0], out_channels))
    for start in range(0, feats.shape[0], chunk_size):
        end = min(start + chunk_size, feats.shape[0])
        indices = neighbors[start:end].reshape(-1).long()
        valid = indices != 0xFFFFFFFF
        columns = feats.new_zeros(((end - start) * volume, in_channels))
        columns[valid] = feats[indices[valid]]
        values = columns.reshape(end - start, volume * in_channels) @ weight
        if bias is not None:
            values = values + bias
        output[start:end] = values
    return output


def sparse_conv3d_forward(module, x):
    from flex_gemm.ops import spconv
    from flex_gemm.ops.spconv.submanifold_conv3d import SubMConv3dFunction

    _, kd, kh, kw, _ = module.weight.shape
    key = f"SubMConv3d_neighbor_cache_{kw}x{kh}x{kd}_dilation{module.dilation}"
    cache = x.get_spatial_cache(key)
    if cache is None:
        previous_algorithm = spconv.ALGORITHM
        spconv.set_algorithm(spconv.Algorithm.EXPLICIT_GEMM)
        try:
            cache = SubMConv3dFunction._compute_neighbor_cache(
                x.coords, torch.Size([*x.shape, *x.spatial_shape]),
                module.kernel_size, module.dilation,
            )
        finally:
            spconv.set_algorithm(previous_algorithm)
        x.register_spatial_cache(key, cache)

    # flex_gemm's explicit algorithm allocates the entire im2col tensor.
    # Chunk the same matrix multiplication to cap that temporary at 64 MiB.
    output = explicit_submanifold_conv(x.feats, cache["neighbor_map"], module.weight, module.bias)
    return x.replace(output)
