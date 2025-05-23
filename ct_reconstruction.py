import torch
import math
from torch.fft import rfft, irfft
import numpy as np

"""
FBP code directly modified from tomosipo
ref: https://github.com/ahendriksen/ts_algorithms/blob/master/ts_algorithms/fbp.py
"""

def ram_lak(n):
    """Compute Ram-Lak filter in real space

    Computes a real space Ram-Lak filter optimized w.r.t. discretization bias
    introduced if a naive ramp function is used to filter projections in
    reciprocal space. For details, see section 3.3.3 in Kak & Staley,
    "Principles of Computerized Tomographic Imaging", SIAM, 2001.

    :param n: `int`
        Length of the filter.

    :returns:
        Real space Ram-Lak filter of length n.
    :rtype: `torch.tensor`
    """

    filter = torch.zeros(n)
    filter[0] = 0.25
    # even indices are zero
    # for odd indices j, filter[j] equals
    #   -1 / (pi * j) ** 2,          when 2 * j <= n
    #   -1 / (pi * (n - j)) ** 2,    when 2 * j >  n
    odd_indices = torch.arange(1, n, 2)
    cond = 2 * odd_indices > n
    odd_indices[cond] = n - odd_indices[cond]
    filter[1::2] = -1 / (np.pi * odd_indices) ** 2

    return filter


def filter_sino(y, filter=None, padded=True, batch_size=10, overwrite_y=False):
    """Filter sinogram for use in FBP

    :param y: `torch.tensor`
        A three-dimensional tensor in sinogram format (height, num_angles, width).

    :param filter: `torch.tensor` (optional)
        If not specified, the ram-lak filter is used. This should be
        one-dimensional tensor that is as wide as the sinogram `y`.

    :param padded: `bool`
        By default, the reconstruction is zero-padded as it is
        filtered. Padding can be skipped by setting `padded=False`.

    :param batch_size: `int`
        Specifies how many projection images will be filtered at the
        same time. Increasing the batch_size will increase the used
        memory. Computation time can be marginally improved by
        tweaking this parameter.

    :param overwrite_y: `bool`
        Specifies whether to overwrite y with the filtered version
        while running this function. Choose `overwrite_y=False` if you
        still want to use y after calling this function. Choose
        `overwrite_y=True` if you would otherwise run out of memory.

    :returns:
        A sinogram filtered with the provided filter.
    :rtype: `torch.tensor`
    """

    original_width = y.shape[-1]
    if padded:
        filter_width = 2 * original_width
    else:
        filter_width = original_width

    if filter is None:
        # Use Ram-Lak filter by default.
        filter = ram_lak(filter_width).to(y.device)
    elif filter.shape[-1] != filter_width:
        raise ValueError(
            f"Filter is the wrong length. "
            f"Expected length: {filter_width}. "
            f"Got: {filter.shape}. "
            f"Sinogram padding argument is set to {padded}"
        )
    filter_rfft = rfft(filter)

    # Filter the sinogram in batches
    def filter_batch(batch):
        # Compute real FFT using zero-padding of the signal
        batch_rfft = rfft(batch, n=filter_width)
        # Filter the sinogram using complex multiplication:
        batch_rfft *= filter_rfft
        # Invert fourier transform.
        # Make sure inverted data matches the shape of y (for
        # sinograms with odd width).
        batch_filtered = irfft(batch_rfft, n=filter_width)
        # Remove padding
        return batch_filtered[..., :original_width]

    if overwrite_y:
        y_filtered = y
    else:
        y_filtered = torch.empty_like(y)

    for batch_start in range(0, y.shape[1], batch_size):
        batch_end = min(batch_start + batch_size, y.shape[1])
        batch = y[:, batch_start:batch_end, :]
        y_filtered[:, batch_start:batch_end, :] = filter_batch(batch)

    return y_filtered


def fbp(A, y, padded=True, filter=None, batch_size=10, overwrite_y=False):
    """Compute FBP reconstruction

    If `y` is located on GPU, the entire algorithm is executed on a single GPU.

    If `y` is located in RAM (CPU in PyTorch parlance), then only the
    foward and backprojection are executed on GPU.

    The algorithm is explained in detail in [1].

    :param A: `tomosipo.operator`
        The tomographic operator.

    :param y: `torch.tensor`
        A three-dimensional tensor in sinogram format (height, num_angles, width).

    :param padded: `bool`
        By default, the reconstruction is zero-padded as it is
        filtered. Padding can be skipped by setting `padded=False`.

    :param filter: `torch.tensor` (optional)
        If not specified, the ram-lak filter is used. This should be
        one-dimensional tensor that is as wide as the sinogram `y`.

    :param batch_size: `int`
        Specifies how many projection images will be filtered at the
        same time. Increasing the batch_size will increase the used
        memory. Computation time can be marginally improved by
        tweaking this parameter.

    :param overwrite_y: `bool`
        Specifies whether to overwrite y with the filtered version
        while running this function. Choose `overwrite_y=False` if you
        still want to use y after calling this function. Choose
        `overwrite_y=True` if you would otherwise run out of memory.

    :returns:
        A reconstruction computed using the FBP algorithm.

    :rtype: `torch.tensor`

    [1] Zeng, G. L., Revisit of the ramp filter, IEEE Transactions on
    Nuclear Science, 62(1), 131–136 (2015).
    http://dx.doi.org/10.1109/tns.2014.2363776

    """

    y_filtered = filter_sino(y, filter=filter, padded=padded,
                             batch_size=batch_size, overwrite_y=overwrite_y)

    # Backproject the filtered sinogram to obtain a reconstruction
    rec = A.T(y_filtered)

    # Scale result to make sure that fbp(A, A(x)) == x holds at least
    # to some approximation. In limited experiments, this is true for
    # this version of FBP up to 1%.
    # *Note*: For some reason, we do not have to scale with respect to
    # the pixel dimension that is orthogonal to the rotation axis (`u`
    # or horizontal pixel dimension). Hence, we only scale with the
    # other pixel dimension (`v` or vertical pixel dimension).
    # vg, pg = A.astra_compat_vg, A.astra_compat_pg

    pixel_height = 1 #(pg.det_size[0] / pg.det_shape[0])
    voxel_volume = 1 # np.prod(np.array(vg.size / np.array(vg.shape)))
    scaling = (np.pi / A.num_angles) * pixel_height / voxel_volume

    rec *= scaling

    return rec

"""
SIRT code directly modified from tomosipo
ref: https://github.com/ahendriksen/ts_algorithms/blob/master/ts_algorithms/sirt.py
"""

def sirt(A, y, num_iterations, min_constraint=None, max_constraint=None, x_init=None):
    """Execute the SIRT algorithm

    If `y` is located on GPU, the entire algorithm is executed on a single GPU.

    IF `y` is located in RAM (CPU in PyTorch parlance), then only the
    forward and backprojection are executed on GPU.

    :param A: `tomosipo.Operator`
        Projection operator
    :param y: `torch.Tensor`
        Projection data
    :param num_iterations: `int`
        Number of iterations
    :param min_constraint: `float`
        Minimum value enforced at each iteration. Setting to None skips this step.
    :param max_constraint: `float`
        Maximum value enforced at each iteration. Setting to None skips this step.
    :param x_init: `torch.Tensor`
        Initial value for the solution. Setting to None will start with zeros.
        Setting x_init to a previously found solution can be useful to
        continue with more iterations of SIRT.
    :param volume_mask: `torch.Tensor`
        Mask for the reconstruction volume. All voxels outside of the mask will
        be assumed to not contribute to the projection data.
        Setting to None will result in using the whole volume.
    :param projection_mask: `torch.Tensor`
        Mask for the projection data. All pixels outside of the mask will
        be assumed to not contribute to the reconstruction.
        Setting to None will result in using the whole projection data.
    :param progress_bar: `bool`
        Whether to show a progress bar on the command line interface.
        Default: False
    :param callbacks:
        Iterable containing functions or callable objects. Each callback will
        be called every iteration with the current estimate and iteration
        number as arguments. If any callback returns True, the algorithm stops
        after this iteration. This can be used for logging, tracking or
        alternative stopping conditions.
    :returns: `torch.Tensor`
        A reconstruction of the volume using num_iterations iterations of SIRT
    :rtype:

    """
    dev = y.device

    # Compute C
    y_tmp = torch.ones(A.projection_shape, device=dev)
    C = A.T(y_tmp)
    C[C < 1e-8] = math.inf
    C.reciprocal_()
    # Compute R
    x_tmp = torch.ones(A.volume_shape, device=dev)
    R = A(x_tmp)
    R[R < 1e-8] = math.inf
    R.reciprocal_()

    if x_init is None:
        x_cur = torch.zeros(A.volume_shape, device=dev)
    else:
        with torch.cuda.device_of(y):
            x_cur = x_init.clone()

    for iteration in range(num_iterations):
        y_tmp = A(x_cur)
        y_tmp -= y
        y_tmp *= R
        x_tmp = A.T(y_tmp)
        x_tmp *= C
        x_cur -= x_tmp
        if (min_constraint is not None) or (max_constraint is not None):
            x_cur.clamp_(min_constraint, max_constraint)

    return x_cur