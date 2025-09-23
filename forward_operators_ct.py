import astra
import astra.experimental
import torch
import numpy as np
from ct_reconstruction import fbp, sirt
import torch.nn as nn
import math


class OperatorFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, volume, projector, projection_shape, volume_shape):
            if volume.ndim == 4:
                batch_size = volume.shape[0]

                # vmap solution is not compatiable with astra's direct_FP3D
                # def apply_single(vol):
                #     projection = torch.zeros(projection_shape, dtype=torch.float32, device='cuda')
                #     astra.experimental.direct_FP3D(projector, vol=vol.detach(), proj=projection)
                #     return projection
                
                # projection = torch.zeros((batch_size, *projection_shape), dtype=torch.float32, device='cuda')

                # # Vectorize the function over batch dimension
                # projection = torch.vmap(apply_single)(volume)

                projection = torch.zeros((batch_size, *projection_shape), dtype=torch.float32, device='cuda')
                for i in range(batch_size):  # Process each batch separately
                    astra.experimental.direct_FP3D(projector, vol=volume[i].contiguous().detach(), proj=projection[i])

            elif volume.ndim ==3:
                projection = torch.zeros(projection_shape, dtype=torch.float32, device='cuda')
                astra.experimental.direct_FP3D(projector, vol=volume.detach(), proj=projection)

            else:
                raise NotImplementedError
            
            ctx.save_for_backward(volume)
            ctx.projector = projector  # Save projector for backward computation
            ctx.volume_shape = volume_shape
            return projection
        
        @staticmethod
        def backward(ctx, grad_output):
            volume, = ctx.saved_tensors
            projector = ctx.projector
            volume_shape = ctx.volume_shape

            if volume.ndim==4:
                batch_size = volume.shape[0]
                grad_volume = torch.zeros((batch_size, *volume_shape), dtype=torch.float32, device='cuda')
                for i in range(batch_size):
                    astra.experimental.direct_BP3D(projector, vol=grad_volume[i], proj=grad_output[i].contiguous().detach())

            elif volume.ndim==3:
                grad_volume = torch.zeros(volume_shape, dtype=torch.float32, device='cuda')
                astra.experimental.direct_BP3D(projector, vol=grad_volume, proj=grad_output.detach())
            
            else:
                raise NotImplementedError
        
            return grad_volume, None, None, None # The second 'None' corresponds to the non-trainable projector

class Operator:
    """A linear tomographic projection operator

    An operator describes and computes the projection from a volume onto a
    projection geometry.
    """

    def __init__(
        self,
        volume_geometry,
        projection_geometry):

        super(Operator, self).__init__()
        self.volume_geometry = volume_geometry
        self.projection_geometry = projection_geometry
        self.projector = astra.create_projector('cuda3d', projection_geometry, volume_geometry)
        self.projection_shape = astra.geom_size(projection_geometry)
        self.volume_shape = astra.geom_size(volume_geometry)
        self.num_angles = len(projection_geometry['ProjectionAngles']) if 'ProjectionAngles' in projection_geometry else len(projection_geometry['Vectors'])
        # Compute C
        y_tmp = torch.ones(self.projection_shape, device='cuda')
        C = self.T(y_tmp)
        C[C < 1e-8] = math.inf
        C.reciprocal_()
        # Compute R
        x_tmp = torch.ones(self.volume_shape, device='cuda')
        R = self(x_tmp)
        R[R < 1e-8] = math.inf
        R.reciprocal_()
        self.R = R
        self.C = C

    def __call__(self, volume):
        # projection = torch.zeros(self.projection_shape, dtype=torch.float32, device='cuda')
        # astra.experimental.direct_FP3D(self.projector_id, vol=volume.detach(), proj=projection)
        # return projection
        return OperatorFunction.apply(volume, self.projector, self.projection_shape, self.volume_shape)
    
    def T(self, projection):
        if projection.ndim==4:
            batch_size = projection.shape[0]
            volume = torch.zeros((batch_size, *self.volume_shape), dtype=torch.float32, device='cuda')
            for i in range(batch_size):
                astra.experimental.direct_BP3D(self.projector, vol=volume[i], proj=projection[i].contiguous().detach())
        
        elif projection.ndim==3:
            volume = torch.zeros(self.volume_shape, dtype=torch.float32, device='cuda')
            astra.experimental.direct_BP3D(self.projector, vol=volume, proj=projection.detach())
        
        else:
            raise NotImplementedError
        
        return volume
    
    def forward(self, volume):
        return self.__call__(volume)
    
    def transpose(self, projection):
        return self.T(projection)
    
    def project(self, volume, projection):
        # calculate (I - C * A^T * R* A)x + C A^T y
        return volume - self.C*self.transpose(self.R*self.forward(volume)) + self.C*self.transpose(self.R*projection)

    def pseudo_inverse(self, projection, method='sirt', num_iterations=100, min_constraint=None, max_constraint=None, x_init=None):
        if method == 'sirt':
            if projection.ndim==4:
                batch_size = projection.shape[0]
                volume = torch.zeros((batch_size, *self.volume_shape), dtype=torch.float32, device='cuda')
                for i in range(batch_size):
                    volume[i] = sirt(self, projection[i], num_iterations, min_constraint, max_constraint, x_init)
            
            elif projection.ndim==3:
                volume = sirt(self, projection, num_iterations, min_constraint, max_constraint, x_init)

            else:
                raise NotImplementedError
        
            return volume
        elif method == 'fbp':
            return fbp(self, projection)
        else:
            raise NotImplementedError


class PoissonNoise(nn.Module):
    def __init__(self, transmittance_rate: float, phonton_count: float):
        super().__init__()
        self.transmittance_rate = transmittance_rate
        self.phonton_count = phonton_count

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Apply Poisson noise to measurement (sinogram) in log domain"""

        self.scale_factor = self.cal_attenuation_factor(data, self.transmittance_rate)

        data *= self.scale_factor

        # take exponential - Converts log-attenuation back to transmission
        data = torch.exp(-data)
        data = torch.poisson(data * self.phonton_count)
        data[data==0] = 1

        data = torch.divide(data, self.phonton_count)
        data = -torch.log(data)

        data /= self.scale_factor

        return data
    
    def transmittance(self, sinogram):
        return torch.mean(torch.exp(-sinogram))
    
    def cal_attenuation_factor(self, sinogram, target_transmittance):
        """
        Directly calculates the attenuation factor to achieve a given target transmittance.
        """
        mean_sinogram = torch.mean(sinogram)  # Avoid zero values
        if mean_sinogram == 0:
            return 1  # If sinogram is zero everywhere, scaling does nothing
        return -np.log(target_transmittance) / mean_sinogram

class NoNoise(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Noiseless forward operator"""
        return data

class PoissonNoiseRing(PoissonNoise):
    def __init__(self, transmittance_rate: float, phonton_count: float, bad_pixel_ratio:float, scale=1, random_seed=123):
        super().__init__(transmittance_rate, phonton_count)
        self.bad_pixel_ratio = bad_pixel_ratio
        self.scale = scale
        self.random_seed = random_seed  # Store for potential re-use

        # Set random seeds
        torch.manual_seed(random_seed)  

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Apply Poisson noise to measurement (sinogram) in log domain"""

        if data.is_cuda:
            torch.cuda.manual_seed_all(self.random_seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        self.scale_factor = self.cal_attenuation_factor(data, self.transmittance_rate)

        data *= self.scale_factor

        # take exponential - Converts log-attenuation back to transmission
        data = torch.exp(-data)
        data = torch.poisson(data * self.phonton_count)
        data[data==0] = 1

        data = torch.divide(data, self.phonton_count)
        data = -torch.log(data)

        data /= self.scale_factor

        data_std = data.std()

        # add pertubation for ring artifacts
        B, A, W = data.shape  # Batch, Angles, Width

        # Create a random mask for bad pixels
        num_bad_pixels = int(B * W * self.bad_pixel_ratio)
        mask = torch.zeros(B * W, device=data.device)
        mask[:num_bad_pixels] = 1
        mask = mask[torch.randperm(B * W)].reshape(B, W)  # Shuffle and reshape

        # Generate Gaussian noise
        noise = torch.randn(B, W, device=data.device) * self.scale * data_std

        # Apply noise only to masked "bad pixels" across all angles
        data = data.clone()  # Avoid modifying input in-place
        data += noise[:, None, :] * mask[:, None, :]  # Broadcast across angles

        return data
    

