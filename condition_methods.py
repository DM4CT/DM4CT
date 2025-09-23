from abc import ABC
import torch
from forward_operators_ct import PoissonNoise

class ConditioningMethod(ABC):
    def __init__(self, operator, noiser, **kwargs):
        self.operator = operator
        self.noiser = noiser

    def project(self, data, noisy_measurement, **kwargs):
        return self.operator.project(data, noisy_measurement, **kwargs)
    
    def grad_and_value(self, x_prev, x_0_hat, measurement, **kwargs):
        Ax = self.operator.forward(x_0_hat, **kwargs)
        difference = measurement-Ax
        # Get smallest non-zero value in the measurement
        min_nonzero = measurement.abs()[measurement.abs() != 0].min()

        # Add epsilon only where measurement == 0
        denominator = measurement.abs().clone()
        denominator[denominator == 0] = min_nonzero

        norm = torch.linalg.norm(difference) / denominator
        norm = norm.mean()
        norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]
             
        return norm_grad, norm

# dps
class PosteriorSampling(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.scale = kwargs.get('scale', 1.0)

    def conditioning(self, x_prev, x_t, x_0_hat, measurement, **kwargs):
        norm_grad, norm = self.grad_and_value(x_prev=x_prev, x_0_hat=x_0_hat, measurement=measurement, **kwargs)
        x_t -= norm_grad * self.scale
        return x_t, norm

#mcg
class ManifoldConstraintGradient(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.scale = kwargs.get('scale', 1.0)
        
    def conditioning(self, x_prev, x_t, x_0_hat, measurement, noisy_measurement, **kwargs):
        # posterior sampling
        norm_grad, norm = self.grad_and_value(x_prev=x_prev, x_0_hat=x_0_hat, measurement=measurement, **kwargs)
        x_t -= norm_grad * self.scale
        
        # projection
        x_t = self.project(x_t, noisy_measurement, **kwargs)
        return x_t, norm

# psld
# implememnted in pipelines as the operation across latent and image space
class PSLD(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.scale = kwargs.get('scale', 1.0)

        
class Resample(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.scale = kwargs.get('scale', 1.0)
        if self.scale is None:
            scale = 0.3
        

class PGDM(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.scale = kwargs.get('scale', 1.0)

    def grad_and_value(self, x_prev, x_0_hat, measurement, **kwargs):
        mat = self.operator.pseudo_inverse(measurement) - self.operator.pseudo_inverse(self.operator.forward(x_0_hat))
        mat_x = (mat.detach() * x_0_hat).sum()
        norm_grad = torch.autograd.grad(mat_x, x_prev)[0] 
             
        return norm_grad, None

    def conditioning(self, x_prev, x_t, x_0_hat, measurement, **kwargs):
        norm_grad, _ = self.grad_and_value(x_prev=x_prev, x_0_hat=x_0_hat, measurement=measurement, **kwargs)
        x_t += norm_grad * self.scale
        return x_t, None
    
class DMPlug(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.scale = kwargs.get('scale', 1.0)


class RedDiff(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.scale = kwargs.get('scale', 1.0)

class HybridReg(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.scale = kwargs.get('scale', 1.0)

class DiffStateGrad(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.scale = kwargs.get('scale', 1.0)
