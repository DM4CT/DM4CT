import inspect
import torch
import diffusers
import numpy as np
from math import sqrt
from diffusers import DDPMPipeline, DDPMScheduler, DDIMScheduler, LDMPipeline
from typing import List, Optional, Tuple, Union
from diffusers.models import UNet2DModel
from diffusers.schedulers import DDPMScheduler,  DDIMScheduler
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import ImagePipelineOutput
from schedulers import DDIMSchedulerReSampler



class DDPMPipelineDPS(DDPMPipeline):
    def __init__(
        self,
        unet: UNet2DModel,
        scheduler: DDPMScheduler,
    ):
        super().__init__(unet=unet, scheduler=scheduler)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        # Load components using the base class method
        pipeline = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        
        # Return an instance of the subclass
        return cls(unet=pipeline.unet, scheduler=pipeline.scheduler)


    # @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 1000,
        output_type: Optional[str] = "pil",
        measurement = None,
        return_dict: bool = True,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            batch_size (`int`, *optional*, defaults to 1):
                The number of images to generate.
            generator (`torch.Generator`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            num_inference_steps (`int`, *optional*, defaults to 1000):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.

        Example:

        ```py
        >>> from diffusers import DDPMPipeline

        >>> # load model and scheduler
        >>> pipe = DDPMPipeline.from_pretrained("google/ddpm-cat-256")

        >>> # run pipeline in inference (sample random noise and denoise)
        >>> image = pipe().images[0]

        >>> # save image
        >>> image.save("ddpm_generated_image.png")
        ```

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.ImagePipelineOutput`] is returned, otherwise a `tuple` is
                returned where the first element is a list with the generated images
        """
        if self.measurement_condition is None:
            raise ValueError("Measurement condition is not set.")

        # Sample gaussian noise to begin loop
        if isinstance(self.unet.config.sample_size, int):
            image_shape = (
                batch_size,
                self.unet.config.in_channels,
                self.unet.config.sample_size,
                self.unet.config.sample_size,
            )
        else:
            image_shape = (batch_size, self.unet.config.in_channels, *self.unet.config.sample_size)

        if self.device.type == "mps":
            # randn does not work reproducibly on mps
            image = randn_tensor(image_shape, generator=generator, dtype=self.unet.dtype)
            image = image.to(self.device)
        else:
            image = randn_tensor(image_shape, generator=generator, device=self.device, dtype=self.unet.dtype)

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)

        for t in self.progress_bar(self.scheduler.timesteps):
            # require grad for measurement condition
            image = image.requires_grad_()

            # 1. predict noise model_output
            model_output = self.unet(image, t).sample

            # 2. compute previous image: x_t -> x_t-1
            out = self.scheduler.step(model_output, t, image, generator=generator)

            image, distance = self.measurement_condition.conditioning(x_t=out.prev_sample,
                                      measurement=measurement,
                                      x_prev=image,
                                      x_0_hat=out.pred_original_sample)

            image = image.detach_()

        image = image.cpu().numpy()

        if output_type == "pil":
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)


class DDPMPipelineMCG(DDPMPipeline):
    def __init__(
        self,
        unet: UNet2DModel,
        scheduler: DDPMScheduler,
    ):
        super().__init__(unet=unet, scheduler=scheduler)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        # Load components using the base class method
        pipeline = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        
        # Return an instance of the subclass
        return cls(unet=pipeline.unet, scheduler=pipeline.scheduler)

    def q_sample(self, x_start, t):
        """
        Applies forward diffusion to x_start by adding noise based on timestep t.
        """
        noise = torch.randn_like(x_start)
        coef1 = self.scheduler.alphas_cumprod[t].sqrt().view(-1, 1, 1, 1).to(x_start.device)
        coef2 = (1 - self.scheduler.alphas_cumprod[t]).sqrt().view(-1, 1, 1, 1).to(noise.device)

        return coef1 * x_start + coef2 * noise


    # @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 1000,
        output_type: Optional[str] = "pil",
        measurement = None,
        return_dict: bool = True,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            batch_size (`int`, *optional*, defaults to 1):
                The number of images to generate.
            generator (`torch.Generator`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            num_inference_steps (`int`, *optional*, defaults to 1000):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.

        Example:

        ```py
        >>> from diffusers import DDPMPipeline

        >>> # load model and scheduler
        >>> pipe = DDPMPipeline.from_pretrained("google/ddpm-cat-256")

        >>> # run pipeline in inference (sample random noise and denoise)
        >>> image = pipe().images[0]

        >>> # save image
        >>> image.save("ddpm_generated_image.png")
        ```

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.ImagePipelineOutput`] is returned, otherwise a `tuple` is
                returned where the first element is a list with the generated images
        """
        if self.measurement_condition is None:
            raise ValueError("Measurement condition is not set.")

        # Sample gaussian noise to begin loop
        if isinstance(self.unet.config.sample_size, int):
            image_shape = (
                batch_size,
                self.unet.config.in_channels,
                self.unet.config.sample_size,
                self.unet.config.sample_size,
            )
        else:
            image_shape = (batch_size, self.unet.config.in_channels, *self.unet.config.sample_size)

        if self.device.type == "mps":
            # randn does not work reproducibly on mps
            image = randn_tensor(image_shape, generator=generator, dtype=self.unet.dtype)
            image = image.to(self.device)
        else:
            image = randn_tensor(image_shape, generator=generator, device=self.device, dtype=self.unet.dtype)

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)

        for t in self.progress_bar(self.scheduler.timesteps):
            # require grad for measurement condition
            image = image.requires_grad_()

            # 1. predict noise model_output
            model_output = self.unet(image, t).sample

            # 2. compute previous image: x_t -> x_t-1
            out = self.scheduler.step(model_output, t, image, generator=generator)

            noisy_measurement = self.q_sample(measurement, t)

            image, distance = self.measurement_condition.conditioning(x_t=out.prev_sample,
                                      measurement=measurement,
                                      noisy_measurement=noisy_measurement,
                                      x_prev=image,
                                      x_0_hat=out.pred_original_sample)

            image = image.detach_()

        image = image.cpu().numpy()

        if output_type == "pil":
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()            
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)


class LDMPipelinePSLD(LDMPipeline):
    def __init__(
        self,
        vqvae, 
        unet, 
        scheduler:DDPMScheduler
    ):
        super().__init__(vqvae=vqvae,
                        unet=unet,
                        scheduler=scheduler)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        # Load components using the base class method
        pipeline = super().from_pretrained(pretrained_model_name_or_path, **kwargs)

        # Override scheduler with DDPM
        ddpm_scheduler = DDPMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler")
        
        # Return an instance of the subclass
        return cls(vqvae=pipeline.vqvae,
            unet=pipeline.unet,
            scheduler=ddpm_scheduler)


    # @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        eta: float = 0.0,
        num_inference_steps: int = 50,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        gamma: float = 0.9,
        omega: float = 0.1,
        measurement = None,
        **kwargs,
    ) -> Union[Tuple, ImagePipelineOutput]:
        r"""
        The call function to the pipeline for generation.

        Args:
            batch_size (`int`, *optional*, defaults to 1):
                Number of images to generate.
            generator (`torch.Generator`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.

        Example:

        ```py
        >>> from diffusers import LDMPipeline

        >>> # load model and scheduler
        >>> pipe = LDMPipeline.from_pretrained("CompVis/ldm-celebahq-256")

        >>> # run pipeline in inference (sample random noise and denoise)
        >>> image = pipe().images[0]
        ```

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.ImagePipelineOutput`] is returned, otherwise a `tuple` is
                returned where the first element is a list with the generated images
        """

        if self.measurement_condition is None:
            raise ValueError("Measurement condition is not set.")

        latents = randn_tensor(
            (batch_size, self.unet.config.in_channels, self.unet.config.sample_size, self.unet.config.sample_size),
            generator=generator,
        )
        latents = latents.to(self.device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma

        self.scheduler.set_timesteps(num_inference_steps)

        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())

        extra_kwargs = {}
        if accepts_eta:
            extra_kwargs["eta"] = eta

        for t in self.progress_bar(self.scheduler.timesteps):
            latents = latents.requires_grad_()
            latent_model_input = self.scheduler.scale_model_input(latents, t)
            # predict the noise residual
            noise_prediction = self.unet(latent_model_input, t).sample

            # compute the previous noisy sample x_t -> x_t-1
            out = self.scheduler.step(noise_prediction, t, latents, **extra_kwargs)

            latents_prev = out.prev_sample
            pred_original_latents = out.pred_original_sample

            # adjust latents with inverse of vae scale
            pred_original_latents = pred_original_latents / self.vqvae.config.scaling_factor
            # decode the image latents with the VAE
            pred_original_image = self.vqvae.decode(pred_original_latents).sample

            # measurement error
            # measurement_pred_original = self.measurement_condition.noiser(self.measurement_condition.operator.forward(pred_original_image))
            measurement_pred_original = self.measurement_condition.operator.forward(pred_original_image)
            # Get smallest non-zero value in the measurement
            min_nonzero = measurement.abs()[measurement.abs() != 0].min()

            # Add epsilon only where measurement == 0
            denominator = measurement.abs().clone()
            denominator[denominator == 0] = min_nonzero

            measurement_difference = torch.linalg.norm(measurement_pred_original - measurement) / denominator
            measurement_difference = measurement_difference.mean()

            projected_pred_original = self.measurement_condition.operator.project(pred_original_image, measurement_pred_original)

            latents_projected_pred_original = self.vqvae.encode(projected_pred_original).latents
            latents_projected_pred_original = latents_projected_pred_original * self.vqvae.config.scaling_factor
            latents_error = torch.linalg.norm(latents_projected_pred_original - pred_original_latents)

            total_error = measurement_difference * omega + latents_error  * gamma 

            grad = torch.autograd.grad(total_error, latents)[0]

            latents = latents_prev - grad

            latents = latents.detach_()


        # adjust latents with inverse of vae scale
        latents = latents / self.vqvae.config.scaling_factor
        # decode the image latents with the VAE
        image = self.vqvae.decode(latents).sample

        image = image.detach_()

        image = image.cpu().numpy()

        if output_type == "pil":
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)


class LDMPipelineReSample(LDMPipeline):
    def __init__(
        self,
        vqvae, 
        unet, 
        scheduler:DDIMSchedulerReSampler
    ):
        super().__init__(vqvae=vqvae,
                        unet=unet,
                        scheduler=scheduler)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        # Load components using the base class method
        pipeline = super().from_pretrained(pretrained_model_name_or_path, **kwargs)

        # Override scheduler with DDPM
        ddim_scheduler = DDIMSchedulerReSampler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler")
        
        # Return an instance of the subclass
        return cls(vqvae=pipeline.vqvae,
            unet=pipeline.unet,
            scheduler=ddim_scheduler)


    # @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        eta: float = 0.0,
        num_inference_steps: int = 50,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        measurement = None,
        inter_timesteps = 5,
        pixel_lr=1e-2,
        latent_lr=5e-3,
        pixel_maxiters=2000,
        latent_maxiters=500,
        pixel_eps=1e-3,
        latent_eps=1e-3,
        **kwargs,
    ) -> Union[Tuple, ImagePipelineOutput]:
        r"""
        The call function to the pipeline for generation.

        Args:
            batch_size (`int`, *optional*, defaults to 1):
                Number of images to generate.
            generator (`torch.Generator`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.

        Example:

        ```py
        >>> from diffusers import LDMPipeline

        >>> # load model and scheduler
        >>> pipe = LDMPipeline.from_pretrained("CompVis/ldm-celebahq-256")

        >>> # run pipeline in inference (sample random noise and denoise)
        >>> image = pipe().images[0]
        ```

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.ImagePipelineOutput`] is returned, otherwise a `tuple` is
                returned where the first element is a list with the generated images
        """

        if self.measurement_condition is None:
            raise ValueError("Measurement condition is not set.")

        latents = randn_tensor(
            (batch_size, self.unet.config.in_channels, self.unet.config.sample_size, self.unet.config.sample_size),
            generator=generator,
        )
        latents = latents.to(self.device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma

        self.scheduler.set_timesteps(num_inference_steps)

        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())

        extra_kwargs = {}
        if accepts_eta:
            extra_kwargs["eta"] = eta

        for t in self.progress_bar(self.scheduler.timesteps):
            latents = latents.requires_grad_()
            # resample specific
            index = t // 2
            a_t = self.scheduler.alphas_cumprod[index]
            a_prev = self.scheduler.alphas_cumprod[index - 1] if index > 0 else torch.tensor(1.0, device=a_t.device)

            latent_model_input = self.scheduler.scale_model_input(latents, t)
            # predict the noise residual
            noise_prediction = self.unet(latent_model_input, t).sample

            # compute the previous noisy sample x_t -> x_t-1
            out = self.scheduler.step(noise_prediction, t, latents, **extra_kwargs)

            latents_prev = out.prev_sample
            pred_original_latents = out.pred_original_sample
            pseudo_original_latents = out.pseudo_original_sample

            # adjust latents with inverse of vae scale
            pseudo_original_latents = pseudo_original_latents / self.vqvae.config.scaling_factor
            # decode the image latents with the VAE
            pseudo_original_image = self.vqvae.decode(pseudo_original_latents).sample

            # conditioning here, original condition method has no access of the network/pipeline
       
            Ax = self.measurement_condition.operator.forward(pseudo_original_image)
            difference = measurement-Ax
            # Get smallest non-zero value in the measurement
            min_nonzero = measurement.abs()[measurement.abs() != 0].min()

            # Add epsilon only where measurement == 0
            denominator = measurement.abs().clone()
            denominator[denominator == 0] = min_nonzero

            norm = torch.linalg.norm(difference) / denominator
            norm = norm.mean()
            norm_grad = torch.autograd.grad(outputs=norm, inputs=latents)[0]
            
            # scale is a_t*0.5
            latents = latents_prev - norm_grad*a_t*0.5

            # Instantiating time-travel parameters, resample specfic
            splits = 3 
            index_split = num_inference_steps // splits

            # Performing time-travel if in selected indices
            if index <= (num_inference_steps - index_split) and index > 0:   
                latents_cp = latents.detach().clone()

                # Performing only every 10 steps (or so)
                if index % 10 == 0 :  
                    for k in range(num_inference_steps-index-1, min(num_inference_steps-index-1+inter_timesteps, num_inference_steps-1)):
                        step_ = self.scheduler.timesteps[k+1]
                        index_ = num_inference_steps - k - 1

                        # Obtain x_{t-k} resample specific
                        # predict the noise residual
                        noise_prediction = self.unet(latents, step_).sample

                        # compute the previous noisy sample x_t -> x_t-1
                        out = self.scheduler.step(noise_prediction, step_, latents, **extra_kwargs)

                        latents_prev = out.prev_sample.detach()
                        pred_original_latents = out.pred_original_sample.detach()
                        pseudo_original_latents = out.pseudo_original_sample.detach()
                        latents = latents_prev
                    
                    # Resample sigma scheduling
                    if index >= 0:
                        sigma = 40*(1 - a_prev) / (1 - a_t) * (1 - a_t / a_prev)  
                    else:
                        sigma = 0.5

                    # Pixel-based optimization for second stage
                    if index >= index_split: 
                         # Enforcing consistency via pixel-based optimization
                         pseudo_original_latents = pseudo_original_latents.detach()
                         # adjust latents with inverse of vae scale
                         pseudo_original_latents = pseudo_original_latents / self.vqvae.config.scaling_factor
                         # decode the image latents with the VAE
                         pseudo_original_image = self.vqvae.decode(pseudo_original_latents).sample

                         opt_image = self.pixel_optimization(measurement, pseudo_original_image, self.measurement_condition.operator.forward, eps=pixel_eps, max_iters=pixel_maxiters, lr=pixel_lr)
                         opt_image = opt_image.detach()
                         opt_latents = self.vqvae.encode(opt_image).latents.detach()

                         latents = self.stochastic_resample(pseudo_x0=opt_latents, x_t=latents_cp, a_t=a_prev, sigma=sigma)
                         latents = latents.requires_grad_()
                    # Latent-based optimization for third stage
                    elif index < index_split:
                        pseudo_original_latents, _ = self.latent_optimization(measurement=measurement,
                                                             z_init=pseudo_original_latents.detach(),
                                                             operator_fn=self.measurement_condition.operator.forward,
                                                             lr=latent_lr, max_iters=latent_maxiters, eps=latent_eps)
                        sigma = 40 * (1-a_prev)/(1 - a_t) * (1 - a_t / a_prev)
                        latents = self.stochastic_resample(pseudo_x0=pseudo_original_latents, x_t=latents_cp, a_t=a_prev, sigma=sigma)   

        # adjust latents with inverse of vae scale
        pseudo_original_latents = pseudo_original_latents / self.vqvae.config.scaling_factor
        # decode the image latents with the VAE
        image = self.vqvae.decode(pseudo_original_latents).sample

        image = image.detach_()

        image = image.cpu().numpy()

        if output_type == "pil":
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()            
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)

    def pixel_optimization(self, measurement, x_prime, operator_fn, eps=1e-3, max_iters=2000, lr=1e-2):
        """
        Function to compute argmin_x ||y - A(x)||_2^2

        Arguments:
            measurement:           Measurement vector y in y=Ax+n.
            x_prime:               Estimation of \hat{x}_0 using Tweedie's formula
            operator_fn:           Operator to perform forward operation A(.)
            eps:                   Tolerance error
            max_iters:             Maximum number of GD iterations
        """

        loss = torch.nn.MSELoss() # MSE loss

        opt_var = x_prime.detach().clone()
        opt_var = opt_var.requires_grad_()
        optimizer = torch.optim.AdamW([opt_var], lr=lr) # Initializing optimizer
        measurement = measurement.detach() # Need to detach for weird PyTorch reasons

        # Training loop

        for _ in range(max_iters):
            optimizer.zero_grad()
            
            measurement_loss = loss(measurement, operator_fn( opt_var ) ) 
            
            measurement_loss.backward() # Take GD step
            optimizer.step()

            # Convergence criteria
            if measurement_loss < eps**2: # needs tuning according to noise level for early stopping
                break

        return opt_var

    def latent_optimization(self, measurement, z_init, operator_fn, eps=1e-3, max_iters=500, lr=5e-3):

        """
        Function to compute argmin_z ||y - A( D(z) )||_2^2

        Arguments:
            measurement:           Measurement vector y in y=Ax+n.
            z_init:                Starting point for optimization
            operator_fn:           Operator to perform forward operation A(.)
            eps:                   Tolerance error
            max_iters:             Maximum number of GD iterations
        
        Optimal parameters seem to be at around 500 steps, 200 steps for inpainting.

        """

        # Base case
        if not z_init.requires_grad:
            z_init = z_init.requires_grad_()

        loss = torch.nn.MSELoss() # MSE loss
        optimizer = torch.optim.AdamW([z_init], lr=lr) # Initializing optimizer ###change the learning rate
        measurement = measurement.detach() # Need to detach for weird PyTorch reasons

        # Training loop
        init_loss = 0
        losses = []
        
        for itr in range(max_iters):
            optimizer.zero_grad()
            output = loss(measurement, operator_fn(self.vqvae.decode( z_init ).sample))          

            if itr == 0:
                init_loss = output.detach().clone()
                
            output.backward() # Take GD step
            optimizer.step()
            cur_loss = output.detach().cpu().numpy() 
            
            # Convergence criteria

            if itr < 200: # may need tuning for early stopping
                losses.append(cur_loss)
            else:
                losses.append(cur_loss)
                if losses[0] < cur_loss:
                    break
                else:
                    losses.pop(0)
                    
            if cur_loss < eps**2:  # needs tuning according to noise level for early stopping
                break


        return z_init, init_loss       

    def stochastic_resample(self, pseudo_x0, x_t, a_t, sigma):
        """
        Function to resample x_t based on ReSample paper.
        """
        device = self.unet.device
        noise = torch.randn_like(pseudo_x0, device=device)
        return (sigma * a_t.sqrt() * pseudo_x0 + (1 - a_t) * x_t)/(sigma + 1 - a_t) + noise * torch.sqrt(1/(1/sigma + 1/(1-a_t)))


class DDPMPipelinePGDM(DDPMPipeline):
    def __init__(
        self,
        unet: UNet2DModel,
        scheduler: DDIMScheduler,
    ):
        super().__init__(unet=unet, scheduler=scheduler)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        # Load components using the base class method
        pipeline = super().from_pretrained(pretrained_model_name_or_path, **kwargs)

        # Replace the scheduler with DDIMScheduler
        ddim_scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
        
        # Return an instance of the subclass
        return cls(unet=pipeline.unet, scheduler=ddim_scheduler)


    # @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 100,
        output_type: Optional[str] = "pil",
        measurement = None,
        return_dict: bool = True,
        scale: float = 1.0,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            batch_size (`int`, *optional*, defaults to 1):
                The number of images to generate.
            generator (`torch.Generator`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            num_inference_steps (`int`, *optional*, defaults to 1000):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.

        Example:

        ```py
        >>> from diffusers import DDPMPipeline

        >>> # load model and scheduler
        >>> pipe = DDPMPipeline.from_pretrained("google/ddpm-cat-256")

        >>> # run pipeline in inference (sample random noise and denoise)
        >>> image = pipe().images[0]

        >>> # save image
        >>> image.save("ddpm_generated_image.png")
        ```

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.ImagePipelineOutput`] is returned, otherwise a `tuple` is
                returned where the first element is a list with the generated images
        """
        if self.measurement_condition is None:
            raise ValueError("Measurement condition is not set.")

        # Sample gaussian noise to begin loop
        if isinstance(self.unet.config.sample_size, int):
            image_shape = (
                batch_size,
                self.unet.config.in_channels,
                self.unet.config.sample_size,
                self.unet.config.sample_size,
            )
        else:
            image_shape = (batch_size, self.unet.config.in_channels, *self.unet.config.sample_size)

        if self.device.type == "mps":
            # randn does not work reproducibly on mps
            image = randn_tensor(image_shape, generator=generator, dtype=self.unet.dtype)
            image = image.to(self.device)
        else:
            image = randn_tensor(image_shape, generator=generator, device=self.device, dtype=self.unet.dtype)

        if image.ndim==4 and measurement.ndim==3:
            measurement = measurement.unsqueeze(0)

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)

        # pgdm specific, already mix pseudo inverse into initial noise
        pseduo_inv_x0 = self.measurement_condition.operator.pseudo_inverse(measurement)
        pseduo_inv_x0 = pseduo_inv_x0.to(image.dtype).to(image.device)
        alpha_prod_t = self.scheduler.alphas_cumprod[self.scheduler.timesteps[0]] 
        image = alpha_prod_t.sqrt() * pseduo_inv_x0 + (1 - alpha_prod_t) * image

        for t in self.progress_bar(self.scheduler.timesteps):
            # require grad for measurement condition
            image = image.requires_grad_()

            # 1. predict noise model_output
            model_output = self.unet(image, t).sample

            # 2. compute previous image: x_t -> x_t-1
            out = self.scheduler.step(model_output, t, image, generator=generator)

            prev_timestep = t - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps

            coeff = self.scheduler.alphas_cumprod[prev_timestep].sqrt() * self.scheduler.alphas_cumprod[t].sqrt() * scale

            self.measurement_condition.scale = coeff

            image, distance = self.measurement_condition.conditioning(x_t=out.prev_sample,
                                      measurement=measurement,
                                      x_prev=image,
                                      x_0_hat=out.pred_original_sample)

            # image = image + coeff * grad

            image = image.detach_()

        image = image.cpu().numpy()

        if output_type == "pil":
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()            
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)


    

class DDPMPipelineDMPlug(DDPMPipeline):
    def __init__(
        self,
        unet: UNet2DModel,
        scheduler: DDIMScheduler,
    ):
        super().__init__(unet=unet, scheduler=scheduler)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        # Load components using the base class method
        pipeline = super().from_pretrained(pretrained_model_name_or_path, **kwargs)

        # Override scheduler with DDPM
        ddim_scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
        
        # Return an instance of the subclass
        return cls(unet=pipeline.unet, scheduler=ddim_scheduler)


    # @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 1000,
        output_type: Optional[str] = "pil",
        measurement = None,
        epochs = 1000,
        lr=1e-2,
        criterion = torch.nn.MSELoss(),
        return_dict: bool = True,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            batch_size (`int`, *optional*, defaults to 1):
                The number of images to generate.
            generator (`torch.Generator`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            num_inference_steps (`int`, *optional*, defaults to 1000):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.

        Example:

        ```py
        >>> from diffusers import DDPMPipeline

        >>> # load model and scheduler
        >>> pipe = DDPMPipeline.from_pretrained("google/ddpm-cat-256")

        >>> # run pipeline in inference (sample random noise and denoise)
        >>> image = pipe().images[0]

        >>> # save image
        >>> image.save("ddpm_generated_image.png")
        ```

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.ImagePipelineOutput`] is returned, otherwise a `tuple` is
                returned where the first element is a list with the generated images
        """
        if self.measurement_condition is None:
            raise ValueError("Measurement condition is not set.")

        # Sample gaussian noise to begin loop
        if isinstance(self.unet.config.sample_size, int):
            image_shape = (
                batch_size,
                self.unet.config.in_channels,
                self.unet.config.sample_size,
                self.unet.config.sample_size,
            )
        else:
            image_shape = (batch_size, self.unet.config.in_channels, *self.unet.config.sample_size)

        if self.device.type == "mps":
            # randn does not work reproducibly on mps
            start_image = randn_tensor(image_shape, generator=generator, dtype=self.unet.dtype)
            start_image = start_image.to(self.device)
        else:
            start_image = randn_tensor(image_shape, generator=generator, device=self.device, dtype=self.unet.dtype)
        
        start_image = start_image.requires_grad_()

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)

        optimizer = torch.optim.Adam([start_image], lr=lr)

        for i in range(epochs):
            self.unet.eval()
            optimizer.zero_grad()
            

            for t in self.progress_bar(self.scheduler.timesteps):
                if t == self.scheduler.timesteps[0]:
                    # 1. predict noise model_output
                    model_output = self.unet(start_image, t).sample
                    # 2. compute previous image: x_t -> x_t-1
                    image = self.scheduler.step(model_output, t, start_image, generator=generator).prev_sample
                else:
                    # 1. predict noise model_output
                    model_output = self.unet(image, t).sample
                    # 2. compute previous image: x_t -> x_t-1
                    image = self.scheduler.step(model_output, t, image, generator=generator).prev_sample

            image = torch.clamp(image, -1, 1)
            loss = criterion(self.measurement_condition.operator.forward(image), measurement)
            loss.backward()
            optimizer.step()

        image = image.detach_()
        image = image.cpu().numpy()

        if output_type == "pil":
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)


class DDPMPipelineRedDiff(DDPMPipeline):
    def __init__(
        self,
        unet: UNet2DModel,
        scheduler: DDIMScheduler,
    ):
        super().__init__(unet=unet, scheduler=scheduler)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        # Load components using the base class method
        pipeline = super().from_pretrained(pretrained_model_name_or_path, **kwargs)

        # Replace the scheduler with DDIMScheduler
        ddim_scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
        
        # Return an instance of the subclass
        return cls(unet=pipeline.unet, scheduler=ddim_scheduler)


    # @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 100,
        output_type: Optional[str] = "pil",
        measurement = None,
        return_dict: bool = True,
        sigma =0,
        loss_measurement_weight = 1.0,
        loss_noise_weight = 1.0,
        lr=0.25,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            batch_size (`int`, *optional*, defaults to 1):
                The number of images to generate.
            generator (`torch.Generator`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            num_inference_steps (`int`, *optional*, defaults to 1000):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of sl        if output_type == "pil":
            image = self.numpy_to_pil(image)ower inference.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.

        Example:

        ```py
        >>> from diffusers import DDPMPipeline

        >>> # load model and scheduler
        >>> pipe = DDPMPipeline.from_pretrained("google/ddpm-cat-256")

        >>> # run pipeline in inference (sample random noise and denoise)
        >>> image = pipe().images[0]

        >>> # save image
        >>> image.save("ddpm_generated_image.png")
        ```

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.ImagePipelineOutput`] is returned, otherwise a `tuple` is
                returned where the first element is a list with the generated images
        """
        if self.measurement_condition is None:
            raise ValueError("Measurement condition is not set.")

        # Sample gaussian noise to begin loop
        if isinstance(self.unet.config.sample_size, int):
            image_shape = (
                batch_size,
                self.unet.config.in_channels,
                self.unet.config.sample_size,
                self.unet.config.sample_size,
            )
        else:
            image_shape = (batch_size, self.unet.config.in_channels, *self.unet.config.sample_size)

        if self.device.type == "mps":
            # randn does not work reproducibly on mps
            image = randn_tensor(image_shape, generator=generator, dtype=self.unet.dtype)
            image = image.to(self.device)
        else:
            image = randn_tensor(image_shape, generator=generator, device=self.device, dtype=self.unet.dtype)

        if image.ndim==4 and measurement.ndim==3:
            measurement = measurement.unsqueeze(0)

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)

        # reddiff specific, only use pesudo inverse at the beginning
        pseduo_inv_x0 = self.measurement_condition.operator.pseudo_inverse(measurement)
        pseduo_inv_x0 = pseduo_inv_x0.to(image.dtype).to(image.device)
        image = pseduo_inv_x0

        mu = torch.autograd.Variable(image, requires_grad=True) 
        optimizer = torch.optim.Adam([mu], lr=lr, betas=(0.9, 0.99), weight_decay=0.0)

        for t in self.progress_bar(self.scheduler.timesteps):

            noise_x0 = torch.randn_like(mu, device=self.device)
            noise_xt = torch.randn_like(mu, device=self.device)

            x0_pred_original_sample = mu + sigma * noise_x0
            xt_pred = self.scheduler.alphas_cumprod[t].sqrt() * x0_pred_original_sample + (1 - self.scheduler.alphas_cumprod[t]).sqrt() * noise_xt

            # 1. predict noise model_output
            model_output = self.unet(xt_pred, t).sample

            loss_measurement = measurement - self.measurement_condition.operator.forward(x0_pred_original_sample)
            loss_measurement = (loss_measurement**2).mean()/2
            loss_noise = torch.mul((model_output.detach_() - noise_xt).detach(), x0_pred_original_sample).mean()

            snr_inv = (1-self.scheduler.alphas_cumprod[t]).sqrt()/(self.scheduler.alphas_cumprod[t]).sqrt()
            loss = loss_measurement_weight * loss_measurement + snr_inv * loss_noise_weight * loss_noise

            optimizer.zero_grad()  #initialize
            loss.backward()
            optimizer.step()

            # image = image.detach_()
        
        image = x0_pred_original_sample.detach_()
        image = image.cpu().numpy()

        if output_type == "pil":
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)


class DDPMPipelineHybridReg(DDPMPipeline):
    def __init__(
        self,
        unet: UNet2DModel,
        scheduler: DDIMScheduler,
    ):
        super().__init__(unet=unet, scheduler=scheduler)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        # Load components using the base class method
        pipeline = super().from_pretrained(pretrained_model_name_or_path, **kwargs)

        # Replace the scheduler with DDIMScheduler
        ddim_scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
        
        # Return an instance of the subclass
        return cls(unet=pipeline.unet, scheduler=ddim_scheduler)


    # @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 100,
        output_type: Optional[str] = "pil",
        measurement = None,
        return_dict: bool = True,
        sigma =0,
        loss_measurement_weight = 1.0,
        loss_noise_weight = 1.0,
        lr=0.25,
        beta = 0.2,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            batch_size (`int`, *optional*, defaults to 1):
                The number of images to generate.
            generator (`torch.Generator`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            num_inference_steps (`int`, *optional*, defaults to 1000):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.

        Example:

        ```py
        >>> from diffusers import DDPMPipeline

        >>> # load model and scheduler
        >>> pipe = DDPMPipeline.from_pretrained("google/ddpm-cat-256")

        >>> # run pipeline in inference (sample random noise and denoise)
        >>> image = pipe().images[0]

        >>> # save image
        >>> image.save("ddpm_generated_image.png")
        ```

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.ImagePipelineOutput`] is returned, otherwise a `tuple` is
                returned where the first element is a list with the generated images
        """
        if self.measurement_condition is None:
            raise ValueError("Measurement condition is not set.")

        # Sample gaussian noise to begin loop
        if isinstance(self.unet.config.sample_size, int):
            image_shape = (
                batch_size,
                self.unet.config.in_channels,
                self.unet.config.sample_size,
                self.unet.config.sample_size,
            )
        else:
            image_shape = (batch_size, self.unet.config.in_channels, *self.unet.config.sample_size)

        if self.device.type == "mps":
            # randn does not work reproducibly on mps
            image = randn_tensor(image_shape, generator=generator, dtype=self.unet.dtype)
            image = image.to(self.device)
        else:
            image = randn_tensor(image_shape, generator=generator, device=self.device, dtype=self.unet.dtype)

        if image.ndim==4 and measurement.ndim==3:
            measurement = measurement.unsqueeze(0)

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)

        # reddiff specific, only use pesudo inverse at the beginning
        pseduo_inv_x0 = self.measurement_condition.operator.pseudo_inverse(measurement)
        pseduo_inv_x0 = pseduo_inv_x0.to(image.dtype).to(image.device)
        image = pseduo_inv_x0

        mu = torch.autograd.Variable(image, requires_grad=True) 
        optimizer = torch.optim.Adam([mu], lr=lr, betas=(0.9, 0.99), weight_decay=0.0)

        last_noise = 0

        for t in self.progress_bar(self.scheduler.timesteps):

            noise_x0 = torch.randn_like(mu, device=self.device)
            noise_xt = torch.randn_like(mu, device=self.device)

            x0_pred_original_sample = mu + sigma * noise_x0

            if t == self.scheduler.timesteps[0]:
                xt_pred = self.scheduler.alphas_cumprod[t].sqrt() * x0_pred_original_sample + (1 - self.scheduler.alphas_cumprod[t]).sqrt() * noise_xt
            else:
                noise = sqrt(1-beta) * last_noise + sqrt(beta) * noise_xt
                # noise =  sqrt(beta) * last_noise + sqrt(1-beta) * noise_xt
                xt_pred = self.scheduler.alphas_cumprod[t].sqrt() * x0_pred_original_sample + (1 - self.scheduler.alphas_cumprod[t]).sqrt() * noise

            # 1. predict noise model_output
            model_output = self.unet(xt_pred, t).sample

            if t == self.scheduler.timesteps[0]:
                last_noise = model_output.clone()
            else:
                loss_measurement = measurement - self.measurement_condition.operator.forward(x0_pred_original_sample)
                loss_measurement = (loss_measurement**2).mean()/2
                loss_noise = torch.mul((model_output.detach_() - noise_xt).detach(), x0_pred_original_sample).mean()
                last_noise = model_output.clone()

            snr_inv = (1-self.scheduler.alphas_cumprod[t]).sqrt()/(self.scheduler.alphas_cumprod[t]).sqrt()
            if t != self.scheduler.timesteps[0] and t != self.scheduler.timesteps[1]:
                loss = loss_measurement_weight * loss_measurement + snr_inv * loss_noise_weight * loss_noise

                optimizer.zero_grad()  #initialize
                loss.backward()
                optimizer.step()

            # image = image.detach_()
        
        image = x0_pred_original_sample.detach_()
        image = image.cpu().numpy()

        if output_type == "pil":
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)

class LDMPipelineDiffStateGrad(LDMPipelineReSample):
    def __init__(
        self,
        vqvae, 
        unet, 
        scheduler:DDIMSchedulerReSampler
    ):
        super().__init__(vqvae=vqvae,
                        unet=unet,
                        scheduler=scheduler)

    # DiffStateGrad specific methods, ref https://github.com/Anima-Lab/DiffStateGrad/blob/main/ldm/models/diffusion/diffstategrad_ddim.py
    def compute_rank_for_explained_variance(self, singular_values, explained_variance_cutoff):
        """
    Computes average rank needed across channels to explain target variance percentage.
    
    Args:
        singular_values: List of arrays containing singular values per channel
        explained_variance_cutoff: Target explained variance ratio (0-1)
    
    Returns:
        int: Average rank needed across RGB channels
    """
        total_rank = 0
        for channel_singular_values in singular_values:
            squared_singular_values = channel_singular_values ** 2
            cumulative_variance = np.cumsum(squared_singular_values) / np.sum(squared_singular_values)
            rank = np.searchsorted(cumulative_variance, explained_variance_cutoff) + 1
            total_rank += rank
        return int(total_rank / 3)

    def compute_svd_and_adaptive_rank(self, z_t, var_cutoff):
        """
        Compute SVD and adaptive rank for the input tensor.
        
        Args:
            z_t: Input tensor (current image representation at time step t)
            var_cutoff: Variance cutoff for rank adaptation
            
        Returns:
            tuple: (U, s, Vh, adaptive_rank) where U, s, Vh are SVD components
                and adaptive_rank is the computed rank
        """
        # Compute SVD of current image representation
        U, s, Vh = torch.linalg.svd(z_t[0], full_matrices=False)
        
        # Compute adaptive rank
        s_numpy = s.detach().cpu().numpy()

        adaptive_rank = self.compute_rank_for_explained_variance([s_numpy], var_cutoff)
        
        return U, s, Vh, adaptive_rank

    def apply_diffstategrad(self, norm_grad, iteration_count, period, U=None, s=None, Vh=None, adaptive_rank=None):
        """
        Compute projected gradient using DiffStateGrad algorithm.
        
        Args:
            norm_grad: Normalized gradient
            iteration_count: Current iteration count
            period: Period of SVD projection
            U: Left singular vectors from SVD
            s: Singular values from SVD
            Vh: Right singular vectors from SVD
            adaptive_rank: Computed adaptive rank
            
        Returns:
            torch.Tensor: Projected gradient if period condition is met, otherwise original gradient
        """
        if period != 0 and iteration_count % period == 0:
            if any(param is None for param in [U, s, Vh, adaptive_rank]):
                raise ValueError("SVD components and adaptive_rank must be provided when iteration_count % period == 0")
            
            # Project gradient
            A = U[:, :, :adaptive_rank]
            B = Vh[:, :adaptive_rank, :]
            
            low_rank_grad = torch.matmul(A.permute(0, 2, 1), norm_grad[0]) @ B.permute(0, 2, 1)
            projected_grad = torch.matmul(A, low_rank_grad) @ B
            
            # Reshape projected gradient to match original shape
            projected_grad = projected_grad.float().unsqueeze(0)  # Add batch dimension back
            
            return projected_grad
        
        return norm_grad
        

    # @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        eta: float = 0.0,
        num_inference_steps: int = 50,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        measurement = None,
        inter_timesteps = 5,
        pixel_lr=1e-2,
        latent_lr=5e-3,
        var_cutoff=0.99,
        period=1,
        **kwargs,
    ) -> Union[Tuple, ImagePipelineOutput]:
        r"""
        The call function to the pipeline for generation.

        Args:
            batch_size (`int`, *optional*, defaults to 1):
                Number of images to generate.
            generator (`torch.Generator`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.

        Example:

        ```py
        >>> from diffusers import LDMPipeline

        >>> # load model and scheduler
        >>> pipe = LDMPipeline.from_pretrained("CompVis/ldm-celebahq-256")

        >>> # run pipeline in inference (sample random noise and denoise)
        >>> image = pipe().images[0]
        ```

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.ImagePipelineOutput`] is returned, otherwise a `tuple` is
                returned where the first element is a list with the generated images
        """

        if self.measurement_condition is None:
            raise ValueError("Measurement condition is not set.")

        latents = randn_tensor(
            (batch_size, self.unet.config.in_channels, self.unet.config.sample_size, self.unet.config.sample_size),
            generator=generator,
        )
        latents = latents.to(self.device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma

        self.scheduler.set_timesteps(num_inference_steps)

        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())

        extra_kwargs = {}
        if accepts_eta:
            extra_kwargs["eta"] = eta

        for t in self.progress_bar(self.scheduler.timesteps):
            latents = latents.requires_grad_()
            # resample specific
            index = t // 2
            a_t = self.scheduler.alphas_cumprod[index]
            a_prev = self.scheduler.alphas_cumprod[index - 1] if index > 0 else torch.tensor(1.0, device=a_t.device)

            latent_model_input = self.scheduler.scale_model_input(latents, t)
            # predict the noise residual
            noise_prediction = self.unet(latent_model_input, t).sample

            # compute the previous noisy sample x_t -> x_t-1
            out = self.scheduler.step(noise_prediction, t, latents, **extra_kwargs)

            latents_prev = out.prev_sample
            pred_original_latents = out.pred_original_sample
            pseudo_original_latents = out.pseudo_original_sample

            # adjust latents with inverse of vae scale
            pseudo_original_latents = pseudo_original_latents / self.vqvae.config.scaling_factor
            # decode the image latents with the VAE
            pseudo_original_image = self.vqvae.decode(pseudo_original_latents).sample

            # conditioning here, original condition method has no access of the network/pipeline
            Ax = self.measurement_condition.operator.forward(pseudo_original_image)
            difference = measurement-Ax
            # Get smallest non-zero value in the measurement
            min_nonzero = measurement.abs()[measurement.abs() != 0].min()

            # Add epsilon only where measurement == 0
            denominator = measurement.abs().clone()
            denominator[denominator == 0] = min_nonzero

            norm = torch.linalg.norm(difference) / denominator
            norm = norm.mean()
            norm_grad = torch.autograd.grad(outputs=norm, inputs=latents)[0]
            
            # scale is a_t*0.5
            latents = latents_prev - norm_grad*a_t*0.5

            # Instantiating time-travel parameters, resample specfic
            splits = 3 
            index_split = num_inference_steps // splits

            # Performing time-travel if in selected indices
            if index <= (num_inference_steps - index_split) and index > 0:   
                latents_cp = latents.detach().clone()

                # Performing only every 10 steps (or so)
                if index % 10 == 0 :  
                    for k in range(num_inference_steps-index-1, min(num_inference_steps-index-1+inter_timesteps, num_inference_steps-1)):
                        step_ = self.scheduler.timesteps[k+1]
                        index_ = num_inference_steps - k - 1

                        # Obtain x_{t-k} resample specific
                        # predict the noise residual
                        noise_prediction = self.unet(latents, step_).sample

                        # compute the previous noisy sample x_t -> x_t-1
                        out = self.scheduler.step(noise_prediction, step_, latents, **extra_kwargs)

                        latents_prev = out.prev_sample.detach()
                        pred_original_latents = out.pred_original_sample.detach()
                        pseudo_original_latents = out.pseudo_original_sample.detach()
                        latents = latents_prev
                    
                    # Resample sigma scheduling
                    if index >= 0:
                        sigma = 40*(1 - a_prev) / (1 - a_t) * (1 - a_t / a_prev)  
                    else:
                        sigma = 0.5

                    # Pixel-based optimization for second stage
                    if index >= index_split: 
                         # Enforcing consistency via pixel-based optimization
                         pseudo_original_latents = pseudo_original_latents.detach()
                         # adjust latents with inverse of vae scale
                         pseudo_original_latents = pseudo_original_latents / self.vqvae.config.scaling_factor
                         # decode the image latents with the VAE
                         pseudo_original_image = self.vqvae.decode(pseudo_original_latents).sample

                         image = self.vqvae.decode(latents.detach() / self.vqvae.config.scaling_factor).sample

                         opt_image = self.pixel_optimization(measurement=measurement, 
                                                            x_prime=pseudo_original_image, 
                                                            operator_fn=self.measurement_condition.operator.forward,
                                                            eps=1e-3,
                                                            max_iters=2000,
                                                            lr=pixel_lr,
                                                            var_cutoff=var_cutoff,
                                                            x_prev=image,
                                                            period=period)
                         opt_image = opt_image.detach()
                         opt_latents = self.vqvae.encode(opt_image).latents.detach()

                         latents = self.stochastic_resample(pseudo_x0=opt_latents, x_t=latents_cp, a_t=a_prev, sigma=sigma)
                         latents = latents.requires_grad_()
                    # Latent-based optimization for third stage
                    elif index < index_split:
                        pseudo_original_latents, _ = self.latent_optimization(measurement=measurement,
                                                             z_init=pseudo_original_latents.detach(),
                                                             operator_fn=self.measurement_condition.operator.forward,
                                                             var_cutoff=var_cutoff,
                                                             z_prev=latents.detach(),
                                                             period=period)

                        sigma = 40 * (1-a_prev)/(1 - a_t) * (1 - a_t / a_prev)
                        latents = self.stochastic_resample(pseudo_x0=pseudo_original_latents, x_t=latents_cp, a_t=a_prev, sigma=sigma)   

        # adjust latents with inverse of vae scale
        pseudo_original_latents = pseudo_original_latents / self.vqvae.config.scaling_factor
        # decode the image latents with the VAE
        image = self.vqvae.decode(pseudo_original_latents).sample

        image = image.detach_()
        image = image.cpu().numpy()

        if output_type == "pil":
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)

    def pixel_optimization(self, measurement, x_prime, operator_fn, eps=1e-3, max_iters=2000, 
                            lr=None, var_cutoff=None, x_prev=None, period=None):
        """
        Function to compute argmin_x ||y - A(x)||_2^2

        Arguments:
            measurement:           Measurement vector y in y=Ax+n.
            x_prime:               Estimation of \hat{x}_0 using Tweedie's formula
            operator_fn:           Operator to perform forward operation A(.)
            eps:                   Tolerance error
            max_iters:             Maximum number of GD iterations
        """

        loss = torch.nn.MSELoss() # MSE loss

        opt_var = x_prime.detach().clone()
        opt_var = opt_var.requires_grad_()

        if lr is None:
            lr = 1e-2

        optimizer = torch.optim.AdamW([opt_var], lr=lr) # Initializing optimizer
        measurement = measurement.detach() # Need to detach for weird PyTorch reasons

        # Calculate SVD and adaptive rank in prep for DiffStateGrad
        U, s, Vh, adaptive_rank = self.compute_svd_and_adaptive_rank(x_prev, var_cutoff) 

        # Training loop

        for _ in range(max_iters):
            optimizer.zero_grad()
            
            measurement_loss = loss(measurement, operator_fn( opt_var ) ) 
            
            measurement_loss.backward() # Take GD step

            opt_var.grad = self.apply_diffstategrad(opt_var.grad, _, period, U, s, Vh, adaptive_rank) 

            optimizer.step()

            # Convergence criteria
            if measurement_loss < eps**2: # needs tuning according to noise level for early stopping
                break

        return opt_var

    def latent_optimization(self, measurement, z_init, operator_fn, eps=1e-3, max_iters=500, lr=None,
                            var_cutoff=None, z_prev=None, period=None):

        """
        Function to compute argmin_z ||y - A( D(z) )||_2^2

        Arguments:
            measurement:           Measurement vector y in y=Ax+n.
            z_init:                Starting point for optimization
            operator_fn:           Operator to perform forward operation A(.)
            eps:                   Tolerance error
            max_iters:             Maximum number of GD iterations
        
        Optimal parameters seem to be at around 500 steps, 200 steps for inpainting.

        """

        # Base case
        if not z_init.requires_grad:
            z_init = z_init.requires_grad_()

        if lr is None:
            lr = 5e-3

        loss = torch.nn.MSELoss() # MSE loss
        optimizer = torch.optim.AdamW([z_init], lr=lr) # Initializing optimizer ###change the learning rate
        measurement = measurement.detach() # Need to detach for weird PyTorch reasons

        # Calculate SVD and adaptive rank in prep for DiffStateGrad
        U, s, Vh, adaptive_rank = self.compute_svd_and_adaptive_rank(z_prev, var_cutoff)

        # Training loop
        init_loss = 0
        losses = []
        
        for itr in range(max_iters):
            optimizer.zero_grad()
            output = loss(measurement, operator_fn(self.vqvae.decode( z_init ).sample))          

            if itr == 0:
                init_loss = output.detach().clone()
                
            output.backward() # Take GD step

            z_init.grad = self.apply_diffstategrad(z_init.grad, itr, period, U, s, Vh, adaptive_rank)

            optimizer.step()
            cur_loss = output.detach().cpu().numpy() 
            
            # Convergence criteria

            if itr < 200: # may need tuning for early stopping
                losses.append(cur_loss)
            else:
                losses.append(cur_loss)
                if losses[0] < cur_loss:
                    break
                else:
                    losses.pop(0)
                    
            if cur_loss < eps**2:  # needs tuning according to noise level for early stopping
                break


        return z_init, init_loss       

class DDPMPipelineDDS(DDPMPipeline):
    def __init__(
        self,
        unet: UNet2DModel,
        scheduler: DDIMScheduler,
    ):
        super().__init__(unet=unet, scheduler=scheduler)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        # Load components using the base class method
        pipeline = super().from_pretrained(pretrained_model_name_or_path, **kwargs)

        # Replace the scheduler with DDIMScheduler
        ddim_scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
        
        # Return an instance of the subclass
        return cls(unet=pipeline.unet, scheduler=ddim_scheduler)


    # @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 100,
        output_type: Optional[str] = "pil",
        measurement = None,
        return_dict: bool = True,
        eta = 0.85,
        cg_inner=5,
        cg_eps=1e-5,
        gamma=0
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            batch_size (`int`, *optional*, defaults to 1):
                The number of images to generate.
            generator (`torch.Generator`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            num_inference_steps (`int`, *optional*, defaults to 1000):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.

        Example:

        ```py
        >>> from diffusers import DDPMPipeline

        >>> # load model and scheduler
        >>> pipe = DDPMPipeline.from_pretrained("google/ddpm-cat-256")

        >>> # run pipeline in inference (sample random noise and denoise)
        >>> image = pipe().images[0]

        >>> # save image
        >>> image.save("ddpm_generated_image.png")
        ```

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.ImagePipelineOutput`] is returned, otherwise a `tuple` is
                returned where the first element is a list with the generated images
        """
        if self.measurement_condition is None:
            raise ValueError("Measurement condition is not set.")

        # Sample gaussian noise to begin loop
        if isinstance(self.unet.config.sample_size, int):
            image_shape = (
                batch_size,
                self.unet.config.in_channels,
                self.unet.config.sample_size,
                self.unet.config.sample_size,
            )
        else:
            image_shape = (batch_size, self.unet.config.in_channels, *self.unet.config.sample_size)

        if self.device.type == "mps":
            # randn does not work reproducibly on mps
            image = randn_tensor(image_shape, generator=generator, dtype=self.unet.dtype)
            image = image.to(self.device)
        else:
            image = randn_tensor(image_shape, generator=generator, device=self.device, dtype=self.unet.dtype)

        if image.ndim==4 and measurement.ndim==3:
            measurement = measurement.unsqueeze(0)

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)

        bcg = self.measurement_condition.operator.transpose(measurement)

        for t in self.progress_bar(self.scheduler.timesteps):
            with torch.no_grad():
                # 1. predict noise model_output
                model_output = self.unet(image, t).sample

            # 2. compute previous image: x_t -> x_t-1
            out = self.scheduler.step(model_output, t, image, generator=generator)

            # conjugate gradient steps
            x0_pred_original_sample = out.pred_original_sample
            if gamma > 0:
                bcg = x0_pred_original_sample + gamma * self.measurement_condition.operator.transpose(measurement)
            r = bcg - self.measurement_condition.operator.transpose(self.measurement_condition.operator(x0_pred_original_sample)) - gamma * x0_pred_original_sample
            p = r.clone()
            rsold = torch.matmul(r.view(1, -1), r.view(1, -1).T)

            for i in range(cg_inner):
                Ap = self.measurement_condition.operator.transpose(self.measurement_condition.operator(p)) + gamma * p
                a = rsold / torch.matmul(p.view(1, -1), Ap.view(1, -1).T)

                x0_pred_original_sample = x0_pred_original_sample + a * p
                r = r - a * Ap

                rsnew = torch.matmul(r.view(1, -1), r.view(1, -1).T)
                if torch.sqrt(rsnew) < cg_eps:
                    break
                p = r + (rsnew / rsold) * p
                rsold = rsnew
            
            prev_timestep =  t - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps

            alpha_prod_t = self.scheduler.alphas_cumprod[t]
            alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod

            variance = self.scheduler._get_variance(t, prev_timestep)
            std_dev_t = eta * variance ** (0.5)

            # 6. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
            pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (0.5) * model_output

            # 7. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf 
            # conjugate graident updated x0_pred_original_sample used here
            prev_sample = alpha_prod_t_prev ** (0.5) * x0_pred_original_sample + pred_sample_direction

            if eta > 0:
                variance_noise = randn_tensor(
                    model_output.shape, generator=generator, device=model_output.device, dtype=model_output.dtype
                )
                variance = std_dev_t * variance_noise

                prev_sample = prev_sample + variance
            
            image = prev_sample

        image = image.cpu().numpy()

        if output_type == "pil":
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()            
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)