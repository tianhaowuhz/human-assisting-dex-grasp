import math
import copy
import functools
from re import S

import numpy as np
from scipy import integrate
from ipdb import set_trace

import torch

points_per_object = 1024

#----- VE SDE -----
#------------------
def ve_marginal_prob(x, t, sigma_min=0.01, sigma_max=90):
    std = sigma_min * (sigma_max / sigma_min) ** t
    mean = x
    return mean, std

def ve_sde(t, sigma_min=0.01, sigma_max=90):
    sigma = sigma_min * (sigma_max / sigma_min) ** t
    drift_coeff = torch.tensor(0)
    diffusion_coeff = sigma * torch.sqrt(torch.tensor(2 * (np.log(sigma_max) - np.log(sigma_min)), device=t.device))
    return drift_coeff, diffusion_coeff

def ve_prior(shape, sigma_min=0.01, sigma_max=90):
    return torch.randn(*shape) * sigma_max

#----- VP SDE -----
#------------------
def vp_marginal_prob(x, t, beta_0=0.1, beta_1=20):
    log_mean_coeff = -0.25 * t ** 2 * (beta_1 - beta_0) - 0.5 * t * beta_0
    mean = torch.exp(log_mean_coeff) * x
    std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
    return mean, std

def vp_sde(t, beta_0=0.1, beta_1=20):
    beta_t = beta_0 + t * (beta_1 - beta_0)
    drift_coeff = -0.5 * beta_t
    diffusion_coeff = torch.sqrt(beta_t)
    return drift_coeff, diffusion_coeff

def vp_prior(shape, beta_0=0.1, beta_1=20):
    return torch.randn(*shape)

#----- sub-VP SDE -----
#----------------------
def subvp_marginal_prob(x, t, beta_0, beta_1):
    log_mean_coeff = -0.25 * t ** 2 * (beta_1 - beta_0) - 0.5 * t * beta_0
    mean = torch.exp(log_mean_coeff) * x
    std = 1 - torch.exp(2. * log_mean_coeff)
    return mean, std

def subvp_sde(t, beta_0, beta_1):
    beta_t = beta_0 + t * (beta_1 - beta_0)
    drift_coeff = -0.5 * beta_t
    discount = 1. - torch.exp(-2 * beta_0 * t - (beta_1 - beta_0) * t ** 2)
    diffusion_coeff = torch.sqrt(beta_t * discount)
    return drift_coeff, diffusion_coeff

def subvp_prior(shape, beta_0=0.1, beta_1=20):
    return torch.randn(*shape)

def init_sde(sde_mode, min=0.1, max=10.0):
    # the SDE-related hyperparameters are copied from https://github.com/yang-song/score_sde_pytorch
    if sde_mode == 've':
        sigma_min = 0.01
        sigma_max = 90
        prior_fn = functools.partial(ve_prior, sigma_min=sigma_min, sigma_max=sigma_max)
        marginal_prob_fn = functools.partial(ve_marginal_prob, sigma_min=sigma_min, sigma_max=sigma_max)
        sde_fn = functools.partial(ve_sde, sigma_min=sigma_min, sigma_max=sigma_max)
    elif sde_mode == 'vp':
        beta_0 = min
        beta_1 = max
        print(beta_0, beta_1)
        prior_fn = functools.partial(vp_prior, beta_0=beta_0, beta_1=beta_1)
        marginal_prob_fn = functools.partial(vp_marginal_prob, beta_0=beta_0, beta_1=beta_1)
        sde_fn = functools.partial(vp_sde, beta_0=beta_0, beta_1=beta_1)
    elif sde_mode == 'subvp':
        beta_0 = 0.1
        beta_1 = 20
        prior_fn = functools.partial(subvp_prior, beta_0=beta_0, beta_1=beta_1)
        marginal_prob_fn = functools.partial(subvp_marginal_prob, beta_0=beta_0, beta_1=beta_1)
        sde_fn = functools.partial(subvp_sde, beta_0=beta_0, beta_1=beta_1)
    else:
        raise NotImplementedError
    return prior_fn, marginal_prob_fn, sde_fn

# for conditional
# irrelevant to state, size dim
def loss_fn_cond(model, x, marginal_prob_fn, sde_fn, is_likelihood_weighting=False, eps=1e-5, device='cuda:0', hand_pcl=False, full_state=None, envs=None, hand_model=None, space='euler', relative=True):
    """
    is_likelihood_weighting = True, can potentially improve likelihood-estimation (e.g., for reward learning)
    """
    hand_dof_batch, obj_pcl_batch = x
    if space == 'riemann':
        hand_dof_batch = action2grad(hand_dof_batch, relative=relative)
    batchsize = hand_dof_batch.shape[0]
    random_t = torch.rand(batchsize, device=device) * (1. - eps) + eps
    # random_t = torch.pow(10,-5*random_t)  
    random_t = random_t.unsqueeze(-1)
    z = torch.randn_like(hand_dof_batch)
    mu, std = marginal_prob_fn(hand_dof_batch, random_t)
    perturbed_hand_dof_batch = mu + z * std

    if hand_pcl:
        if space == 'riemann':
            hand_dof = action2grad(perturbed_hand_dof_batch.clone(), relative=relative, inv=True)
        else:
            hand_dof = perturbed_hand_dof_batch.clone()  
        hand_pos_2_w = full_state[:,18:21].clone().to(device).float()
        hand_quat_2_w = full_state[:,21:25].clone().to(device).float()
        hand_pos_2_h, hand_quat_2_h = envs.transform_target2source(hand_quat_2_w, hand_pos_2_w, hand_quat_2_w, hand_pos_2_w)

        ori_hand_dof = envs.dof_norm(hand_dof.clone(),inv=True)
        hand_pcl_2h = hand_model.get_hand_pcl(hand_pos=hand_pos_2_h, hand_quat=hand_quat_2_h, hand_dof=ori_hand_dof)
        obj_pcl_batch = torch.cat([obj_pcl_batch, hand_pcl_2h.reshape(hand_pcl_2h.size(0),hand_pcl_2h.size(2),hand_pcl_2h.size(1))],2)

    output = model((perturbed_hand_dof_batch.reshape(batchsize, -1, 1), obj_pcl_batch), random_t)

    total_loss = (output + z / std) ** 2
    if is_likelihood_weighting:
        _, diffusion_coeff = sde_fn(random_t)
        loss_weighting = diffusion_coeff ** 2
        node_l2 = torch.sum(total_loss, dim=-1) * loss_weighting
    else:
        loss_weighting = std ** 2
        node_l2 = torch.sum(total_loss * loss_weighting, dim=-1)
    loss_ = torch.mean(node_l2)
    return loss_

def action2grad(x,inv=False, relative=True):
    if not inv:
      if relative:
        batch_size = x.size(0)
        state_dim = x.size(1)
        x = torch.cat([torch.sin(x).reshape(batch_size,state_dim,1), torch.cos(x).reshape(batch_size,state_dim,1)],2).reshape(batch_size,-1)
      else:
        batch_size = x.size(0)
        hand_pose_dim = 7
        hand_dof_dim = 18
        hand_pose = x[:,18:25]
        hand_dof = x[:,:18]
        hand_dof = torch.cat([torch.sin(hand_dof).reshape(batch_size,hand_dof_dim,1), torch.cos(hand_dof).reshape(batch_size,hand_dof_dim,1)],2).reshape(batch_size,-1)
        x = torch.cat([hand_dof, hand_pose],-1)
      return x
    else:
      if len(x.size())==3:
        step = x.size(0)
        batch_size = x.size(1)
        
        if relative:
          state_dim = x.size(2)
          x = x.reshape(step,batch_size,int(state_dim/2),2)
          x = torch.atan2(x[:,:,:,0:1],x[:,:,:,1:2]).reshape(step,batch_size,int(state_dim/2))
        else:
          state_dim = x.size(2) - 7
          # 18 * 2 sin cos + 7 hand pose
          hand_pose = x[:,:,36:43]
          hand_dof = x[:,:,:36]
          hand_dof = hand_dof.reshape(step,batch_size,int(state_dim/2),2)
          hand_dof = torch.atan2(hand_dof[:,:,:,0:1],hand_dof[:,:,:,1:2]).reshape(step,batch_size,int(state_dim/2))
          x = torch.cat([hand_dof, hand_pose],-1)
        return x
      elif len(x.size())==2:
        batch_size = x.size(0)
        
        if relative:
          state_dim = x.size(1)
          x = x.reshape(batch_size,int(state_dim/2),2)
          x = torch.atan2(x[:,:,0:1],x[:,:,1:2]).reshape(batch_size,int(state_dim/2))
        else:
          state_dim = x.size(1) - 7
          hand_pose = x[:, 36:43]
          hand_dof = x[:, :36]
          hand_dof = hand_dof.reshape(batch_size,int(state_dim/2),2)
          hand_dof = torch.atan2(hand_dof[:,:,0:1],hand_dof[:,:,1:2]).reshape(batch_size,int(state_dim/2))
          x = torch.cat([hand_dof, hand_pose],-1)
        return x

def cond_ode_sampler(
    score_model,
    prior_fn,
    sde_fn,
    state,
    batch_size=64,
    atol=1e-5,
    rtol=1e-5,
    device='cuda',
    eps=1e-5,
    t0=1,
    num_steps=None,
    is_random=True,
    denoise=True, 
    hand_pcl=False, 
    full_state=None, 
    envs=None, 
    hand_model=None,
    space='euler',
    relative=True,
):
    hand_dof_batch, obj_pcl_batch = state
    if space == 'riemann':
      hand_dof_batch = action2grad(hand_dof_batch, relative=relative)
    t0_ = torch.ones(batch_size, device=device)*t0

    if is_random:
        init_x = prior_fn(hand_dof_batch.shape).to(device) # normal distribution
        # init_x = torch.randn_like(hand_dof_batch, device=device) * marginal_prob_std(t0_)
        # init_x = -torch.ones_like(hand_dof_batch, device=device)
        # init_x = torch.tensor([ 0.0000, -0.7143, -1.0000,  0.0000, -0.7143, -1.0000,  0.0000, -0.7143,
        #  -1.0000, -1.0000,  0.0000, -0.7143, -1.0000,  0.0000, -1.0000,  0.0000,
        #   0.0000, -1.0000,1,1,1,1,1,1,1], device=device).reshape(1,-1)[:,:hand_dof_batch.size(1)].expand_as(hand_dof_batch)
    else:
        batch_size = hand_dof_batch.size(0)
        init_x = hand_dof_batch
        
    # Create the latent code
    # init_x = torch.randn_like(hand_dof_batch, device=device) * marginal_prob_std(t0_)
    # !!! for dex hand only, set to same init state
    # init_x = hand_dof_batch
    shape = init_x.shape
    state_dim = shape[-1]

    def score_eval_wrapper(sample, time_steps):
        """A wrapper of the score-based model for use by the ODE solver."""
        with torch.no_grad():
            score = score_model(sample, time_steps)
        # return score.cpu().numpy().reshape((-1,))
        return score.cpu().numpy().reshape(-1)

    def ode_func(t, x):
        """The ODE function for use by the ODE solver."""
        x = torch.tensor(x.reshape(-1, state_dim)).to(device).float()
        time_steps = torch.ones(batch_size, device=device).unsqueeze(1) * t
        # if batch_size == 1:
        #     time_steps = torch.ones(batch_size, device=device).unsqueeze(1) * t
        # else:
        #     time_steps = torch.ones(batch_size, device=device) * t
        drift, diffusion = sde_fn(torch.tensor(t))
        drift = drift.cpu().numpy()
        diffusion = diffusion.cpu().numpy()
        if hand_pcl:
          hand_dof = x.clone()  
          hand_pos_2_w = full_state[:,18:21].clone().to(device).float()
          hand_quat_2_w = full_state[:,21:25].clone().to(device).float()
          hand_pos_2_h, hand_quat_2_h = envs.transform_target2source(hand_quat_2_w, hand_pos_2_w, hand_quat_2_w, hand_pos_2_w)

          if space == 'riemann':
              hand_dof = action2grad(hand_dof.clone(), relative=relative, inv=True)
          else:
              hand_dof = perturbed_hand_dof_batch.clone()  

          ori_hand_dof = envs.dof_norm(hand_dof.clone(),inv=True)
          hand_pcl_2h = hand_model.get_hand_pcl(hand_pos=hand_pos_2_h, hand_quat=hand_quat_2_h, hand_dof=ori_hand_dof)
          objhand_pcl_batch = torch.cat([obj_pcl_batch, hand_pcl_2h.reshape(hand_pcl_2h.size(0),hand_pcl_2h.size(2),hand_pcl_2h.size(1))],2)
          gradient = score_eval_wrapper((x, objhand_pcl_batch), time_steps)
        else:
          gradient = score_eval_wrapper((x, obj_pcl_batch), time_steps)
        # gradient[:6]*=100
        # gradient[6:30]*=10
        return drift - 0.5 * (diffusion**2) * gradient
    
    # Run the black-box ODE solver.
    t_eval = None
    if num_steps is not None:
        # num_steps, from t0 -> eps
        t_eval = np.linspace(t0, eps, num_steps)

    res = integrate.solve_ivp(ode_func, (t0, eps), init_x.reshape(-1).cpu().numpy(), rtol=rtol, atol=atol,
                              method='RK45', t_eval=t_eval)
    # process, xs: [total_nodes*3, samples_num]
    # clamp for now TODO
    # xs = torch.clamp(torch.tensor(res.y, device=device).T, min=-1.0, max=1.0)    
    xs = torch.tensor(res.y, device=device).T
    xs = xs.view(num_steps, hand_dof_batch.shape[0], -1)

    # result x: [total_nodes, 3]
    x = torch.clamp(torch.tensor(res.y[:, -1], device=device).reshape(shape), min=-1.0, max=1.0)
    # x = torch.tensor(res.y[:, -1], device=device).reshape(shape)

    # denoise, using the predictor step in P-C sampler
    if denoise:
        # Reverse diffusion predictor for denoising
        vec_eps = torch.ones((x.shape[0], 1), device=x.device) * eps
        drift, diffusion = sde_fn(vec_eps)
        grad = score_model((x.float(), obj_pcl_batch), vec_eps)
        drift = drift - diffusion ** 2 * grad # R-SDE
        mean_x = x + drift * ((1 - eps) / (1000 if num_steps is None else num_steps))
        x = mean_x
    
    if space=='riemann':
      xs = action2grad(xs, inv=True, relative=relative)
      x = action2grad(x, inv=True, relative=relative)
      
    return xs, x

class ExponentialMovingAverage:
  """
  Maintains (exponential) moving average of a set of parameters.
  """

  def __init__(self, parameters, decay, use_num_updates=True):
    """
    Args:
      parameters: Iterable of `torch.nn.Parameter`; usually the result of
        `model.parameters()`.
      decay: The exponential decay.
      use_num_updates: Whether to use number of updates when computing
        averages.
    """
    if decay < 0.0 or decay > 1.0:
      raise ValueError('Decay must be between 0 and 1')
    self.decay = decay
    self.num_updates = 0 if use_num_updates else None
    self.shadow_params = [p.clone().detach()
                          for p in parameters if p.requires_grad]
    self.collected_params = []

  def update(self, parameters):
    """
    Update currently maintained parameters.

    Call this every time the parameters are updated, such as the result of
    the `optimizer.step()` call.

    Args:
      parameters: Iterable of `torch.nn.Parameter`; usually the same set of
        parameters used to initialize this object.
    """
    decay = self.decay
    if self.num_updates is not None:
      self.num_updates += 1
      decay = min(decay, (1 + self.num_updates) / (10 + self.num_updates))
    one_minus_decay = 1.0 - decay
    with torch.no_grad():
      parameters = [p for p in parameters if p.requires_grad]
      for s_param, param in zip(self.shadow_params, parameters):
        s_param.sub_(one_minus_decay * (s_param - param)) # only update the ema-params

  def copy_to(self, parameters):
    """
    Copy current parameters into given collection of parameters.

    Args:
      parameters: Iterable of `torch.nn.Parameter`; the parameters to be
        updated with the stored moving averages.
    """
    parameters = [p for p in parameters if p.requires_grad]
    for s_param, param in zip(self.shadow_params, parameters):
      if param.requires_grad:
        param.data.copy_(s_param.data)

  def store(self, parameters):
    """
    Save the current parameters for restoring later.

    Args:
      parameters: Iterable of `torch.nn.Parameter`; the parameters to be
        temporarily stored.
    """
    self.collected_params = [param.clone() for param in parameters]

  def restore(self, parameters):
    """
    Restore the parameters stored with the `store` method.
    Useful to validate the model with EMA parameters without affecting the
    original optimization process. Store the parameters before the
    `copy_to` method. After validation (or model saving), use this to
    restore the former parameters.

    Args:
      parameters: Iterable of `torch.nn.Parameter`; the parameters to be
        updated with the stored parameters.
    """
    for c_param, param in zip(self.collected_params, parameters):
      param.data.copy_(c_param.data)

  def state_dict(self):
    return dict(decay=self.decay, num_updates=self.num_updates,
                shadow_params=self.shadow_params)

  def load_state_dict(self, state_dict):
    self.decay = state_dict['decay']
    self.num_updates = state_dict['num_updates']
    self.shadow_params = state_dict['shadow_params']
