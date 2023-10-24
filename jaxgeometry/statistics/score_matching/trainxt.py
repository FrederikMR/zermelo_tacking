#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 12:20:21 2023

@author: fmry
"""

#%% Sources


#%% Modules

from jaxgeometry.setup import *
from jaxgeometry.statistics.score_matching.model_loader import save_model

#%% TrainingState

class TrainingState(NamedTuple):
    params: hk.Params
    state_val:  dict
    opt_state: optax.OptState
    rng_key: Array

#%% Score Training
#https://scoste.fr/posts/diffusion/#denoising_score_matching
def train_s1(M:object,
             model:object,
             data_generator:Callable[[], jnp.ndarray],
             update_coords:Callable[[jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray]],
             proj_grad:Callable[[hk.Params, dict, Array, jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray],
             N_dim:int,
             batch_size:int,
             state:TrainingState = None,
             lr_rate:float = 0.001,
             gamma:float=1.0,
             epochs:int=100,
             save_step:int=100,
             optimizer:object=None,
             save_path:str = "",
             loss_type:str='vsm',
             seed:int=2712
             )->None:
    
    def loss_vsm(params:hk.Params, state_val:dict, rng_key:Array, data:jnp.ndarray)->float:
        """ compute loss."""
        
        x0 = data[:,:N_dim]
        xt = data[:,N_dim:2*N_dim]
        t = data[:,2*N_dim]
        #noise = data[:,(2*N_dim+1):-1]
        #dt = data[:,-1]
        
        s = apply_fn(params, data[:,:(2*N_dim+1)], rng_key, state_val)
        norm2s = jnp.sum(s*s, axis=1)
        
        (xts,chartts) = vmap(update_coords)(xt)
        
        s1 = lambda x,y,t: apply_fn(params, jnp.hstack((x,y,t)), rng_key, state_val)
        divs = vmap(lambda x0, xt, chart, t: M.div((xt, chart), lambda x: proj_grad(s1, x0, x, t)))(x0,xts,chartts,t)
        
        return jnp.mean(norm2s+2.0*divs)

    def loss_dsm(params:hk.Params, state_val:dict, rng_key:Array, data:jnp.ndarray)->float:
        
        def f(x0,xt,t,noise,dt):
            
            s1 = lambda x,y,t: apply_fn(params, jnp.hstack((x,y,t)), rng_key, state_val)
            s1 = proj_grad(s1, x0, xt, t)
            #s1 = s1(x0, M.F((xt,chart)), t)
            
            #JFx = M.JF((xt,chart))
            #noise = jnp.dot(JFx,noise)

            loss = noise/dt+s1
            
            return jnp.sum(loss*loss)
        
            #s1_model = lambda x,y,t: apply_fn(params, jnp.hstack((x,y,t)), rng_key, state_val)
            #s1 = s1_model(x0,M.F((xt,chart)), t)
            #s1_proj = proj_grad(s1_model,x0,(xt,chart), t)
            #loss = jnp.sum(noise*noise/(dt**2))+jnp.sum(s1*s1)+2*jnp.sum(noise*s1_proj/dt)
            #return loss
        
        x0 = data[:,:N_dim]
        xt = data[:,N_dim:(2*N_dim)]
        t = data[:,2*N_dim]
        noise = data[:,(2*N_dim+1):-1]
        dt = data[:,-1]

        #loss = jnp.mean(vmap(
        #            vmap(
        #                f,
        #                (0,0,0,0)),
        #            (1,1,None,1))(x0,xt,t,noise))
        
        
        loss = jnp.mean(vmap(
                        f,
                        (0,0,0,0,0))(x0,xt,t,noise,dt))
    
        return loss
    #@jit
    def update(state:TrainingState, data:jnp.ndarray):
        
        rng_key, next_rng_key = random.split(state.rng_key)
        loss, gradients = value_and_grad(loss_fun)(state.params, state.state_val, rng_key, data)
        if jnp.isnan(loss):
            return state, loss
        else:
            updates, new_opt_state = optimizer.update(gradients, state.opt_state)
            new_params = optax.apply_updates(state.params, updates)
            return TrainingState(new_params, state.state_val, new_opt_state, rng_key), loss
    
    if loss_type == "vsm":
        loss_fun = jit(loss_vsm)
    elif loss_type == "dsm":
        loss_fun = jit(loss_dsm)
    else:
        print("Invalid loss function: Using Denoising Score Matching as default")
        loss_fun = jit(loss_dsm)
        
    if optimizer is None:
        optimizer = optax.adam(learning_rate = lr_rate,
                               b1 = 0.9,
                               b2 = 0.999,
                               eps = 1e-08,
                               eps_root = 0.0,
                               mu_dtype=None)
        
    train_dataset = tf.data.Dataset.from_generator(data_generator,output_types=tf.float32,
                                                   output_shapes=([batch_size,2*N_dim+M.dim+2]))
    train_dataset = iter(tfds.as_numpy(train_dataset))
        
    initial_rng_key = random.PRNGKey(seed)
    if type(model) == hk.Transformed:
        if state is None:
            initial_params = model.init(random.PRNGKey(seed), next(train_dataset)[:,:(2*N_dim+1)])
            initial_opt_state = optimizer.init(initial_params)
            state = TrainingState(initial_params, None, initial_opt_state, initial_rng_key)
        apply_fn = lambda params, data, rng_key, state_val: model.apply(params, rng_key, data)
    elif type(model) == hk.TransformedWithState:
        if state is None:
            initial_params, init_state = model.init(random.PRNGKey(seed), next(train_dataset)[:,:(2*N_dim+1)])
            initial_opt_state = optimizer.init(initial_params)
            state = TrainingState(initial_params, init_state, initial_opt_state, initial_rng_key)
        apply_fn = lambda params, data, rng_key, state_val: model.apply(params, state_val, rng_key, data)[0]
    
    loss = []
    for step in range(epochs):
        state, loss_val = update(state, next(train_dataset))
        if (step+1) % save_step == 0:
            loss_val = device_get(loss_val).item()
            loss.append(loss_val)
            
            np.save(os.path.join(save_path, "loss_arrays.npy"), jnp.stack(loss))
            
            save_model(save_path, state)
            print("Epoch: {} \t loss = {:.4f}".format(step+1, loss_val))

    loss.append(loss_val)
    
    np.save(os.path.join(save_path, "loss_arrays.npy"), jnp.stack(loss))
    
    save_model(save_path, state)
    print("Epoch: {} \t loss = {:.4f}".format(step+1, loss_val))
    
    return

#%% Score Training for SN
#https://scoste.fr/posts/diffusion/#denoising_score_matching
def train_s2(M:object,
             s1_model:Callable[[ndarray, ndarray, ndarray], ndarray],
             s2_model:object,
             data_generator:Callable[[], jnp.ndarray],
             update_coords:Callable[[jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray]],
             proj_grad:Callable[[Callable, jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray],
             proj_hess:Callable[[Callable, jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray],
             N_dim:int,
             batch_size:int,
             gamma:float=1.0,
             lr_rate:float=0.0002,
             epochs:int=100,
             save_step:int=100,
             optimizer:object=None,
             save_path:str = "",
             seed:int=2712
             )->None:
    
    @jit
    def loss_fun(params:hk.Params, 
                 state_val:dict, 
                 rng_key:Array, 
                 data:jnp.ndarray
                 )->float:
        
        def f(x0,xt,chart,t,noise,dt):
            
            xp = xt+noise
            xm = xt-noise
            
            s2_model = lambda x,y,t: apply_fn(params, jnp.hstack((x,y,t)), rng_key, state_val)
            
            s1 = proj_grad(s1_model, x0, (xt,chart), t)
            s2 = proj_hess(s1_model, s2_model, x0, (xt,chart), t)

            s1p = proj_grad(s1_model, x0, (xp,chart), t)
            s2p = proj_hess(s1_model, s2_model, x0, (xp,chart), t)
            
            s1m = proj_grad(s1_model, x0, (xm,chart), t)
            s2m = proj_hess(s1_model, s2_model, x0, (xm,chart), t)
            
            psi = s2+jnp.einsum('i,j->ij', s1, s1)
            psip = s2p+jnp.einsum('i,j->ij', s1p, s1p)
            psim = s2m+jnp.einsum('i,j->ij', s1m, s1m)
            
            loss_s2 = psim**2+psim**2\
                +2*(jnp.eye(N_dim)-jnp.eye(N_dim)-jnp.einsum('i,j->ij', noise, noise)/dt)*\
                    (psip+psim-2*psi)
                                
            return loss_s2
        
            #s2_model = lambda x,y,t: apply_fn(params, jnp.hstack((x,y,t)), rng_key, state_val)
            
            #s1 = proj_grad(s1_model, x0, (xt,chart), t)
            #s2 = proj_hess(s1_model, s2_model, x0, (xt,chart), t)
            
            #loss_s2 = s2+jnp.einsum('i,j->ij', s1, s1)+\
            #    (jnp.eye(N_dim)-jnp.einsum('i,j->ij', noise, noise)/dt)/dt
                            
            #return jnp.sum(loss_s2*loss_s2)
        
        x0 = data[:,:N_dim]
        xt = data[:,N_dim:(2*N_dim)]
        (xt,chart) = vmap(update_coords)(xt)
        t = data[:,2*N_dim]
        noise = data[:,(2*N_dim+1):-1]
        dt = data[:,-1]

        #loss = jnp.mean(vmap(
        #            vmap(
        #                f,
        #                (0,0)),
        #            (None, 1))(t,data[:,2*N_dim]))
        
        loss = jnp.mean(vmap(
                        f,
                        (0,0,0,0,0,0))(x0,xt,chart,t,noise,dt))
    
        return loss
    
    @jit
    def update(state:TrainingState, data:jnp.ndarray):
        
        rng_key, next_rng_key = random.split(state.rng_key)
        loss, gradients = value_and_grad(loss_fun)(state.params, state.state_val, rng_key, data)
        updates, new_opt_state = optimizer.update(gradients, state.opt_state)
        new_params = optax.apply_updates(state.params, updates)
        
        return TrainingState(new_params, state.state_val, new_opt_state, rng_key), loss
        
    if optimizer is None:
        optimizer = optax.adam(learning_rate = lr_rate,
                               b1 = 0.9,
                               b2 = 0.999,
                               eps = 1e-08,
                               eps_root = 0.0,
                               mu_dtype=None)
        
    train_dataset = tf.data.Dataset.from_generator(data_generator,output_types=tf.float32,
                                                   output_shapes=([batch_size,2*N_dim+M.dim+2]))
    train_dataset = iter(tfds.as_numpy(train_dataset))
    
    initial_rng_key = random.PRNGKey(seed)
    if type(s2_model) == hk.Transformed:
        initial_params = s2_model.init(random.PRNGKey(seed), next(train_dataset)[:,:(2*N_dim+1)])
        initial_opt_state = optimizer.init(initial_params)
        state = TrainingState(initial_params, None, initial_opt_state, initial_rng_key)
        apply_fn = lambda params, data, rng_key, state_val: s2_model.apply(params, rng_key, data)
    elif type(s2_model) == hk.TransformedWithState:
        initial_params, init_state = s2_model.init(random.PRNGKey(seed), next(train_dataset)[:,:(2*N_dim+1)])
        initial_opt_state = optimizer.init(initial_params)
        state = TrainingState(initial_params, init_state, initial_opt_state, initial_rng_key)
        apply_fn = lambda params, data, rng_key, state_val: s2_model.apply(params, state_val, rng_key, data)[0]
    
    loss = []
    for step in range(epochs):
        state, loss_val = update(state, next(train_dataset))
        if (step+1) % save_step == 0:
            loss_val = device_get(loss_val).item()
            loss.append(loss_val)
            
            np.save(os.path.join(save_path, "loss_arrays.npy"), jnp.stack(loss))
            
            save_model(save_path, state)
            print("Epoch: {} \t loss = {:.4f}".format(step+1, loss_val))

    loss.append(loss_val)
    
    np.save(os.path.join(save_path, "loss_arrays.npy"), jnp.stack(loss))
    
    save_model(save_path, state)
    print("Epoch: {} \t loss = {:.4f}".format(step+1, loss_val))
    
    return