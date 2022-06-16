import jax
import jax.numpy as jnp
from jax import lax


def _beta_r_exp(beta_ref:float,r_ref:float,y_ref:float,r:float):
    z = (1./y_ref)*(r-r_ref)
    return beta_ref * jnp.exp(-z)

def _beta_abs_r_exp(beta_ref:float,r_ref:float,y_ref:float,r:float):
    z = (1./y_ref)*(jnp.abs(r-r_ref))
    return beta_ref * jnp.exp(-z)

def _beta_r_linear(beta_ref:float,r_ref:float,y_ref:float,r:float):
    z = 1. - (1./y_ref)*(r-r_ref)
    return beta_ref * z

def _beta_abs_r_linear(beta_ref:float,r_ref:float,y_ref:float,r:float):
    z = 1. - (1./y_ref)*(jnp.abs(r-r_ref))
    return beta_ref * z

def _beta0(beta_ref:float):
    return beta_ref

def _f_beta(method:str):
    if method == 'exp':
        def wrapper(*args):
            return _beta_r_exp(*args)
        return wrapper  
    elif method == 'exp_abs':
        def wrapper(*args):
            return _beta_abs_r_exp(*args)
        return wrapper
    elif method == 'linear':
        def wrapper(*args):
            return _beta_r_linear(*args)
        return wrapper  
    elif method == 'linear_abs':
        def wrapper(*args):
            return _beta_abs_r_linear(*args)
        return wrapper  
    elif method == 'exp_freezeR':
        def wrapper(*args):
            return _beta_r_exp(args[0],lax.stop_gradient(args[1]),args[2],args[3])
        return wrapper  
    elif method == 'exp_abs_freezeR':
        def wrapper(*args):
            return _beta_abs_r_exp(args[0],lax.stop_gradient(args[1]),args[2],args[3])
        return wrapper  
    elif method == 'linear_freezeR':
        def wrapper(*args):
            return _beta_r_linear(args[0],lax.stop_gradient(args[1]),args[2],args[3])
        return wrapper  
    elif method == 'linear_abs_freezeR':
        def wrapper(*args):
            return _beta_abs_r_linear(args[0],lax.stop_gradient(args[1]),args[2],args[3])
        return wrapper   
    elif method == 'constant'or method == 'c' or method == 'randW':
        def wrapper(*args):
            return _beta0(*args[:1])
        return wrapper  
