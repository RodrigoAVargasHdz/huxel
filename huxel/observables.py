from typing import Any, Dict, Tuple
import jax
import jax.numpy as jnp
from typing import Any, Callable

from huxel.huckel import homo_lumo_pred, polarizability_pred
from huxel.utils import normalize_params_polarizability, normalize_params_wrt_C, get_external_field
from huxel.beta_functions import _f_beta
from huxel.data_utils import data_normalization


def _preprocessing_params(observable: str = 'homo_lumo') -> Callable:
    """Wrapper function to normalize the parameters with respect to C atom

    Args:
        observable (str): target observable. Default: homo_lumo

    Returns:
        Callable: pre-processing function
    """
    if observable.lower() == 'homo_lumo' or observable.lower() == 'hl':
        def wrapper(*args):
            return normalize_params_wrt_C(*args)
        return wrapper
    elif observable.lower() == 'polarizability' or observable.lower() == 'pol':
        def wrapper(*args):
            return normalize_params_polarizability(*args)
        return wrapper


def _f_observable(observable: str, beta: str, external_field: Any = None) -> Callable:
    """Wrapper function for the observable computation.

    Args:
        observable (str): target observable.
        beta (str): atom-atom interaction
        external_field (Any, optional): external field value. Defaults to None.

    Returns:
        Callable: observable function
    """
    f_beta = _f_beta(beta)
    ext_field = get_external_field(observable, external_field)

    f_phys_params = _preprocessing_params(observable)

    if observable.lower() == 'homo_lumo' or observable.lower() == 'hl':
        def wrapper(*args):
            params = f_phys_params(args[0])  # Carbon normalization
            return homo_lumo_pred(params, *args[1:], f_beta)
        return wrapper

    elif observable.lower() == 'polarizability' or observable.lower() == 'pol':
        def wrapper(*args):
            params = f_phys_params(args[0])  # Carbon normalization
            return polarizability_pred(params, args[1], f_beta, ext_field)
        return wrapper


def loss_rmse(params_tot: Dict, batch: Any, f_obs: Callable) -> Tuple:
    """Root-mean square error 

    Args:
        params_tot (Dict): parameters
        batch (Any): batch (molecules)
        f_obs (Callable): function to compute observable

    Returns:
        Tuple: rmse, (predicted values, true values)
    """

    # params_tot = update_params_all(params_tot) # Carbon normalization
    y_pred, z_pred, y_true = f_obs(params_tot, batch)
    diff_y = (y_pred - y_true) ** 2

    return jnp.mean(diff_y), (y_pred, z_pred, y_true)


def _loss_function(observable: str, beta: str, external_field: Any = None):
    if observable.lower() == "hl_pol" or observable.lower() == "homo_lumo_polarizability" or observable.lower() == "all":
        f_hl = _f_observable("homo_lumo", beta)
        f_pol = _f_observable("polarizability", beta, external_field)

        def wrapper(*args):
            labmdas_ = args[-1]
            error_hl, _ = loss_rmse(*args[:-1], f_hl)
            error_pol, _ = loss_rmse(*args[:-1], f_pol)
            errors = jnp.stack([error_hl, error_pol])
            return jnp.vdot(labmdas_, errors),  errors
        return wrapper
    else:
        f_obs = _f_observable(observable, beta, external_field)

        def wrapper(*args):
            return loss_rmse(*args, f_obs)
        return wrapper
