import time
import argparse

from huxel.optimization import _optimization as _opt
from huxel.prediction import _pred, _pred_def


def main():
    parser = argparse.ArgumentParser(description="Huxel = JAX + Huckel model")
    parser.add_argument("--N", type=int, default=0, help="traning data")
    parser.add_argument("--l", type=int, default=0, help="label")
    parser.add_argument("--lr", type=float, default=2e-3, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=16, help="batches")
    parser.add_argument("--job", type=str, default="pred_def", help="job type")
    parser.add_argument('--obs', type=str, default='homo_lumo', help="molecular observable")
    parser.add_argument("--beta", type=str, default="c", help="beta function type")
    parser.add_argument(
        "--randW", type=bool, default=False, help="random initial params"
    )
    parser.add_argument(
        "-Wdecay",
        "--item",
        action="store",
        dest="alist",
        type=str,
        nargs="*",
        default=["alpha", "beta", "h_x", "h_xy", "r_xy", "y_xy"],
        help="Examples: -i h_x h_xy r_xy y_xy'",
    )

    # bathch_size = #1024#768#512#256#128#64#32
    args = parser.parse_args()
    l = args.l
    n_tr = args.N
    lr = args.lr
    batch_size = args.batch_size
    job_ = args.job
    obs_ = args.obs
    beta_ = args.beta
    bool_randW = args.randW
    list_Wdecay = args.alist

    # assert 0

    if job_ == "opt":
        _opt(obs_,n_tr, batch_size, lr, l, beta_, list_Wdecay, bool_randW)
    elif job_ == "pred":
        _pred(obs_,n_tr, l, beta_, bool_randW)
    elif job_ == "pred_def":
        _pred_def(obs_,beta_)

if __name__ == "__main__":
    main()
