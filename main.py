import argparse

from huxel import opt, pred, pred_def


def main():
    parser = argparse.ArgumentParser(description="Huxel = JAX + Huckel model")
    parser.add_argument("--N", type=int, default=101, help="training data")
    parser.add_argument("--l", type=int, default=0, help="label")
    parser.add_argument("--lr", type=float, default=2e-2, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=16, help="batches")
    parser.add_argument("--job", type=str, default="pred_def",
                        help="job type", choices=['opt', 'pred', 'pred_def'])
    parser.add_argument('--obs', type=str, default='homo_lumo',
                        help="molecular observable [homo_lumo,pol]")
    parser.add_argument("--beta", type=str, default="c",
                        help="beta function type")
    parser.add_argument("--pred_data", type=str, default="val",
                        help="predict val or training data: True -> validation")
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
        default=[],  # ["hl_params","pol_params", "hl_b", "h_x", "h_xy", "r_xy", "y_xy", "all"],
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
    pred_data = args.pred_data

    if job_ == "opt":
        opt(obs_, n_tr, batch_size, lr, l, beta_, list_Wdecay, bool_randW)
    elif job_ == "pred":
        pred(obs_, n_tr, l, beta_, bool_randW)
    elif job_ == "pred_def" or job_ == "pred0":
        pred_def(obs_, beta_, pred_data)


if __name__ == "__main__":
    main()
