import time
import argparse

from huxel.optimization import _optimization as _opt
from huxel.optimization_ml import _optimization as _optml
from huxel.prediction import _pred

def main():
    parser = argparse.ArgumentParser(description='opt overlap NN')
    parser.add_argument('--N', type=int, default=5, help='traning data')
    parser.add_argument('--l', type=int, default=10, help='label')
    parser.add_argument('--lr', type=float, default=2E-3, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=16, help='batches')
    parser.add_argument('--job', type=str, default='optml', help='job type')
    parser.add_argument('--beta', type=str, default='exp', help='beta function type')
    parser.add_argument('--randW', type=bool, default=False, help='random initial params')
    parser.add_argument('-Wdecay', '--item', action='store', dest='alist',
                    type=str, nargs='*', default=['alpha', 'beta', 'h_x', 'h_xy',  'y_xy'],
                    help="Examples: -i h_x h_xy r_xy y_xy'")
     # removed 'r_xy',

    # bathch_size = #1024#768#512#256#128#64#32
    args = parser.parse_args()
    l = args.l
    n_tr = args.N
    lr = args.lr
    batch_size = args.batch_size
    job_ = args.job
    beta_ = args.beta
    bool_randW = args.randW
    list_Wdecay = args.alist

    if job_ == 'opt':
        _opt(n_tr,batch_size,lr,l,beta_,list_Wdecay,bool_randW)
    elif job_ == 'optml':
        _optml(n_tr,batch_size,lr,l,beta_,list_Wdecay,bool_randW)
    elif job_ == 'pred':
        _pred(n_tr,l,beta_,bool_randW)




if __name__ == "__main__":
    main()
