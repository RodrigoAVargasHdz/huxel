# HÜXEL =  JAX + HÜCKEL

## Optimization of the Hückel model *à la* machine learning

Using JAX, we created a fully differentiable Hückel model where all parameters are optimized with respect to some reference data.

## Beta functions (atom-atom interaction)

We implemented various *atom-atom* interaction functions.

1. constant (`--beta c`)

   ![equation](https://latex.codecogs.com/svg.image?\beta_{\ell,k}&space;=&space;\beta^{0}_{\ell,k})

2. exponential (`--beta exp`)

   ![equation](https://latex.codecogs.com/svg.image?\beta_{\ell,k}^{exp}&space;=&space;-\beta^{0}_{\ell,k}\exp^{-\Delta&space;R_{\ell,k}/y_{\ell,k}})

4. linear (`--beta linear`)

   ![equation](https://latex.codecogs.com/svg.image?\beta_{\ell,k}^{exp}&space;=&space;-\beta^{0}_{\ell,k}\left&space;(&space;1-&space;y_{\ell,k}^{-1}\Delta&space;R_{\ell,k}&space;\right&space;))

5. absolute exponential (`--beta exp_abs`)

   ![equation](https://latex.codecogs.com/svg.image?\beta_{\ell,k}^{exp}&space;=&space;-\beta^{0}_{\ell,k}\exp^{-\left|R_{\ell,k}-&space;R^{0}_{\ell,k}\right|/y_{\ell,k}&space;})

5. absolute linear (`--beta linear_abs`)

   ![equation](https://latex.codecogs.com/svg.image?\beta_{\ell,k}^{exp}&space;=&space;-\beta^{0}_{\ell,k}\left&space;(&space;1-&space;y_{\ell,k}^{-1}\left|R_{\ell,k}-&space;R^{0}_{\ell,k}\right|&space;\right&space;))

6. absolute exponential with no gradient for ![equation](https://latex.codecogs.com/svg.image?R^{0}_{\ell,k}) (`--beta exp_abs_freezeR`)

7. absolute linear with no gradient for ![equation](https://latex.codecogs.com/svg.image?R^{0}_{\ell,k}) (`--beta linear_abs_freezeR`)




## Quick start

`python main.py *args`

1. `--N`, number of training data points
2. `--l`, integer (for random number start `jax.random.PRNGKey(l)`)
3. `--lr`,  initial learning rate
4. `--batch_size`, batch size
5. `--job`, type of job (options `['opt','pred','pred_def']`)
   1. `opt` -> training
   2. `pred` -> prediction of test set
   3. `pred_def` -> prediction using default parameters (from literature)
6. `--obs`, objective to optimize, options `[homo_lumo,pol]`
7. `--beta`, beta function (`['c','linear','exp','linear_abs','exp_freezeR','exp_abs_freezeR','linear_freezeR','linear_abs_freezeR']`)

#### examples

```
python main.py --N 101 --obs pol --lr 2E-2 --l 0 --batch_size 32 --job opt --beta c -Wdecay h_x h_xy 

python main.py --N 50 --job pred --beta exp_abs 
```

## Reference data

The data set is a subset of GDB-13, all molecules encompass cyclic $\pi$-systems to increase representation of the relevant structural space that is appropriate for simulation via the Hückel method.

Total data:

1. training: 100,000  (file name:`/huxel/data/data_gdb13_training.npy`)
2. test: 40,000 (file name:`/huxel/data/data_gdb13_test.npy`)

**HOMO-LUMO gap**

Reference data was computed at the **TDA-SCS-![equation](https://latex.codecogs.com/svg.image?\omega)PBEPP86/def2-SVP** level of theory using Orca (version 5.0.1).

**Polarizability**

Reference data for molecular polarizabilities were computed using **dftd4** (version 3.4.0) via the default methodology, summing atomic polarizabilities.

<!-- ## Requirements
- JAX
```
pip install --upgrade pip
pip install --upgrade "jax[cpu]"
```
- FLAX 
```
pip install flax
```
- OPTAX (only for the optax branch)
```
pip install optax
``` -->
