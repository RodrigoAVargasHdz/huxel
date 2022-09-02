# HÜXEL =  JAX + HÜCKEL

Optimization of the Hückel model *à la* machine learning. 
Using JAX, we created a fully differentiable Huckel model where all parameters could be optimized with respect to some reference data.

# Beta functions
constant ('--beta c')

![equation](https://latex.codecogs.com/gif.latex?%5Clarge%20h_%7Bxy%7D%20%3D%20%5Cbeta_%7Bxy%7D)

exponential ('--beta exp')

![equation](https://latex.codecogs.com/gif.latex?%5Clarge%20h_%7Bxy%7D%20%3D%20%5Cbeta_%7Bxy%7D%5C%3Be%5E%7B-%5Cell%5E%7B-1%7D_%7Bxy%7D%28R_%7Bxy%7D-R%5E%7B0%7D_%7Bxy%7D%29%7D)

absolute exponential ('--beta exp_abs')

![equation](https://latex.codecogs.com/gif.latex?%5Clarge%20h_%7Bxy%7D%20%3D%20%5Cbeta_%7Bxy%7D%5C%3Be%5E%7B-%5Cell%5E%7B-1%7D_%7Bxy%7D%28%7CR_%7Bxy%7D-R%5E%7B0%7D_%7Bxy%7D%7C%29%7D)

absolute exponential with no gradient for ![equation](https://latex.codecogs.com/gif.latex?%5Csmall%20R%5E%7B0%7D_%7Bxy%7D) ('--beta exp_abs_freezeR')

linear ('--beta linear')

![equation](https://latex.codecogs.com/gif.latex?%5Clarge%20h_%7Bxy%7D%20%3D%20%5Cbeta_%7Bxy%7D%5C%3B%281-%5Cell%5E%7B-1%7D_%7Bxy%7D%28R_%7Bxy%7D-R%5E%7B0%7D_%7Bxy%7D%29%29)

absolute linear ('--beta linear_abs')

![equation](https://latex.codecogs.com/gif.latex?%5Clarge%20h_%7Bxy%7D%20%3D%20%5Cbeta_%7Bxy%7D%5C%3B%281-%5Cell%5E%7B-1%7D_%7Bxy%7D%28%7CR_%7Bxy%7D-R%5E%7B0%7D_%7Bxy%7D%7C%29%29)

absolute linear with no gradient for ![equation](https://latex.codecogs.com/gif.latex?%5Csmall%20R%5E%7B0%7D_%7Bxy%7D) ('--beta linear_abs_freezeR')


## Quickstart
```
python main.py --N 50 --lr 2E-2 --l 1 --batch_size 128 --job opt --beta exp_abs 
python main.py --N 50 --job pred --beta exp_abs 
```

## Requirements
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
```

