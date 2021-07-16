# Sensitivity Games
Early experimentation for prototyping and proof of concept for a game theory / optimal control theory approach to finding robust controllers. 

## Documentation
To Do

## Dependencies 
To Do

## Getting Started
### Experimentation and Training
Experimentation for point mass regulation and training held at `point_mass_regulation_train.py`.

Arg parser controls these variables:

```py
parser.add_argument("--viz", type=bool, default=False)
parser.add_argument("--verbose", type=bool, default=True)
parser.add_argument("--verbose_freq", type=int, default=500)
parser.add_argument("--epochs", type=int, default=500)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--T", type=int, default=10)
parser.add_argument("--x0", nargs='+', type=float, default=[5, 0, 5, 0])
parser.add_argument("--xf", nargs='+', type=float, default=[0, 0, 0, 0])
parser.add_argument("--block", type=bool, default=False)
```


### Trajectory Visualization after training
To view a single trajectory with different K values, run `point_mass_regulation_viztraj.py`

Arg parser controls these variables:
```py
parser.add_argument("--viz", type=bool, default=True)
parser.add_argument("--T", type=int, default=10)
parser.add_argument("--K", nargs='+', type=float, default=dare_K)
parser.add_argument("--dm", type=float, default=0.0)
parser.add_argument("--x0", nargs='+', type=float, default=[5, 0, 10, 0])
parser.add_argument("--xf", nargs='+', type=float, default=[0, 0, 0, 0])
parser.add_argument("--block", type=bool, default=True)
```

Ex: 
### Test
`test_point_mass.py` launches the experiment and training, then checks: 
```py
def test_point_mass_experiment_K_grad_should_converge_to_zero():

def test_point_mass_experiment_theta_grad_should_converge_to_zero():

def test_controller_k_curvature_should_be_positive():

def test_dm_curvature_should_be_positive():

def test_point_mass_should_converge_to_dare_k():
```

Be sure to set variables in `point_mass_regulation_train.py`