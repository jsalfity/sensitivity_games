# Sensitivity Games
Early experimentation for prototyping and proof of concept for a game theory / optimal control theory approach to finding robust controllers. 

## Documentation
To Do

## Dependencies 
To Do

## Getting Started
### Experimentation
Experimentation for point mass regulation and training held at `point_mass_regulation_train.py`.

Arg parser controls these variables:

```py
("--viz", default=False)
("--verbose", default=True)
("--verbose_every", default=10000)
("--epochs", default=500)
("--lr", default=1e-3)
("--T", default=25)
```

Variables hardcoded in the file:
```py
x0 = Tensor([5, 0, 5, 0])
goal = [0, 0, 0, 0]
```

To view a single trajectory with different K values, run `point_mass_regulation_viztraj.py`


### Test
`test_point_mass.py` launches the experiment and training, then checks: 
```py
def test_point_mass_experiment_K_grad_should_converge_to_zero():

def test_point_mass_experiment_theta_grad_should_converge_to_zero():

def test_controller_k_curvature_should_be_positive():

def test_dm_curvature_should_be_positive():

def test_point_mass_should_converge_to_dare_k():
```

Be sure to set variables in `point_mass_regulation.py`