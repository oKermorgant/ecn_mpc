# Model-predictive control lab

**work in progress**

This package shows how to perform MPC for a car-like vehicle tracking trajectories on the Centrale Nantes campus.

3 approaches are proposed:

- hand-made linear MPC in C++ or Python using QP solver (**TODO**)
- [Acados API](https://docs.acados.org/) from Python with C-generated code
- [dompc API](https://www.do-mpc.com/en/latest) from Python with symbolic models

## Running the simulation

```bash
ros2 launch ecn_mpc sim_launch.py
```

## The control node

The node is run through `ros2 run ecn_mpc control.py`.

It should instantiate either your Acados or dompc code to have the car track the required path.

## Examples

`Acados` and `dompc` API examples lie in the `examples` folder.
