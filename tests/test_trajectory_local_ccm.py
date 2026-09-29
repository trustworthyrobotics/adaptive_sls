import numpy as np

from baselines.ccm import BoxBounds, SolverConfig, TrajectoryLocalCCMSolver, UncertaintySet


def test_tvlqr_shapes_and_sampled_tube_contains_scalar_disturbance():
    def step(x, u):
        return np.array([x[0] + u[0]])

    def uncertain_step(x, u, parameter, disturbance):
        return step(x, u) + parameter + disturbance

    def linearize(x, u):
        del x, u
        return np.ones((1, 1)), np.ones((1, 1))

    solver = TrajectoryLocalCCMSolver(
        nominal_step=step,
        uncertain_step=uncertain_step,
        linearize=linearize,
        state_cost=np.eye(1),
        input_cost=np.eye(1),
        terminal_cost=np.eye(1),
        bounds=BoxBounds(
            state_lower=np.array([-10.0]),
            state_upper=np.array([10.0]),
            input_lower=np.array([-10.0]),
            input_upper=np.array([10.0]),
        ),
        uncertainty=UncertaintySet(np.array([0.1]), np.array([0.05])),
        config=SolverConfig(max_iterations=1, tube_directions=2),
    )
    states = np.zeros((4, 1))
    inputs = np.zeros((3, 1))
    gains, metrics = solver.tvlqr(states, inputs)
    radii, widths, input_widths, obstacle_widths = solver.propagate_tube(
        states, inputs, gains, metrics, obstacle_count=0
    )

    assert gains.shape == (3, 1, 1)
    assert metrics.shape == (4, 1, 1)
    assert widths.shape == (4, 1)
    assert input_widths.shape == (3, 1)
    assert obstacle_widths.shape == (4, 0)
    assert widths[1, 0] >= 0.15
    assert np.all(np.isfinite(radii))
    assert np.all(radii >= 0.0)


def test_short_solve_without_obstacles_is_robustly_feasible():
    def step(x, u):
        return np.array([x[0] + 0.2 * u[0]])

    def uncertain_step(x, u, parameter, disturbance):
        return step(x, u) + parameter + disturbance

    def linearize(x, u):
        del x, u
        return np.ones((1, 1)), np.array([[0.2]])

    solver = TrajectoryLocalCCMSolver(
        nominal_step=step,
        uncertain_step=uncertain_step,
        linearize=linearize,
        state_cost=np.eye(1),
        input_cost=np.eye(1),
        terminal_cost=np.eye(1),
        bounds=BoxBounds(
            state_lower=np.array([-2.0]),
            state_upper=np.array([2.0]),
            input_lower=np.array([-1.0]),
            input_upper=np.array([1.0]),
        ),
        uncertainty=UncertaintySet(np.array([0.01]), np.array([0.01])),
        config=SolverConfig(max_iterations=3, tube_directions=2),
    )
    result = solver.solve(
        initial_state=np.zeros(1),
        reference=np.zeros((5, 1)),
        initial_inputs=np.zeros((4, 1)),
    )

    assert result.states.shape == (5, 1)
    assert result.diagnostics["final_sampled_tube_constraints_feasible"] is True
    assert result.diagnostics["maximum_dynamics_defect"] < 1e-6
