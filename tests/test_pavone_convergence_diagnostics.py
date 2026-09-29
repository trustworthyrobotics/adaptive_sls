import numpy as np

from baselines.pavone_mpc import (
    AffineDisturbanceFeedbackSolver,
    BoxBounds,
    SolverConfig,
)
from experiments.car_umatched.car_pavone_mpc import load_ccm_seed


def test_pavone_solver_records_convergence_history() -> None:
    solver = AffineDisturbanceFeedbackSolver(
        nominal_step=lambda state, control: state + control,
        linearize=lambda state, control: (np.ones((1, 1)), np.ones((1, 1))),
        state_cost=np.eye(1),
        input_cost=np.eye(1),
        terminal_cost=np.eye(1),
        bounds=BoxBounds(
            state_lower=np.array([-10.0]),
            state_upper=np.array([10.0]),
            input_lower=np.array([-10.0]),
            input_upper=np.array([10.0]),
        ),
        disturbance_half_width=np.array([0.01]),
        config=SolverConfig(max_iterations=4, convergence_tolerance=1e-3),
    )

    result = solver.solve(
        initial_state=np.array([0.0]),
        reference=np.zeros((3, 1)),
        initial_inputs=np.full((2, 1), 0.5),
        initial_states=np.array([[0.0], [0.5], [1.0]]),
    )

    assert 1 <= result.iterations <= 4
    assert len(result.convergence_history) == result.iterations
    assert result.diagnostics["termination_reason"] in {
        "converged",
        "maximum iterations reached",
    }
    final = result.convergence_history[-1]
    assert final["iterate_change_inf"] == max(
        final["state_change_inf"], final["input_change_inf"]
    )
    assert "virtual_control_linf" in final
    assert final["parameterized_problem_dpp"] is True
    assert result.diagnostics["warm_start_enabled"] is True
    assert "primary_convergence_blocker" in result.diagnostics


def test_ccm_nominal_plan_can_seed_pavone(tmp_path) -> None:
    ccm_plan = tmp_path / "nominal_plan.csv"
    ccm_plan.write_text(
        "step,time_s,x_m,y_m,heading_rad,speed_m_per_s\n"
        "0,0.0,0.0,1.0,-1.5,0.4\n"
        "1,0.1,0.01,0.9,-1.4,0.5\n"
    )

    states, inputs, diagnostics = load_ccm_seed(
        ccm_plan,
        initial_state=np.array([0.0, 1.0, -1.5, 0.4]),
        goal_state=np.array([0.01, 0.9, -1.4, 0.5]),
        horizon=2,
        dt=0.05,
        terminal_tolerance=1e-6,
    )

    assert states.shape == (3, 4)
    assert inputs.shape == (2, 2)
    assert np.allclose(states[1], [0.005, 0.95, -1.45, 0.45])
    assert np.allclose(inputs, 1.0)
    assert diagnostics["warmstart_source"] == "CCM nominal trajectory"
    assert diagnostics["ccm_warmstart_interpolated"] is True
