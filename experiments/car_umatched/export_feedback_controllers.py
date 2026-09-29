"""Export the three car feedback laws into portable NPZ artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baselines.ccm import car_to_linearizing_coordinates, design_feedback_linearized_car_ccm


def read_states(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    columns = ("x_m", "y_m", "heading_rad", "speed_m_per_s")
    states = np.asarray([[float(row[column]) for column in columns] for row in rows])
    times = np.asarray([float(row["time_s"]) for row in rows])
    if len(states) < 2 or not np.all(np.isfinite(states)):
        raise ValueError(f"invalid nominal trajectory: {path}")
    return states, times


def export_ccm(
    run_dir: Path,
    destination: Path,
    *,
    parameter_bound: float,
    disturbance_scale: float,
    maximum_speed: float,
) -> tuple[int, float]:
    nominal_states, times = read_states(run_dir / "nominal_plan.csv")
    dt_values = np.diff(times)
    if not np.allclose(dt_values, dt_values[0]):
        raise ValueError("CCM nominal plan must use a uniform time grid")
    dt = float(dt_values[0])
    nominal_linear_states = np.asarray(
        [car_to_linearizing_coordinates(state) for state in nominal_states]
    )
    nominal_virtual_inputs = np.diff(nominal_linear_states[:, 2:], axis=0) / dt
    certificate = design_feedback_linearized_car_ccm(
        parameter_half_width=parameter_bound,
        continuous_disturbance_bound=disturbance_scale / dt,
        certified_maximum_speed=maximum_speed,
    )
    np.savez_compressed(
        destination,
        controller_type=np.asarray("feedback_linearized_ccm"),
        nominal_states=nominal_states,
        nominal_linear_states=nominal_linear_states,
        nominal_virtual_inputs=nominal_virtual_inputs,
        metric=certificate.metric,
        differential_gain=certificate.differential_gain,
        contraction_rate=np.asarray(certificate.contraction_rate),
        disturbance_bound=np.asarray(certificate.disturbance_bound),
        position_support=np.asarray(certificate.position_support),
        velocity_support=np.asarray(certificate.velocity_support),
        virtual_input_support=np.asarray(certificate.virtual_input_support),
        certificate_max_eigenvalue=np.asarray(certificate.certificate_max_eigenvalue),
        maximum_disturbance_metric_norm=np.asarray(
            certificate.maximum_disturbance_metric_norm
        ),
        dt=np.asarray(dt),
        parameter_uncertainty_bound=np.asarray(parameter_bound),
        exogenous_disturbance_scale=np.asarray(disturbance_scale),
    )
    return len(nominal_states) - 1, dt


def export_adaptive(run_dir: Path, destination: Path) -> tuple[int, float]:
    source = run_dir / "controller.npz"
    if not source.is_file():
        raise FileNotFoundError(
            f"{source} is missing; rerun car_adaptive_leb.py after the controller-export update"
        )
    with np.load(source, allow_pickle=False) as controller:
        payload = {name: controller[name] for name in controller.files}
    if str(payload["controller_type"]) != "adaptive_leb_sls":
        raise ValueError(f"unexpected adaptive controller type in {source}")
    np.savez_compressed(destination, **payload)
    return int(payload["nominal_inputs"].shape[0]), float(payload["dt"])


def export_pavone(run_dir: Path, destination: Path) -> tuple[int, float]:
    source = run_dir / "policy.npz"
    if not source.is_file():
        raise FileNotFoundError(f"Pavone controller is missing: {source}")
    nominal_states, times = read_states(run_dir / "nominal_plan.csv")
    dt = float(np.diff(times)[0])
    with np.load(source, allow_pickle=False) as policy:
        payload = {name: policy[name] for name in policy.files}
    np.savez_compressed(
        destination,
        controller_type=np.asarray("pavone_affine_disturbance_feedback"),
        dt=np.asarray(dt),
        nominal_states=payload["nominal_states"],
        nominal_inputs=payload["nominal_inputs"],
        disturbance_gains=payload["disturbance_gains"],
        state_response=payload["state_response"],
        linearization_a=payload["linearization_a"],
        linearization_b=payload["linearization_b"],
        disturbance_half_width=payload["disturbance_half_width"],
    )
    return int(payload["nominal_inputs"].shape[0]), dt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    base = Path(__file__).with_name("comparison_results")
    parser.add_argument("--ccm-dir", type=Path, default=base / "certified_ccm")
    parser.add_argument(
        "--adaptive-dir", type=Path, default=base / "adaptive_true__leb_true"
    )
    parser.add_argument(
        "--pavone-dir",
        type=Path,
        default=Path(__file__).with_name("pavone_tight_no_ccm_results")
        / "pavone_affine_df",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).with_name("saved_controllers")
    )
    parser.add_argument("--parameter-bound", type=float, default=0.075)
    parser.add_argument("--disturbance-scale", type=float, default=0.00030)
    parser.add_argument("--maximum-speed", type=float, default=2.0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    outputs = {
        "ccm": args.output_dir / "ccm_controller.npz",
        "adaptive_leb": args.output_dir / "adaptive_leb_controller.npz",
        "pavone": args.output_dir / "pavone_controller.npz",
    }
    dimensions = {
        "ccm": export_ccm(
            args.ccm_dir,
            outputs["ccm"],
            parameter_bound=args.parameter_bound,
            disturbance_scale=args.disturbance_scale,
            maximum_speed=args.maximum_speed,
        ),
        "adaptive_leb": export_adaptive(args.adaptive_dir, outputs["adaptive_leb"]),
        "pavone": export_pavone(args.pavone_dir, outputs["pavone"]),
    }
    if len(set(dimensions.values())) != 1:
        raise ValueError(f"controller horizon/dt mismatch: {dimensions}")

    manifest = {
        "schema_version": 1,
        "horizon": dimensions["ccm"][0],
        "dt": dimensions["ccm"][1],
        "parameter_half_width": args.parameter_bound,
        "exogenous_disturbance_half_width": args.disturbance_scale,
        "controllers": {name: str(path.resolve()) for name, path in outputs.items()},
        "source_directories": {
            "ccm": str(args.ccm_dir.resolve()),
            "adaptive_leb": str(args.adaptive_dir.resolve()),
            "pavone": str(args.pavone_dir.resolve()),
        },
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    print(f"Saved feedback controllers to {args.output_dir}")


if __name__ == "__main__":
    main()
