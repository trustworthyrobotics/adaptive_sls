"""Audit the certified CCM on the inverse-mass/scalar-disturbance model."""

from __future__ import annotations

import json

from .config import ExperimentConfig
from .official_ccm import OfficialCCM, audit_candidate_metric, parameter_lipschitz_constants


def main() -> None:
    config = ExperimentConfig()
    ccm = OfficialCCM()
    original_audit = audit_candidate_metric(ccm, config)
    if original_audit.passed:
        audit = original_audit
    else:
        adapted_rate = max(0.0, 0.95 * original_audit.minimum_sampled_contraction_rate)
        audit = audit_candidate_metric(ccm, config, requested_rate=adapted_rate)
    result = {
        "ccm_source": ccm.source,
        "original_rate_audit": original_audit.as_json(),
        "adapted_rate_audit": audit.as_json(),
    }
    result["parameter_lipschitz_constants"] = parameter_lipschitz_constants(
        ccm, config
    ).tolist()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
