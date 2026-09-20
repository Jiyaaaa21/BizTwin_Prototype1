"""
CalibrationEngine - Final Design Section 7. Credibility-weighted blending,
Z = n/(n+k), mathematically identical to Bayesian Beta-Binomial posterior
updating: Calibrated = Z*Observed + (1-Z)*Prior. Every adjustment is logged
(CalibrationLogEntry) - an audit trail, never a silent overwrite (Guiding
Principle 7).
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .parameters import Parameter, ParameterSet


@dataclass
class CalibrationLogEntry:
    parameter_name: str
    old_value: float
    new_value: float
    n_observations: float
    k_credibility: float
    z_weight: float
    data_source: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class CalibrationEngine:
    """
    Operates on a ParameterSet, returning updated copies (never mutating a
    shared registry in place - Final Design Section 7 / Guiding Principle 7:
    every run's parameters are a known, frozen snapshot).
    """

    def __init__(self, params: ParameterSet):
        self.params = params
        self.log: list[CalibrationLogEntry] = []

    def credibility_weight(self, n_observations: float, k: float) -> float:
        """Z = n / (n + k) - the prior's implied sample size is k."""
        if n_observations + k <= 0:
            return 0.0
        return n_observations / (n_observations + k)

    def calibrate(
        self,
        parameter_name: str,
        observed_value: float,
        n_observations: float,
        k_credibility: float,
        data_source: str,
    ) -> ParameterSet:
        """
        Observed Behavior -> Error Measurement -> Parameter Update ->
        Recalibrated Model (Final Design Section 7's cycle diagram).
        Returns a NEW ParameterSet; self.params is updated to point at it.
        """
        registry = self.params.as_dict()
        if parameter_name not in registry:
            raise KeyError(f"Unknown parameter: {parameter_name}")
        param = registry[parameter_name]

        z = self.credibility_weight(n_observations, k_credibility)
        calibrated_value = param.clamp(z * observed_value + (1 - z) * param.value)

        new_params = copy.deepcopy(self.params)
        new_param = getattr(new_params, parameter_name)
        old_value = new_param.value
        new_param.value = calibrated_value

        self.log.append(CalibrationLogEntry(
            parameter_name=parameter_name, old_value=old_value,
            new_value=calibrated_value, n_observations=n_observations,
            k_credibility=k_credibility, z_weight=z, data_source=data_source,
        ))
        self.params = new_params
        return new_params

    def worked_example_renewal(self) -> CalibrationLogEntry:
        """
        The Contract component's own calibration story - added per project
        review: 'calibration is the make-or-break component,' not the
        number of classes built. Prior renewal_probability=90%, observed
        cohort renewal rate=82% on n=150 term-boundary observations,
        k=50 -> Z=0.75 -> calibrated=84.0%. Same Bayesian credibility-blend
        mechanism as worked_example_lead_conversion(), applied to a second
        parameter, demonstrating the method generalizes rather than being a
        one-off trick for lead conversion alone.
        """
        registry = self.params.as_dict()
        original = copy.deepcopy(self.params.renewal_probability)
        self.params.renewal_probability.value = 0.90
        self.calibrate(
            "renewal_probability", observed_value=0.82, n_observations=150,
            k_credibility=50, data_source="Worked example - Contract renewal calibration",
        )
        entry = self.log[-1]
        self.params.renewal_probability = original
        return entry

    def worked_example_lead_conversion(self) -> CalibrationLogEntry:
        """
        Reproduces the Final Design's own worked example exactly: prior=15%,
        n=200 observed leads, k=75 -> Z=0.727 -> calibrated=17.2%.
        Demonstrates the mechanism rather than asserting the number.
        """
        registry = self.params.as_dict()
        prior = registry["r1_lead_to_qo"].value
        # Temporarily treat the prior as 15% for this illustrative reproduction,
        # matching the project brief's own example, then restore afterward.
        original = copy.deepcopy(self.params.r1_lead_to_qo)
        self.params.r1_lead_to_qo.value = 0.15
        entry_params = self.calibrate(
            "r1_lead_to_qo", observed_value=0.18, n_observations=200,
            k_credibility=75, data_source="Worked example, Final Design Section 7",
        )
        entry = self.log[-1]
        self.params.r1_lead_to_qo = original  # restore the real default
        return entry
