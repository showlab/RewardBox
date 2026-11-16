"""Metric tests.  ordering.py is covered by test_core.py::TestMetrics; this file only
covers the newer detection / ranking / shape / cost metrics.
"""
from __future__ import annotations

import numpy as np
import pytest

from rewardbox.metrics.cost import (CostBreakdown, latency_per_backend_call,
                                    latency_per_input_frame, latency_per_output_point)
from rewardbox.metrics.detection import (UndefinedMetricError, balanced_accuracy,
                                         success_auroc, success_f1)
from rewardbox.metrics.ranking import (all_pairs_accuracy, paired_accuracy,
                                       preference_accuracy)
from rewardbox.metrics.shape import noise_robustness, smoothness, terminal_saturation


# ---------------------------------------------------------------- detection
class TestSuccessAUROC:
    def test_perfect_separation(self):
        assert success_auroc([0, 1, 2, 3], [0, 0, 1, 1]) == 1.0

    def test_complete_reverse(self):
        assert success_auroc([0, 1, 2, 3], [1, 1, 0, 0]) == 0.0

    def test_ties_score_half(self):
        """Complete ties (symmetric, fully overlapping positive/negative scores) -> 0.5, favouring neither side."""
        assert success_auroc([1, 1, 2, 2], [0, 1, 0, 1]) == 0.5

    def test_single_class_is_undefined(self):
        with pytest.raises(UndefinedMetricError):
            success_auroc([0, 1, 2], [1, 1, 1])
        with pytest.raises(UndefinedMetricError):
            success_auroc([0, 1, 2], [0, 0, 0])

    def test_nonfinite_scores_rejected(self):
        with pytest.raises(ValueError, match="non-finite"):
            success_auroc([0, float("nan"), 2], [0, 1, 1])

    def test_labels_must_be_binary(self):
        with pytest.raises(ValueError, match="0/1"):
            success_auroc([0, 1, 2], [0, 1, 2])

    def test_length_mismatch_rejected(self):
        with pytest.raises(ValueError, match="length mismatch"):
            success_auroc([0, 1, 2], [0, 1])

    def test_invariant_under_affine_transform(self):
        """AUROC is invariant under strictly monotone transforms (min-max included); that is why it is the detection metric."""
        rng = np.random.default_rng(0)
        raw = rng.normal(size=20)
        labels = (rng.random(20) > 0.5).astype(int)
        base = success_auroc(raw, labels)
        scaled = raw * 3.7 + 100.0          # positive affine
        mm = (raw - raw.min()) / (raw.max() - raw.min())   # min-max
        assert success_auroc(scaled, labels) == base
        assert success_auroc(mm, labels) == pytest.approx(base)


class TestSuccessF1:
    def test_perfect(self):
        assert success_f1([0, 0, 1, 1], [0, 0, 1, 1], threshold=0.5) == 1.0

    def test_no_positive_predictions_returns_zero(self):
        """With tp=0 the standard convention returns 0.0, not a "plausible" high score."""
        assert success_f1([0, 0, 0, 0], [0, 0, 1, 1], threshold=0.5) == 0.0

    def test_complete_reverse_returns_zero(self):
        assert success_f1([0, 0, 1, 1], [1, 1, 0, 0], threshold=0.5) == 0.0

    def test_threshold_is_required(self):
        with pytest.raises(TypeError):
            success_f1([0, 1], [0, 1])   # threshold missing

    def test_nonfinite_threshold_rejected(self):
        with pytest.raises(ValueError, match="threshold"):
            success_f1([0, 1], [0, 1], threshold=float("nan"))


class TestBalancedAccuracy:
    def test_perfect(self):
        assert balanced_accuracy([0, 0, 1, 1], [0, 0, 1, 1], threshold=0.5) == 1.0

    def test_complete_reverse(self):
        assert balanced_accuracy([0, 0, 1, 1], [1, 1, 0, 0], threshold=0.5) == 0.0

    def test_single_class_is_undefined(self):
        with pytest.raises(UndefinedMetricError):
            balanced_accuracy([0, 1, 2], [1, 1, 1], threshold=0.5)

    def test_threshold_is_required(self):
        with pytest.raises(TypeError):
            balanced_accuracy([0, 1], [0, 1])


# ---------------------------------------------------------------- ranking
class TestPairedAccuracy:
    def test_all_correct(self):
        assert paired_accuracy([1, 2, 3], [0, 1, 2]) == 1.0

    def test_all_wrong(self):
        assert paired_accuracy([0, 1, 2], [1, 2, 3]) == 0.0

    def test_ties_use_tie_value(self):
        assert paired_accuracy([1, 1], [1, 1], tie_value=0.5) == 0.5
        assert paired_accuracy([1, 1], [1, 1], tie_value=0.0) == 0.0

    def test_length_mismatch_rejected(self):
        with pytest.raises(ValueError, match="must have the same length"):
            paired_accuracy([1, 2, 3], [0, 1])

    def test_tie_value_must_be_probability(self):
        with pytest.raises(ValueError, match="tie_value"):
            paired_accuracy([1], [1], tie_value=2)


class TestAllPairsAccuracy:
    def test_perfect_separation(self):
        assert all_pairs_accuracy([5, 6, 7], [1, 2, 3]) == 1.0

    def test_complete_overlap_reverse(self):
        assert all_pairs_accuracy([1, 2, 3], [5, 6, 7]) == 0.0

    def test_matches_auroc_on_equivalent_data(self):
        """all_pairs_accuracy(pos, neg) and success_auroc(scores, labels) are mathematically
        equivalent on the same data, only the interfaces differ."""
        pos, neg = [3.0, 5.0, 1.0], [2.0, 0.5]
        scores = pos + neg
        labels = [1, 1, 1, 0, 0]
        assert all_pairs_accuracy(pos, neg) == pytest.approx(success_auroc(scores, labels))


class TestPreferenceAccuracy:
    def test_all_correct(self):
        assert preference_accuracy([1.0, 0.5, 0.1]) == 1.0

    def test_all_wrong(self):
        assert preference_accuracy([-1.0, -0.5]) == 0.0

    def test_zero_is_tie(self):
        assert preference_accuracy([1.0, 0.0, -1.0]) == pytest.approx(1 / 3 + 0.5 / 3)


# ---------------------------------------------------------------- shape
class TestSmoothness:
    def test_constant_curve_is_zero(self):
        assert smoothness([0.5, 0.5, 0.5, 0.5]) == 0.0

    def test_monotonic_linear_smoother_than_jagged(self):
        linear = smoothness([0.0, 0.25, 0.5, 0.75, 1.0])
        jagged = smoothness([0.0, 1.0, 0.0, 1.0, 0.0])
        assert linear > jagged

    def test_invalid_scale_rejected(self):
        with pytest.raises(ValueError, match="scale"):
            smoothness([0.0, 1.0], scale="percent")

    def test_short_curve_returns_zero(self):
        assert smoothness([0.5]) == 0.0

    def test_empty_curve_rejected(self):
        with pytest.raises(ValueError, match="must not be empty"):
            smoothness([])


class TestTerminalSaturation:
    def test_saturated_curve_is_one(self):
        x = [0.0, 0.3, 0.6, 0.9, 1.0, 1.0, 1.0]
        assert terminal_saturation(x, window=3, target=1.0) == pytest.approx(1.0)

    def test_unsaturated_curve_is_low(self):
        x = [0.0, 0.1, 0.2, 0.3]
        val = terminal_saturation(x, window=1, target=1.0)
        assert val < 0.8

    def test_window_larger_than_curve_rejected(self):
        with pytest.raises(ValueError, match="exceeds the curve length"):
            terminal_saturation([0.0, 1.0], window=5, target=1.0)

    def test_window_required_explicit(self):
        with pytest.raises(TypeError):
            terminal_saturation([0.0, 1.0])

    def test_constant_curve_no_division_by_zero(self):
        assert terminal_saturation(
            [0.5, 0.5, 0.5], window=2, target=0.5) == pytest.approx(1.0)


class TestNoiseRobustness:
    def _out(self, progress, frame_indices=(0, 2, 4)):
        from rewardbox.core.types import RewardOutput, SamplingSpec
        return RewardOutput(
            progress=progress, frame_indices=frame_indices, model_name="m",
            sampling=SamplingSpec(num_frames=3))

    def test_fraction_within_explicit_tolerance(self):
        clean = self._out([0.0, 0.5, 1.0])
        noisy = self._out([0.01, 0.7, 0.95])
        assert noise_robustness(clean, noisy, tolerance=0.1) == pytest.approx(2 / 3)

    def test_grid_mismatch_rejected(self):
        with pytest.raises(ValueError, match="frame_indices"):
            noise_robustness(
                self._out([0, 1, 2]), self._out([0, 1, 2], (0, 3, 4)),
                tolerance=0.1)

    def test_requires_real_outputs_and_explicit_tolerance(self):
        with pytest.raises(TypeError):
            noise_robustness([0, 1], [0, 1], tolerance=0.1)
        with pytest.raises(TypeError):
            noise_robustness(self._out([0, 1, 2]), self._out([0, 1, 2]))


# ---------------------------------------------------------------- cost
class TestLatencyMetrics:
    def test_per_input_frame(self):
        assert latency_per_input_frame(10.0, 5) == 2.0

    def test_per_output_point(self):
        assert latency_per_output_point(9.0, 3) == 3.0

    def test_per_backend_call(self):
        assert latency_per_backend_call(45.0, 45) == 1.0

    def test_zero_denominator_rejected(self):
        with pytest.raises(ValueError):
            latency_per_input_frame(1.0, 0)
        with pytest.raises(ValueError):
            latency_per_output_point(1.0, 0)
        with pytest.raises(ValueError):
            latency_per_backend_call(1.0, 0)

    def test_different_denominators_give_different_numbers(self):
        """Motivation: Robo-Dopamine's 3x(T-1) calls vs Robometer's single forward pass must not
        be compared with the same denominator."""
        total = 60.0
        per_frame = latency_per_input_frame(total, 16)
        per_call = latency_per_backend_call(total, 45)
        assert per_frame != per_call

    @pytest.mark.parametrize("bad", (-1.0, float("nan"), float("inf")))
    def test_invalid_latency_rejected(self, bad):
        with pytest.raises(ValueError, match="total_latency"):
            latency_per_input_frame(bad, 1)


class TestCostBreakdown:
    def test_to_dict_roundtrip_fields(self):
        cb = CostBreakdown(bridge_s=1.0, inference_s=9.0, total_s=10.0, peak_vram_gb=8.5)
        d = cb.to_dict()
        assert d == {"bridge_s": 1.0, "inference_s": 9.0, "total_s": 10.0,
                    "peak_vram_gb": 8.5}

    def test_defaults(self):
        cb = CostBreakdown()
        assert cb.to_dict() == {"bridge_s": 0.0, "inference_s": 0.0, "total_s": 0.0,
                                "peak_vram_gb": None}

    def test_negative_cost_rejected(self):
        with pytest.raises(ValueError, match="bridge_s"):
            CostBreakdown(bridge_s=-1)
