import math

import numpy as np

from src.model_suite import (
    FrequencyBaselineModel,
    ModelPrediction,
    MixtureDensityNetwork,
    NormalizingFlowModel,
    TransformerEncoderModel,
    UniformBaselineModel,
    XGBoostProxyModel,
    aggregate_model_predictions,
    model_null_hypothesis_benchmark,
)


def test_uniform_baseline_is_uniform() -> None:
    model = UniformBaselineModel(domain_size=8)
    probabilities = model.predict()
    assert np.allclose(probabilities, np.full(8, 1.0 / 8))
    assert math.isclose(float(probabilities.sum()), 1.0, abs_tol=1e-12)


def test_frequency_baseline_respects_empirical_distribution() -> None:
    model = FrequencyBaselineModel([1, 2, 3, 4])
    probabilities = model.predict()
    expected = np.array([0.1, 0.2, 0.3, 0.4])
    assert np.allclose(probabilities, expected)


def test_transformer_model_outputs_probability_distribution() -> None:
    model = TransformerEncoderModel(domain_size=12)
    probabilities = model.predict(np.linspace(1.0, 12.0, 12))
    assert probabilities.shape == (12,)
    assert math.isclose(float(probabilities.sum()), 1.0, abs_tol=1e-12)
    assert np.all(probabilities >= 0.0)


def test_xgboost_proxy_model_returns_valid_probability_vector() -> None:
    model = XGBoostProxyModel(domain_size=16)
    probabilities = model.predict(np.ones(16))
    assert probabilities.shape == (16,)
    assert math.isclose(float(probabilities.sum()), 1.0, abs_tol=1e-12)


def test_mdn_and_nf_produce_valid_probability_vectors() -> None:
    mdn = MixtureDensityNetwork(domain_size=10)
    nf = NormalizingFlowModel(domain_size=10)
    mdn_probs = mdn.predict(np.linspace(0.0, 1.0, 10))
    nf_probs = nf.predict(np.linspace(0.0, 1.0, 10))
    assert mdn_probs.shape == (10,)
    assert nf_probs.shape == (10,)
    assert math.isclose(float(mdn_probs.sum()), 1.0, abs_tol=1e-12)
    assert math.isclose(float(nf_probs.sum()), 1.0, abs_tol=1e-12)


def test_model_null_benchmark_matches_uniform_expectation() -> None:
    summary = model_null_hypothesis_benchmark(domain_size=8, trials=20, seed=7)
    assert summary['theoretical_entropy'] == 3.0
    assert summary['mean_empirical_entropy'] > 0.0
    assert summary['trials'] == 20


def test_ensemble_weights_sum_to_one() -> None:
    first = ModelPrediction(model_name='a', probabilities=np.array([0.8, 0.2]), metadata={})
    second = ModelPrediction(model_name='b', probabilities=np.array([0.2, 0.8]), metadata={})
    combined = aggregate_model_predictions([first, second], weights=[0.25, 0.75])
    assert math.isclose(float(combined.probabilities.sum()), 1.0, abs_tol=1e-12)
    assert combined.model_name == 'ensemble'
