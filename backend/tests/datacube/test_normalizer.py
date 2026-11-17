"""
Unit tests for ModalityNormalizer and zero-NaN sentinel imputation.
"""
from __future__ import annotations

import numpy as np
import pytest

from datacube.normalizer import ModalityNormalizer


class TestModalityNormalizer:
    def test_fit_and_statistics(self, synthetic_unified_cube):
        normalizer = ModalityNormalizer()
        # Train indices: steps 0 to 7
        train_indices = list(range(8))
        normalizer.fit(synthetic_unified_cube, train_indices=train_indices)

        assert normalizer.is_fitted()
        assert "tir1" in normalizer.stats
        assert "composite_reflectivity" in normalizer.stats
        assert "flash_count" in normalizer.stats
        assert "cape" in normalizer.stats

        tir_stats = normalizer.stats["tir1"]
        assert tir_stats.valid_count > 0
        assert 200.0 <= tir_stats.mean <= 280.0
        assert tir_stats.std > 0.0

    def test_transform_zero_nan_guarantee(self, synthetic_unified_cube):
        normalizer = ModalityNormalizer().fit(synthetic_unified_cube)

        raw_tir1 = synthetic_unified_cube["tir1"].values[0].copy()
        # Ensure raw has NaNs
        assert np.any(np.isnan(raw_tir1))

        vmask = synthetic_unified_cube["tir1_valid_mask"].values[0]
        norm_tir1 = normalizer.transform("tir1", raw_tir1, valid_mask=vmask)

        # CRITICAL TEST: Transformed array must contain ZERO NaNs
        assert not np.any(np.isnan(norm_tir1)), "Normalized tensor must contain zero NaNs!"

        # Missing locations must be imputed with sentinel 0.0
        unobserved = ~vmask
        assert np.all(norm_tir1[unobserved] == 0.0)

    def test_missing_modality_transform_all_nan(self, synthetic_unified_cube):
        """Even when an entire modality (e.g. radar) is absent, transform must produce clean zero-NaN array."""
        normalizer = ModalityNormalizer().fit(synthetic_unified_cube)

        all_nan = np.full((64, 64), np.nan, dtype=np.float32)
        norm_radar = normalizer.transform("composite_reflectivity", all_nan)

        assert not np.any(np.isnan(norm_radar))
        assert np.all(norm_radar == 0.0)

    def test_inverse_transform(self, synthetic_unified_cube):
        normalizer = ModalityNormalizer().fit(synthetic_unified_cube)
        raw_cape = np.full((10, 10), 2000.0, dtype=np.float32)
        norm_cape = normalizer.transform("cape", raw_cape)

        reconstructed = normalizer.inverse_transform("cape", norm_cape)
        assert np.allclose(reconstructed, raw_cape, rtol=1e-2)

    def test_save_and_load_json(self, synthetic_unified_cube, datacube_tmp_dir):
        normalizer = ModalityNormalizer().fit(synthetic_unified_cube)
        json_path = datacube_tmp_dir / "test_norm_stats.json"

        normalizer.save_to_json(json_path)
        assert json_path.exists()

        loaded = ModalityNormalizer.load_from_json(json_path)
        assert loaded.is_fitted()
        assert loaded.stats["tir1"].mean == pytest.approx(normalizer.stats["tir1"].mean)
