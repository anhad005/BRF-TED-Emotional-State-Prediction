"""
src/brf.py

Behavioral Rhythm Fusion (BRF) Feature Extraction
"""

from __future__ import annotations

import numpy as np

FEATURE_COLS = [
    # Rhythm Features
    "ibd",
    "isie",
    "cii",
    "mlue",

    # Aggregates
    "session_count",
    "screen_time_min",
    "unlock_freq",

    # Temporal
    "sleep_proxy_h",
    "activity_level",

    # Social
    "conv_count",
    "coloc_diversity",

    # EMA Auxiliary
    "ema_stress",
    "ema_social",
]


class BRFExtractor:
    """
    Behavioral Rhythm Fusion Extractor.

    Learns per-student reference statistics from the first
    ref_days days and performs z-score normalization.
    """

    def __init__(
        self,
        ref_days: int = 14,
        eps: float = 1e-6,
        features=None
    ):
        self.ref_days = ref_days
        self.eps = eps
        self.features = features if features else list(FEATURE_COLS)

        self._means = {}
        self._stds = {}

        self._global_mean = None
        self._global_std = None

    def fit(self, df):
        reference_df = df[df["day"] < self.ref_days]

        global_ref = reference_df[self.features]

        self._global_mean = global_ref.mean().values
        self._global_std = global_ref.std().values

        for student_id, group in reference_df.groupby("student_id"):
            values = group[self.features].values

            self._means[student_id] = values.mean(axis=0)
            self._stds[student_id] = values.std(axis=0)

        return self

    def transform(self, df) -> np.ndarray:
        raw_features = df[self.features].values.astype(np.float64)

        output = np.empty_like(
            raw_features,
            dtype=np.float32
        )

        student_ids = df["student_id"].values

        for idx, sid in enumerate(student_ids):

            mu = self._means.get(
                sid,
                self._global_mean
            )

            sigma = self._stds.get(
                sid,
                self._global_std
            )

            output[idx] = (
                (raw_features[idx] - mu)
                / (sigma + self.eps)
            ).astype(np.float32)

        return output

    def fit_transform(self, df) -> np.ndarray:
        return self.fit(df).transform(df)

    @property
    def num_features(self) -> int:
        return len(self.features)

    def get_feature_names(self):
        return list(self.features)

    def __repr__(self):
        return (
            f"BRFExtractor("
            f"ref_days={self.ref_days}, "
            f"features={len(self.features)})"
        )