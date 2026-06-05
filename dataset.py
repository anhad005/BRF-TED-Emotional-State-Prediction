import numpy as np
import pandas as pd

from .config import N_STUDENTS, N_DAYS
from .utils import shannon_entropy

EXAM_DAYS = [34, 35, 36, 69, 70]
WINDOWS_PER_DAY = 22
def _circadian(sid):
    rng = np.random.default_rng(sid)
    b = np.zeros(24)

    m = rng.integers(7, 10)
    b[m:m+3] = rng.uniform(0.5, 0.9)

    e = rng.integers(19, 22)
    b[e:e+2] = rng.uniform(0.6, 1.0)

    ss = int(rng.integers(23, 26)) % 24

    for h in range(6):
        b[(ss+h) % 24] *= 0.1

    return np.clip(
        b + rng.uniform(0, 0.1, 24),
        0,
        1
    )

def _stress(day, sid):
    rng = np.random.default_rng(sid * 1000 + day)

    dist = min(abs(day - e) for e in EXAM_DAYS)

    return float(
        np.clip(
            rng.uniform(0.1, 0.4)
            + rng.uniform(0.5, 1.5)
            * np.exp(-dist / 5)
            * rng.uniform(0.4, 0.7),
            0,
            1
        )
    )

def generate_synthetic():
    print("=" * 55)
    print("  WARNING: SYNTHETIC DATA — pipeline testing only")
    print("  Do NOT report these results in your paper.")
    print("  Get the real dataset: studentlife.cs.dartmouth.edu")
    print("=" * 55)

    rows = []

    for sid in range(N_STUDENTS):

        tmpl = _circadian(sid)
        baseline_sleep = np.random.uniform(6, 8)

        for day in range(N_DAYS):

            stress = _stress(day, sid)

            sleep = float(
                np.clip(
                    baseline_sleep
                    - stress * np.random.uniform(1, 3)
                    + np.random.normal(0, 0.3),
                    3,
                    9
                )
            )

            cii = float(
                np.clip(
                    stress * np.random.uniform(0.3, 0.8)
                    + np.random.normal(0, 0.05),
                    0,
                    1
                )
            )

            for window in range(WINDOWS_PER_DAY):

                session_count = int(
                    np.clip(
                        np.random.poisson(3 + stress * 6),
                        0,
                        30
                    )
                )

                ibd = float(
                    np.clip(
                        session_count / 180
                        + np.random.normal(0, 0.005),
                        0,
                        1
                    )
                )

                gaps = np.random.exponential(
                    180 / max(session_count, 1),
                    max(session_count - 1, 1)
                )

                gaps = gaps / (gaps.sum() + 1e-9)

                isie = (
                    shannon_entropy(gaps)
                    if session_count > 1
                    else 0.0
                )

                category_probs = np.random.dirichlet(
                    [5, 1, 1, 1]
                    if stress > 0.6
                    else [2, 4, 2, 2]
                )

                mlue = shannon_entropy(category_probs)

                mood = float(
                    np.clip(
                        5
                        - stress * 3
                        + np.random.normal(0, 0.6),
                        1,
                        5
                    )
                )

                rows.append({
                    "student_id": sid,
                    "day": day,
                    "window_start_hour": window,

                    "session_count": session_count,

                    "screen_time_min": round(
                        float(
                            np.clip(
                                session_count
                                * np.random.uniform(2, 8),
                                0,
                                180
                            )
                        ),
                        2
                    ),

                    "unlock_freq": int(
                        np.clip(
                            session_count
                            + np.random.poisson(1),
                            0,
                            40
                        )
                    ),

                    "ibd": round(ibd, 5),
                    "isie": round(isie, 4),
                    "cii": round(cii, 4),
                    "mlue": round(mlue, 4),

                    "sleep_proxy_h": round(sleep, 2),

                    "activity_level": round(
                        float(
                            np.clip(
                                tmpl[window % 24]
                                + stress * np.random.uniform(-0.2, 0.4),
                                0,
                                1
                            )
                        ),
                        4
                    ),

                    "conv_count": int(
                        np.clip(
                            np.random.poisson(
                                max(0, (1 - stress) * 3)
                            ),
                            0,
                            15
                        )
                    ),

                    "coloc_diversity": int(
                        np.clip(
                            np.random.poisson(
                                max(0, (1 - stress) * 4)
                            ),
                            0,
                            20
                        )
                    ),

                    "ema_stress": round(
                        float(
                            np.clip(
                                1
                                + stress * 4
                                + np.random.normal(0, 0.3),
                                1,
                                5
                            )
                        ),
                        2
                    ),

                    "ema_social": round(
                        float(
                            np.clip(
                                5
                                - stress * 2
                                + np.random.normal(0, 0.5),
                                1,
                                5
                            )
                        ),
                        2
                    ),

                    "ema_mood": round(mood, 2),

                    "label": (
                        2 if mood >= 4
                        else 1 if mood >= 3
                        else 0
                    )
                })

    return pd.DataFrame(rows)