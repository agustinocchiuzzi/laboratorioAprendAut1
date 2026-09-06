from __future__ import annotations

import pandas as pd

from aa_futbol.preprocessing import MixedTypeDiscretizer


def test_unknown_category_uses_reserved_zero() -> None:
    train = pd.DataFrame({"team": ["A", "B", "A"], "form": [0.0, 1.0, 2.0]})
    transformer = MixedTypeDiscretizer(["team"], ["form"], n_bins=2).fit(train)

    transformed = transformer.transform(pd.DataFrame({"team": ["new"], "form": [1.5]}))

    assert transformed.shape == (1, 2)
    assert transformed[0, 0] == 0
    assert transformed[0, 1] >= 1
