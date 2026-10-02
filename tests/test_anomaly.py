import numpy as np

from atler_ml.anomaly import _past_z


def test_past_z_only_uses_the_past():
    x = np.log(np.array([100, 110, 95, 105, 100, 1000, 100.0]))
    z = _past_z(x)
    assert np.isnan(z[:5]).all()
    assert z[5] > 3.5          # the spike
    assert z[6] < 1            # judged against history that includes the spike, still normal
