from pathlib import Path
import sys
import numpy as np
import pandas as pd

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / 'src'))
from run_closed_loop_simulation import summarize, METHODS

trials = pd.read_csv(root / 'results/closed_loop/test/episode_trials.csv')
expected = pd.read_csv(root / 'results/closed_loop/test/summary.csv').set_index('method')
assert len(trials) == 7000
assert trials.groupby('method').size().eq(1000).all()
assert set(trials.method) == set(METHODS)
assert not trials.duplicated(['method', 'episode']).any()
assert trials.groupby('episode').method.nunique().eq(7).all()
actual = summarize(trials).set_index('method').loc[expected.index, expected.columns]
np.testing.assert_allclose(actual.to_numpy(float), expected.to_numpy(float), rtol=1e-10, atol=1e-12)
print('PASS: 7,000 method-episode records; 1,000 paired scenarios; summary matches raw records.')
