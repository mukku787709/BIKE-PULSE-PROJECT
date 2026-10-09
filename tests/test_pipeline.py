import numpy as np
import pandas as pd
import pytest
from bikepulse.data import FEATURES, wrangle
from bikepulse.insights import answer

def sample():
    return pd.DataFrame({"dteday":["2011-01-01"]*3,"cnt":[3,3,-1],**{k:[1,1,1] for k in FEATURES[:12]}})

def test_wrangling_rejects_invalid_targets_and_duplicates():
    clean=wrangle(sample())
    assert len(clean)==1
    assert set(FEATURES)<=set(clean.columns)
    assert np.isclose(clean.hour_sin.iloc[0]**2+clean.hour_cos.iloc[0]**2,1)
    assert not {"casual","registered","cnt"}&set(FEATURES)

def test_schema_validation():
    with pytest.raises(ValueError,match="Missing columns"):
        wrangle(pd.DataFrame({"cnt":[1]}))

def test_assistant_uses_actual_metrics():
    report={"winner":"Ridge","test":{"Ridge":{"MAE":12.34}}}
    assert "12.34" in answer("best model",report,None)
    assert "not a generative" in answer("hello",report,None)
