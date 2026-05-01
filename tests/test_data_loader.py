import pandas as pd

from motoropt.data.loader import select_numeric_data


def test_select_numeric_data():
    df = pd.DataFrame(
        {
            "air_gap_mm": [0.8, "bad"],
            "magnet_thickness_mm": [5.0, 6.0],
            "slot_opening_mm": [2.0, 2.2],
            "torque_Nm": [15.0, 16.0],
        }
    )
    out = select_numeric_data(
        df,
        ["air_gap_mm", "magnet_thickness_mm", "slot_opening_mm"],
        ["torque_Nm"],
    )
    assert len(out) == 1
