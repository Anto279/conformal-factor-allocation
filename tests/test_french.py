"""Lecture du format French sur un fichier fabriqué à la main."""

import io
import zipfile

import numpy as np

from src.data.french import has_missing_codes, parse_french_csv, read_french_zip

SAMPLE = """This file was created by CMPT_ME_BEME_OP_INV_RETS using the 202508 CRSP database.
The 1-month TBill rate data until 202405 are from Ibbotson Associates.

,Mkt-RF,SMB,HML,RMW,CMA,RF
196307,   -0.39,   -0.48,   -0.81,    0.64,   -1.15,    0.27
196308,    5.07,   -0.80,    1.70,    0.40,   -0.38,    0.25
196309,   -1.57,   -0.43,    0.00,   -0.78,    0.15,    0.27

 Annual Factors: January-December 
,Mkt-RF,SMB,HML,RMW,CMA,RF
  1964,   12.51,    0.46,    9.32,    2.09,    6.39,    3.54

Copyright 2025 Eugene F. Fama and Kenneth R. French
"""


def _zip(text: str, tmp_path):
    path = tmp_path / "sample.zip"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("sample.csv", text.replace("\n", "\r\n"))
    return path


def test_two_tables_monthly_then_annual():
    tables = parse_french_csv(SAMPLE)
    assert [t.frequency for t in tables] == ["M", "A"]
    m = tables[0].data
    assert list(m.columns) == ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]
    assert len(m) == 3
    assert str(m.index[0].date()) == "1963-07-31"
    assert m.loc["1963-08-31", "Mkt-RF"] == 5.07


def test_read_zip_keeps_only_requested_frequency(tmp_path):
    df = read_french_zip(_zip(SAMPLE, tmp_path), "M")
    assert len(df) == 3 and df.index.is_monotonic_increasing


def test_daily_dates_and_missing_codes():
    text = ",Mom   \n19261103,   0.56\n19261104,  -99.99\n"
    t = parse_french_csv(text)[0]
    assert t.frequency == "D" and list(t.data.columns) == ["Mom"]
    assert has_missing_codes(t.data)
    assert not has_missing_codes(t.data.iloc[:1])
    assert np.isclose(t.data.iloc[0, 0], 0.56)
