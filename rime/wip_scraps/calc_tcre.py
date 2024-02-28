# -*- coding: utf-8 -*-
"""
This script determines the linear relationship between cumulative CO2 emissions
and global mean surface air temperature.
"""

# %% Simple function based on peak warming and cumulative CO2 to year of net-zero


# %%


if __name__ == "__main__":
    import pyam
    from scipy.stats import linregress
    import matplotlib.pyplot as plt
    import numpy as np
    from rime_functions import co2togwl_simple

    fd = "C:\\Users\\byers\\IIASA\\IPCC WG3 Chapter 3 - Documents\\IPCC_AR6DB\\snapshots\\snapshot_ar6_public_v1.1\\uploaded\\"
    world_filename = "AR6_Scenarios_Database_World_v1.1.csv"

    dfar6 = pyam.IamDataFrame(fd + world_filename)
    dfar6.load_meta(
        fd + "AR6_Scenarios_Database_metadata_indicators_v1.1.xlsx",
        sheet_name="meta_Ch3vetted_withclimate",
    )

    df = dfar6.filter(Category="C*")
    varins = [
        "Emissions|CO2",
        "AR6 climate diagnostics|Surface Temperature (GSAT)|MAGICCv7.5.3|50.0th Percentile",
        "AR6 climate diagnostics|Infilled|Emissions|CO2",
    ]
    df = df.filter(variable=varins)
    # %%
    start_year = 2015
    df = df.filter(year=range(2015, 2101))
    df.interpolate(time=range(start_year, 2101), inplace=True)

    # %%
    # =============================================================================
    # Simple regression based on
    # cumulative CO2 to year of net-zero VS median peak warming
    # =============================================================================

    meta_pw50 = "Median peak warming (MAGICCv7.5.3)"
    meta_cumNZ = "Cumulative net CO2 (2020 to netzero, Gt CO2) (Harm-Infilled)"

    cols = ["Category", meta_pw50, meta_cumNZ]
    dfm = df.meta[cols]

    # % do regression
    slope, intercept, r, p, se = linregress(dfm[meta_cumNZ], dfm[meta_pw50])

    dfm.plot.scatter(meta_cumNZ, meta_pw50, c="Category")
    x = range(0, 8000)
    plt.plot(x, intercept + slope * x, "r", label="fitted line")

    # %% Examples
    co2togwl_simple(np.array([500, 600, 800]), regr={"slope": 0.0004, "intercept": 1.4})

    co2togwl_simple(x, dfm[[meta_cumNZ, meta_pw50]])
