# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 07:52:57 2023

@author: byers
"""

import dask.dataframe as dd
from dask.diagnostics import ProgressBar
import matplotlib.pyplot as plt
import os
import pandas
import pyam

wd = "C:\\Users\\byers\\IIASA\\ECE.prog - Documents\\Research Theme - NEXUS\\Hotspots_Explorer_2p0\\rcre_testing\\output"
meta_folder = "C:\\Users\\byers\\IIASA\\IPCC WG3 Chapter 3 - Documents\\IPCC_AR6DB\\snapshots\\snapshot_ar6_public_v1.1\\uploaded\\"
meta_file = "AR6_Scenarios_Database_metadata_indicators_v1.1.xlsx"


os.chdir(wd)
filename = "rcre_output_small5yr_output.csv"
filename = "AR6full_rcre_output_COUNTRIES_tr20_5yrpara.csv"
# %% Load data

df = dd.read_csv(filename)


# %% Downselect variables
varis_all = list(df.variable.unique())
varis_selected = [x for x in varis_all if "Exposure" in x]
varis_selected = [
    str for str in varis_selected if any(sub in str for sub in ["High", "Low"]) == False
]
# %% Downselect indicator and convert to pyam

indis = ["pr_r10", "iavar"]
indis = ["tr20"]
ind = indis[0]
varis_i = [str for str in varis_selected if any(sub in str for sub in indis)]
dfi = df.loc[df.variable.isin(varis_i)]
dfi = dfi.compute()  # .drop(columns='Unnamed: 0')
dfp = pyam.IamDataFrame(dfi)
dfp.load_meta(f"{meta_folder}{meta_file}", sheet_name="meta_Ch3vetted_withclimate")


# %% Plot one country, two categories

indi = f"RCRE|{ind}|Exposure|Land area"

iso = "PAK"
dfpp = dfp.filter(variable=indi, Category=["C1", "C6"], region=iso)


fig = plt.figure(figsize=(4, 3))
ax = fig.add_subplot(1, 1, 1)


dfpp.plot(
    color="Category",
    alpha=0.3,
    fill_between=True,
    final_ranges=False,
    title=f"AR6DB - {iso} - {ind}|Exposure|Land area",
    ax=ax,
    order=["year"],
)


# %% plot - multiple countries, one scenario


indi = f"RCRE|{ind}|Exposure|Population|%"

iso = ["TUR", "AGO", "ESP", "GBR", "AUT"]
dfpp = dfp.filter(
    variable=indi,
    # model='AIM/CGE 2.1', scenario='CO_Bridge',
    Category=[
        "C3",
    ],
    region=iso,
)


fig = plt.figure(figsize=(4, 3))
ax1 = fig.add_subplot(1, 1, 1)


dfpp.plot(
    color="region",
    alpha=0.3,
    fill_between=True,
    final_ranges=False,
    title=f"AR6DB - {iso} - {ind}|Exposure|Population|%",
    ax=ax1,
    order=["year"],
    legend=True,
)


# %% load ar6db

world_filename = "AR6_Scenarios_Database_World_v1.1.csv"

dfar6 = pyam.IamDataFrame(meta_folder + world_filename)
dfar6.load_meta(meta_folder + meta_file)

# %% GMT plots for presentation


dfe = dfar6.filter(
    variable="AR6 climate diagnostics|Surface Temperature (GSAT)|MAGICCv7.5.3|50.0th Percentile",
    Category=["C1", "C3", "C8"],
)
# %%

fig = plt.figure(figsize=(4, 3))
ax = fig.add_subplot(1, 1, 1)

dfe.plot(
    color="Category",
    legend=False,
    fill_between=True,
    final_ranges=False,
    title="AR6DB - GSAT p50 - MAGICC",
    ax=ax,
    alpha=0.4,
    order=["year"],
)

ax.set_xlim([2010, 2105])
ax.hlines(1.1, 2000, 2100, color="k", linestyle="--", alpha=0.5)

fig.savefig()


# %% by emissions
dfe = dfar6.filter(
    variable="AR6 climate diagnostics|Infilled|Emissions|CO2",
    Category=["C1", "C3", "C8"],
)
dfe.convert_unit("Mt CO2/yr", "Gt CO2/yr", inplace=True)


#
fig = plt.figure(figsize=(4, 4))
ax = fig.add_subplot(1, 1, 1)

dfe.plot(
    color="Category",
    legend=False,
    fill_between=True,
    final_ranges=True,
    title="AR6DB - CO2 emissions",
    ax=ax,
)

ax.set_xlim([2010, 2105])
ax.hlines(0, 2000, 2100, color="k")


# %%
