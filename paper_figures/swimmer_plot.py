#!/usr/bin/env python3
"""
Generate swimmer plot showing longitudinal patient timelines.

Shows hospital record spans, narcolepsy diagnosis events, and deaths
for all narcolepsy patients in the cohort, aligned by calendar year.

Usage:
    python swimmer_plot.py
"""

import os
import time
import random
import numpy as np
import polars as pl
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection, LineCollection
from matplotlib.patches import Rectangle
from datetime import datetime

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, '..', 'data')
MANUSCRIPT_FIG_DIR = os.path.join(BASE, '..', 'manuscript', 'figures')


def create_swimmer_plot(df, figsize=(15, 10), color_map=None,
                        use_calendar_years=True, show_legend=True):
    """
    Create a swimmer plot of patient timelines.

    Parameters
    ----------
    df : polars.DataFrame
        Must contain columns: id, state, start_date, end_date
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=150)

    required_cols = ['id', 'state', 'start_date', 'end_date']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"DataFrame must contain: {required_cols}")

    event_order = sorted(df['state'].unique().to_list())

    # Default colors
    if color_map is None:
        color_map = {}
    state_colors = {}
    for state in event_order:
        if state in color_map:
            state_colors[state] = color_map[state]
        else:
            state_colors[state] = "#" + ''.join(
                [random.choice('0123456789ABCDEF') for _ in range(6)])

    # Compute per-patient stats
    grouped = df.group_by('id').agg([
        pl.col('start_date').min().alias('min_date'),
        pl.col('start_date').max().alias('max_date'),
    ]).sort('min_date')

    patient_ids = grouped['id'].to_list()
    patient_to_idx = {pid: idx for idx, pid in enumerate(patient_ids)}
    ref_date = grouped['min_date'].min()

    min_days = (grouped['min_date'].min() - ref_date).days
    max_days = (grouped['max_date'].max() - ref_date).days

    # Group data by state
    state_data = {}
    for state in event_order:
        sd = df.filter(pl.col('state') == state)
        if len(sd) > 0:
            state_data[state] = sd

    # Build polygons and lines
    for state in event_order:
        if state not in state_data:
            continue
        bars = []
        lines = []
        for row in state_data[state].iter_rows(named=True):
            pid = row['id']
            if pid not in patient_to_idx:
                continue
            y_pos = patient_to_idx[pid]
            start_x = (row['start_date'] - ref_date).days
            end_x = (row['end_date'] - ref_date).days if row['end_date'] else max_days

            if start_x == end_x:
                lines.append([(start_x, y_pos), (start_x, y_pos + 1)])
            else:
                bars.append([
                    (start_x, y_pos), (end_x, y_pos),
                    (end_x, y_pos + 1), (start_x, y_pos + 1)
                ])

        color = state_colors[state]
        if bars:
            ax.add_collection(PolyCollection(
                bars, facecolors=color, edgecolors='none'))
        if lines:
            ax.add_collection(LineCollection(
                lines, colors=color, linewidths=1))

    # Axes
    ax.set_ylim(0, len(patient_ids))
    ax.set_yticks([])

    if use_calendar_years:
        min_date = grouped['min_date'].min()
        max_date = grouped['max_date'].max()
        years = range(min_date.year, max_date.year + 2)
        tick_dates = [datetime(year, 1, 1) for year in years]
        x_ticks = [(d - ref_date).days for d in tick_dates]
        x_labels = [str(y) for y in years]
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels, rotation=90)
        ax.set_xlim(min_days, max_days)
        ax.set_xlabel('Calendar Year')
    else:
        years_total = max_days // 365.25
        x_ticks = np.arange(0, (years_total + 1) * 365.25, 365.25)
        x_labels = [int(t // 365.25) for t in x_ticks]
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels)
        ax.set_xlim(min_days, max_days)
        ax.set_xlabel('Years Since First Event')

    n_patients = df['id'].n_unique()
    ax.set_title(f'Narcolepsy Patient Timeline (n = {n_patients})',
                 fontsize=20, fontweight='bold')
    ax.grid(True, axis='x', color='white', linestyle='-', linewidth=0.5, zorder=3)
    ax.set_axisbelow(False)

    if show_legend:
        handles = []
        labels = []
        for state in event_order:
            if state in state_data:
                handles.append(Rectangle((0, 0), 1, 1,
                               facecolor=state_colors[state], edgecolor='none'))
                labels.append(state.replace('_', ' ').title())
        ax.legend(handles, labels, bbox_to_anchor=(1.05, 1),
                  loc='upper left', frameon=True, fancybox=True, shadow=True)

    plt.tight_layout()
    return fig, ax


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    print("Loading swimmer data...")
    swimmer_df = pl.read_parquet(
        os.path.join(DATA_DIR, 'discriminative-modeling', 'bdsp_narco_swimmer.parquet'))

    # Filter to post-1990 for cleaner visualization
    swimmer_df = swimmer_df.filter(pl.col('start_date').dt.year() > 1990)
    print(f"  {swimmer_df['id'].n_unique()} patients, {len(swimmer_df)} events")

    fig, ax = create_swimmer_plot(
        swimmer_df,
        figsize=(15, 10),
        color_map={
            'narcolepsy': '#ff7f0e',
            'hospital_record': '#F5F5F5',
            'death': '#C5C5C5',
        },
        use_calendar_years=True,
    )

    os.makedirs(MANUSCRIPT_FIG_DIR, exist_ok=True)
    out_path = os.path.join(MANUSCRIPT_FIG_DIR, 'efigure1_swimmer_plot.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {out_path}")
