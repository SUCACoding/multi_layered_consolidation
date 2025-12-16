# -*- coding: utf-8 -*-
"""
Created on Fri Dec 12 12:23:12 2025

@author: SUCA
"""
import streamlit as st
import numpy as np
from pore_pressure import compute_explicit
import matplotlib.pyplot as plt
import pandas as pd
import pdf_report

st.set_page_config(layout="wide")

# Helper function for settlement calculation
def set_calc(UPore_Calc, strat):
    d_eps = np.zeros_like(UPore_Calc)
    d_eps[:, 1:] = (UPore_Calc[:, 0:-1] - UPore_Calc[:, 1:]) / strat['eoed'][:, np.newaxis]
    eps = np.cumsum(d_eps, axis=1)
    d_sett = strat['dZ'][:, np.newaxis] * eps * 1000  # mm
    return np.cumsum(d_sett[::-1, :], axis=0)[::-1, :]

st.title('Results')
st.write("""
    This page displays the results of the consolidation calculations, including multiple pore pressure and 
    settlement plots generated based on chosen input parameters.
""")

if 'visited_pages' not in st.session_state:
    st.session_state.visited_pages = set()

if 2 not in st.session_state.visited_pages:
    st.warning("You must complete Loads before accessing Results.")
    st.stop()
    

#import streamlit as st

default_values = [0, 25, 50, 60, 70, 80, 90]

if "num_points" not in st.session_state:
    st.session_state.num_points = len(default_values)
if "input_values" not in st.session_state:
    st.session_state.input_values = default_values.copy()

def resize_input_values():
    np_val = st.session_state.num_points_input
    if np_val > len(st.session_state.input_values):
        st.session_state.input_values.extend(
            [st.session_state.input_values[-1]] * (np_val - len(st.session_state.input_values))
        )
    else:
        st.session_state.input_values = st.session_state.input_values[:np_val]
    st.session_state.num_points = np_val



col7, col8, col9 = st.columns([1, 1, 2])

with col7:
    st.write("Select if the boundary at the bottom of the soil column is Open (Free flow), or Closed (No flow)")
    
    st.session_state.boundary = st.segmented_control(
        label="Boundary",
        options=["Open", "Closed"],
        default="Closed"
    )
    
    #st.session_state.boundary = boundary
with col8:
    st.write("The number of Consolidation points can be selected, which is used in the table and figure results.")
    
    num_points_input = st.number_input(
        "Number of Consolidation Points",
        min_value=1, max_value=20,
        value=st.session_state.num_points,
        step=1,
        key="num_points_input",
        on_change=resize_input_values,
    )
    


with col9:
    num_columns = 4
    cols = st.columns(num_columns)

    for i in range(st.session_state.num_points):
        with cols[i % num_columns]:
            val = st.number_input(
                f"Cond Deg {i + 1}",
                min_value=0, max_value=90, step=5,
                value=st.session_state.input_values[i],
                key=f"num_input_{i}"
            )
            st.session_state.input_values[i] = val

# Run calculation and store results
if st.button("Run Calculation"):
    #remove duplicates
    unique_lst = list(dict.fromkeys(st.session_state.input_values))
    set_frac = [x / 100 for x in unique_lst]
    
    strat = st.session_state.strat
    gamW = 10

    UPore_Calc = np.zeros((len(strat['depth']), 1000))
    UPore_Calc[:, 0] = strat['load']

    dset_tot = strat['load'] / strat['eoed'] * strat['dZ'] * 1000  # mm
    set_tot = np.cumsum(dset_tot[::-1])[::-1]

    pow_n = 1.5
    i = 1
    dT = 0.5
    set_u_calc = 0.1
    t_step_sum = np.zeros(1000)

    while (set_u_calc / set_tot[0]) < 0.90:
        t_step = (i * dT) ** pow_n
        UPore_Calc[:, i] = compute_explicit(
            UPore_Calc[:, i - 1], strat['k_perm'], strat['eoed'],
            strat['void'], gamW, t_step * 24 * 3600, strat['dZ'], st.session_state.boundary
        )
        dset_calc = (strat['load'] - UPore_Calc[:, i]) / strat['eoed'] * strat['dZ'] * 1000
        set_temp_calc = np.cumsum(dset_calc[::-1])[::-1]
        set_u_calc = set_temp_calc[0]
        t_step_sum[i] = t_step + t_step_sum[i - 1]
        i += 1

    t_step_sum = t_step_sum[0:i - 1]
    UPore_Calc = UPore_Calc[:, 0:i - 1]

    d_eps = np.zeros_like(UPore_Calc)
    d_eps[:, 1:] = (UPore_Calc[:, 0:-1] - UPore_Calc[:, 1:]) / strat['eoed'][:, np.newaxis]
    eps = np.cumsum(d_eps, axis=1)
    d_sett = strat['dZ'][:, np.newaxis] * eps * 1000
    sett = np.cumsum(d_sett[::-1, :], axis=0)[::-1, :]

    #set_frac = [0, 0.25, 0.50, 0.60, 0.70, 0.80, 0.90]
    idx_frac = np.zeros(len(set_frac), dtype=int)
    for i_, val in enumerate(set_frac):
        temp = np.abs(sett[0, :] / set_tot[0] - val)
        idx_frac[i_] = temp.argmin()
    
    # Display settlement summary table
    data = {
        'Settlement [mm]': np.ceil(sett[0, idx_frac]),
        'Time [days]': np.ceil(t_step_sum[idx_frac])
    }
    df = pd.DataFrame(data)
    
    # Transpose, then use columns as the new index (row names)
    df_t = df.transpose()
    
    # Optional: rename the index for clarity
    df_t.index.name = 'Degree of Consolidation'
    
    # The columns are now labeled by the original dataframe row numbers (e.g. 0, 1, 2...)
    # If you want to label columns using percentage labels:
    col_names = [f"{int(x*100)}%" for x in set_frac]
    df_t.columns = col_names
    st.dataframe(df_t)
    cons_deg = (1 - UPore_Calc / strat['load'][:, np.newaxis]).round(1)

    idx_cons = np.zeros([len(strat['depth']), len(set_frac)], dtype=int)
    for i_, val in enumerate(set_frac):
        test = (cons_deg >= val)
        has_value = test.any(axis=1)
        temp = np.argmax(test, axis=1)
        temp[~has_value] = len(t_step_sum) - 1
        idx_cons[:, i_] = temp

    figs_and_titles = []

    col1, col2 = st.columns(2)
    with col1:
        st.write("**Pore Pressure vs Depth:** Distribution of pore pressure across the soil depth at various consolidation degrees.")
        fig, ax = plt.subplots()
        for leg_str, idx in zip(set_frac, idx_frac):
            ax.plot(UPore_Calc[:, idx], -strat['depth'], label=f"{leg_str * 100:.0f}%")
        ax.set_title('Pore Pressure vs Depth')
        ax.set_xlabel('Pore Pressure [kPa]')
        ax.set_ylabel('Depth [m]')
        ax.legend(title='Consolidation Degree')
        figs_and_titles.append(("Pore Pressure vs Depth", fig))
        st.pyplot(fig)

    with col2:
        st.write("**Consolidation Degree vs Depth:** Consolidation degree profiles.")
        fig2, ax = plt.subplots()
        for leg_str, idx in zip(set_frac, idx_frac):
            ax.plot(1 - UPore_Calc[:, idx] / strat['load'], -strat['depth'], label=f"{leg_str * 100:.0f}%")
        ax.set_title('Consolidation Degree')
        ax.invert_xaxis()
        ax.set_xlabel('Consolidation Degree [%]')
        ax.set_ylabel('Depth [m]')
        ax.legend(title='Consolidation Degree')
        figs_and_titles.append(("Consolidation Degree vs Depth", fig2))
        st.pyplot(fig2)

    col3, col4 = st.columns(2)

    with col3:
        st.write("**Settlement over Depth:** Settlement variation with depth at different settlement fractions.")
        fig3, ax = plt.subplots()
        for leg_str, idx in zip(set_frac, idx_frac):
            ax.plot(-sett[:, idx], -strat['depth'], label=f"{leg_str * 100:.0f}%")
        ax.set_title('Settlement over Depth')
        ax.invert_xaxis()
        ax.set_xlabel('Settlement [mm]')
        ax.set_ylabel('Depth [m]')
        ax.legend(title='Consolidation Degree')
        figs_and_titles.append(("Settlement over Depth", fig3))
        st.pyplot(fig3)

    with col4:
        st.write("**Time until Consolidation Degree vs Depth:** Time required at various depths.")
        fig4, ax = plt.subplots()
        for i_, leg_str in enumerate(set_frac):
            ax.plot(t_step_sum[idx_cons[:, i_]], -strat['depth'], label=f"{leg_str * 100:.0f}%")
        ax.set_title('Time until Consolidation Degree')
        ax.set_xlabel('Time [Days]')
        ax.set_ylabel('Depth [m]')
        ax.legend(title='Consolidation Degree')
        figs_and_titles.append(("Time until Consolidation Degree vs Depth", fig4))
        st.pyplot(fig4)

    col5, col6 = st.columns(2)

    with col5:
        st.write("**Settlement vs Logarithmic Time:** Settlement over log time scale.")
        fig5, ax = plt.subplots()
        t_plot = t_step_sum.copy()
        t_plot[t_plot <= 0] = 1e-10
        ax.plot(t_plot, -sett[0, :])
        ax.set_title('Settlement vs Log Time')
        ax.set_xlabel('Time [Days]')
        ax.set_ylabel('Settlement [mm]')
        ax.set_xscale('log')
        figs_and_titles.append(("Settlement vs Log Time", fig5))
        st.pyplot(fig5)

    with col6:
        st.write("**Consolidation Degree vs Logarithmic Time:** Consolidation Degree over log time scale.")
        fig6, ax = plt.subplots()
        ax.plot(t_plot, 1 - sum(UPore_Calc) / sum(strat['load']))
        ax.set_title('Consolidation Degree vs Log Time')
        ax.set_xlabel('Time [Days]')
        ax.set_ylabel('U [-]')
        ax.set_xscale('log')
        figs_and_titles.append(("Consolidation Degree vs Log Time", fig6))
        st.pyplot(fig6)

    # Save results in session state to share with PDF generator
    st.session_state['results'] = {
        'strat': strat,
        'set_frac': set_frac,
        'idx_frac': idx_frac,
        'UPore_Calc': UPore_Calc,
        'sett': sett,
        't_step_sum': t_step_sum,
        'idx_cons': idx_cons,
        'figs_and_titles': figs_and_titles,
        'soil_layers' : st.session_state.edited_df,
        'cons_deg' : df_t,
        'boundary' : st.session_state.boundary,
        'dZ' : strat['dZ'][0].round(1)
    }

# Show PDF button only if results exist
if 'results' in st.session_state:
    st.write("---")
    if st.button("Generate PDF Report"):
        r = st.session_state['results']
        pdf_bytes = pdf_report.create_pdf_report(
            r['strat'], r['set_frac'], r['idx_frac'], r['UPore_Calc'],
            r['sett'], r['t_step_sum'], r['idx_cons'], r['figs_and_titles'],
            r['soil_layers'], r['cons_deg'], r['boundary'], r['dZ']
        )
        st.download_button(
            label="Download PDF Report",
            data=pdf_bytes,
            file_name="consolidation_report.pdf",
            mime="application/pdf"
        )
else:
    st.info("Run calculation first to generate PDF report.")