import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import pdf_report

from pore_pressure import compute_explicit

st.set_page_config(layout="wide")

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

def calculate_consolidation_results(strat, load_vector, cons_degrees_percent):
    set_frac = np.array(cons_degrees_percent) / 100.0
    n_depth = len(strat['depth'])
    UPore_Calc = np.zeros((n_depth, 1000))
    UPore_Calc[:, 0] = load_vector
    dset_tot = load_vector / strat['eoed'] * strat['dZ'] * 1000
    set_tot = np.cumsum(dset_tot[::-1])[::-1]
    pow_n, dT, i, set_u_calc = 1.5, 0.5, 1, 0.1
    t_step_sum = np.zeros(1000)

    while (set_u_calc / set_tot[0]) < 0.90 and i < 999:
        t_step = (i * dT)**pow_n
        UPore_Calc[:, i] = compute_explicit(
            UPore_Calc[:, i - 1], strat['k_perm'], strat['eoed'], strat['void'],
            10, t_step * 24 * 3600, strat['dZ'], st.session_state.boundary
        )
        dset_calc = (load_vector - UPore_Calc[:, i]) / strat['eoed'] * strat['dZ'] * 1000
        set_temp_calc = np.cumsum(dset_calc[::-1])[::-1]
        set_u_calc = set_temp_calc[0]
        t_step_sum[i] = t_step + t_step_sum[i - 1]
        i += 1

    UPore_Calc = UPore_Calc[:, :i]
    t_step_sum = t_step_sum[:i]

    d_eps = np.zeros_like(UPore_Calc)
    d_eps[:, 1:] = (UPore_Calc[:, :-1] - UPore_Calc[:, 1:]) / strat['eoed'][:, None]
    eps = np.cumsum(d_eps, axis=1)
    d_sett = strat['dZ'][:, None] * eps * 1000
    sett = np.cumsum(d_sett[::-1, :], axis=0)[::-1, :]

    idx_frac = [np.abs(sett[0, :] / set_tot[0] - val).argmin() for val in set_frac]
    df_summary = pd.DataFrame({
        'Settlement [mm]': np.ceil(sett[0, idx_frac]),
        'Time [days]': np.ceil(t_step_sum[idx_frac])
    }, index=[f"{int(v*100)}%" for v in set_frac])
    df_summary.index.name = "Degree of Consolidation"

    cons_deg = (1 - UPore_Calc / load_vector[:, None]).round(2)
    idx_cons = np.zeros((n_depth, len(set_frac)), dtype=int)
    for j, val in enumerate(set_frac):
        test = cons_deg >= val
        has_val = test.any(axis=1)
        argmax = np.argmax(test, axis=1)
        argmax[~has_val] = len(t_step_sum) - 1
        idx_cons[:, j] = argmax

    col_labels = df_summary.index.tolist()
    df_pore_pressure = pd.DataFrame(UPore_Calc[:, idx_frac], index=strat['depth'], columns=col_labels)
    df_cons_degree = pd.DataFrame(cons_deg[:, idx_frac], index=strat['depth'], columns=col_labels)
    df_settlement = pd.DataFrame(sett[:, idx_frac], index=strat['depth'], columns=col_labels)
    df_time_depth = pd.DataFrame({lbl: t_step_sum[idx_cons[:, i]] for i, lbl in enumerate(col_labels)}, index=strat['depth'])

    t_plot = np.maximum(t_step_sum, 1e-10)

    df_log_settlement = pd.DataFrame({'Time [days]': t_plot[1:], 'Settlement [mm]': -sett[0, 1:]})
    avg_cons_deg = 1 - np.sum(UPore_Calc, axis=0) / np.sum(load_vector)
    avg_cons_deg = np.clip(avg_cons_deg, 1e-10, None)  # similarly clip
    df_log_cons_degree = pd.DataFrame({'Time [days]': t_plot[1:], 'Consolidation Degree [-]': avg_cons_deg[1:]})

    df_log_settlement = df_log_settlement.set_index('Time [days]')
    df_log_cons_degree = df_log_cons_degree.set_index('Time [days]')

    return {
        'strat': strat,
        'UPore_Calc': UPore_Calc,
        'sett': sett,
        't_step_sum': t_step_sum,
        'idx_frac': idx_frac,
        'idx_cons': idx_cons,
        'df_summary': df_summary,
        'df_pore_pressure': df_pore_pressure,
        'df_cons_degree': df_cons_degree,
        'df_settlement': df_settlement,
        'df_time_depth': df_time_depth,
        'df_log_settlement': df_log_settlement,
        'df_log_cons_degree': df_log_cons_degree,
        'set_frac': set_frac
    }

def plot_from_df(df, x_label, y_label, title, invert_x=False, log_x=False, invert_y=False):
    fig, ax = plt.subplots()

    # Convert Series to DataFrame for consistent handling
    if isinstance(df, pd.Series):
        df = df.to_frame()

    # Use DataFrame index as x-axis values
    x_vals = df.index.to_numpy()

    # Fix for zeros or negatives on log scale
    if log_x:
        # Replace zeros or negative x_vals with small positive number
        x_vals = np.where(x_vals <= 0, 1e-10, x_vals)

    for col in df.columns:
        y_vals = df[col].to_numpy()
        if invert_y:
            y_vals = -y_vals
            ax.invert_yaxis()
        ax.plot(x_vals, y_vals, label=str(col))

    ax.set_xlabel(x_label)
    ax.set_ylabel('Depth [m]' if invert_y else y_label)
    ax.set_title(title)

    if invert_x:
        ax.invert_xaxis()
    if log_x:
        ax.set_xscale('log')
        ax.set_xlim(left=x_vals[x_vals > 0].min(), right=x_vals.max())

    if invert_y:
        ax.invert_yaxis()

    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.legend()
    return fig

# UI: Input controls
default_values = [0, 25, 50, 60, 70, 80, 90]
st.session_state.setdefault('input_values', default_values.copy())
st.session_state.setdefault('num_points', len(default_values))
col7, col8, col9 = st.columns([1, 1, 2])
with col7:
    st.write("Select if the boundary at the bottom of the soil column is Open (Free flow), or Closed (No flow)")
    st.session_state.boundary = st.segmented_control("Boundary", ["Open", "Closed"], default="Closed")
with col8:
    st.write("The number of Consolidation points can be selected, which is used in the table and figure results.")
    num_points = st.number_input("Number of Consolidation Points", 1, 20, value=st.session_state.get('num_points', 7), step=1, key="num_points_input")
    input_values = st.session_state.get('input_values', [0, 25, 50, 60, 70, 80, 90])
    if num_points != len(input_values):
        if num_points > len(input_values):
            input_values.extend([input_values[-1]] * (num_points - len(input_values)))
        else:
            input_values = input_values[:num_points]
        st.session_state['input_values'] = input_values
    st.session_state['num_points'] = num_points
with col9:
    cols = st.columns(4)
    for i in range(num_points):
        with cols[i % 4]:
            val = st.number_input(f"Cond Deg {i+1}", 0, 90, step=5, value=st.session_state.input_values[i], key=f"num_input_{i}")
            st.session_state.input_values[i] = val

if st.button("Run Calculation"):
    strat = st.session_state.strat
    results = calculate_consolidation_results(strat, strat['load'], st.session_state.input_values)
    st.session_state['results'] = results

if 'results' in st.session_state:
    r = st.session_state['results']
    st.write("### Settlement Summary")
    st.dataframe(r['df_summary'])

    plot_configs = [
        ("Pore Pressure vs Depth", r['df_pore_pressure'], 'Pore Pressure [kPa]', 'Depth [m]', False, False, True),
        ("Consolidation Degree vs Depth", r['df_cons_degree'], 'Consolidation Degree [-]', 'Depth [m]', True, False, True),
        ("Settlement over Depth", r['df_settlement'], 'Settlement [mm]', 'Depth [m]', True, False, True),
        ("Time until Consolidation Degree", r['df_time_depth'], 'Time [Days]', 'Depth [m]', False, False, True),
        ("Settlement vs Log Time", r['df_log_settlement'], 'Time [days]', 'Settlement [mm]', False, True, False),
        ("Consolidation Degree vs Log Time", r['df_log_cons_degree'], 'Time [days]', 'U [-]', False, True, False),
    ]

    col1, col2 = st.columns(2)
    for i, (title, df, xlabel, ylabel, inv_x, log_x, inv_y) in enumerate(plot_configs):
        with (col1 if i % 2 == 0 else col2):
            st.write(f"**{title}:**")
            fig = plot_from_df(df, xlabel, ylabel, title, inv_x, log_x, inv_y)
            st.pyplot(fig)

    col_pdf, col_excel = st.columns(2)
    with col_pdf:
        if st.button("Generate PDF Report"):
            figs_for_pdf = [(title, plot_from_df(df, xlabel, ylabel, title, inv_x, log_x, inv_y))
                            for title, df, xlabel, ylabel, inv_x, log_x, inv_y in plot_configs]
            pdf_bytes = pdf_report.create_pdf_report(
                r['strat'], r['df_summary'], r['idx_frac'], r['UPore_Calc'],
                r['sett'], r['t_step_sum'], r['idx_cons'], figs_for_pdf,
                st.session_state.get('edited_df'), r['df_summary'], st.session_state.boundary, r['strat']['dZ'][0].round(1)
            )
            st.download_button("Download PDF Report", pdf_bytes, "consolidation_report.pdf", "application/pdf")

    with col_excel:
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            r['df_pore_pressure'].to_excel(writer, sheet_name='Pore Pressure')
            r['df_cons_degree'].to_excel(writer, sheet_name='Consolidation Degree')
            r['df_settlement'].to_excel(writer, sheet_name='Settlement')
            r['df_time_depth'].to_excel(writer, sheet_name='Time to Consolidation')
            r['df_log_settlement'].to_excel(writer, sheet_name='Settlement vs Log Time', index=True)
            r['df_log_cons_degree'].to_excel(writer, sheet_name='Consolidation vs Log Time', index=True)

            workbook = writer.book
            chart_sheet = workbook.add_worksheet('Charts')

            def add_scatter(sheetname, df, x_label, y_label, title, row_pos, reverse_y=False):
                n_rows, n_cols = df.shape
                chart = workbook.add_chart({'type': 'scatter', 'subtype': 'straight'})
                for col_num in range(1, n_cols + 1):
                    chart.add_series({
                        'name': [sheetname, 0, col_num],
                        'categories': [sheetname, 1, 0, n_rows, 0],
                        'values': [sheetname, 1, col_num, n_rows, col_num],
                    })
                chart.set_x_axis({'name': x_label})
                y_opts = {'name': y_label}
                if reverse_y:
                    y_opts['reverse'] = True
                chart.set_y_axis(y_opts)
                chart.set_title({'name': title})
                chart_sheet.insert_chart(row_pos, 1, chart, {'x_scale': 1.5, 'y_scale': 1.5})

            row_pos = 1
            add_scatter('Pore Pressure', r['df_pore_pressure'], 'Pore Pressure [kPa]', 'Depth [m]', 'Pore Pressure vs Depth', row_pos, reverse_y=True)
            row_pos += 20
            add_scatter('Consolidation Degree', r['df_cons_degree'], 'Consolidation Degree [-]', 'Depth [m]', 'Consolidation Degree vs Depth', row_pos, reverse_y=True)
            row_pos += 20
            add_scatter('Settlement', r['df_settlement'], 'Settlement [mm]', 'Depth [m]', 'Settlement over Depth', row_pos, reverse_y=True)
            row_pos += 20

            n_rows, n_cols = r['df_time_depth'].shape
            time_chart = workbook.add_chart({'type': 'scatter', 'subtype': 'straight'})
            for col_num in range(1, n_cols + 1):
                time_chart.add_series({
                    'name': ['Time to Consolidation', 0, col_num],
                    'categories': ['Time to Consolidation', 1, 0, n_rows, 0],
                    'values': ['Time to Consolidation', 1, col_num, n_rows, col_num],
                })
            time_chart.set_x_axis({'name': 'Time [days]'})
            time_chart.set_y_axis({'name': 'Depth [m]', 'reverse': True})
            time_chart.set_title({'name': 'Time until Consolidation Degree vs Depth'})
            chart_sheet.insert_chart(row_pos, 1, time_chart, {'x_scale': 1.5, 'y_scale': 1.5})

            row_pos += 20

            log_sett = r['df_log_settlement']
            log_sett_index = log_sett.index
            # Avoid error in case index is empty or too short
            min_time_log_sett = max(log_sett_index[1] if len(log_sett_index) > 1 else 1e-10, 1e-10)

            log_sett_chart = workbook.add_chart({'type': 'scatter', 'subtype': 'straight'})
            log_sett_chart.add_series({
                'name': ['Settlement vs Log Time', 0, 1],
                'categories': ['Settlement vs Log Time', 1, 0, len(log_sett), 0],
                'values': ['Settlement vs Log Time', 1, 1, len(log_sett), 1],
            })
            log_sett_chart.set_x_axis({'name': 'Time [days]', 'log_base': 10, 'min': min_time_log_sett})
            log_sett_chart.set_y_axis({'name': 'Settlement [mm]'})
            log_sett_chart.set_title({'name': 'Settlement vs Log Time'})
            chart_sheet.insert_chart(row_pos, 1, log_sett_chart, {'x_scale': 1.5, 'y_scale': 1.5})

            row_pos += 20

            log_cons = r['df_log_cons_degree']
            log_cons_index = log_cons.index
            min_time_log_cons = max(log_cons_index[1] if len(log_cons_index) > 1 else 1e-10, 1e-10)

            log_cons_chart = workbook.add_chart({'type': 'scatter', 'subtype': 'straight'})
            log_cons_chart.add_series({
                'name': ['Consolidation vs Log Time', 0, 1],
                'categories': ['Consolidation vs Log Time', 1, 0, len(log_cons), 0],
                'values': ['Consolidation vs Log Time', 1, 1, len(log_cons), 1],
            })
            log_cons_chart.set_x_axis({'name': 'Time [days]', 'log_base': 10, 'min': min_time_log_cons})
            log_cons_chart.set_y_axis({'name': 'Consolidation Degree [-]'})
            log_cons_chart.set_title({'name': 'Consolidation Degree vs Log Time'})
            chart_sheet.insert_chart(row_pos, 1, log_cons_chart, {'x_scale': 1.5, 'y_scale': 1.5})

        output.seek(0)
        st.download_button("Download Excel Data with Charts", output.getvalue(), "consolidation_data_with_charts.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

else:
    st.info("Run calculation first to generate results.")
