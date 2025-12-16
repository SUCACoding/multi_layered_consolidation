import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import json

st.title("Loads Input (Pore Pressure)")

st.write("""
The following feature enables selection of pore pressure either as a static distribution over depth 
or as a varying profile. Users can employ sliders to adjust the varying pore pressure values interactively. 
Additionally, the number of sliders can be increased or decreased, allowing finer discretization and 
improved accuracy in representing the pore pressure variation.
""")

if 'visited_pages' not in st.session_state:
    st.session_state.visited_pages = set()

if 1 not in st.session_state.visited_pages:
    st.warning("You must complete Soil Layers before accessing Loads.")
    st.stop()

st.session_state.visited_pages.add(2)

# Full depth vector from session_state
y_points_full = st.session_state.strat['depth']
depth_min, depth_max = np.min(y_points_full), np.max(y_points_full)

# --- Upload JSON BEFORE slider/control render ---
uploaded_file = st.file_uploader("Upload load_vector JSON file", type="json")
if uploaded_file is not None:
    try:
        uploaded_data = json.load(uploaded_file)
        new_load_vector = np.array(uploaded_data, dtype=float)
        if new_load_vector.shape == y_points_full.shape:
            st.session_state.strat['load'] = new_load_vector
            st.session_state.load_vector_uploaded = True  # trigger slider update
            st.success("load_vector successfully uploaded.")
        else:
            st.error(f"Uploaded data shape {new_load_vector.shape} does not match expected shape {y_points_full.shape}.")
    except Exception as e:
        st.error(f"Failed to parse JSON: {e}")

col1, col2 = st.columns(2)

with col1:
    # Number of sliders
    num_sliders = st.number_input("Number of Pore Pressure Points", min_value=2, max_value=20, value=5, step=1)
    selected_depths = np.linspace(depth_min, depth_max, int(num_sliders))

    pore_pressure = st.number_input(
        "Pore Pressure:", min_value=0, max_value=200, step=5,
        value=st.session_state.get('pore_pressure', 100), key='pore_pressure'
    )

    pore_pressure_last = st.session_state.get('last_pore_pressure', None)
    pore_pressure_changed = pore_pressure_last != pore_pressure

    if ('pore_pressures' not in st.session_state
            or len(st.session_state.pore_pressures) != num_sliders
            or pore_pressure_changed
            or st.session_state.get('load_vector_uploaded', False)):

        if st.session_state.get('load_vector_uploaded', False) and 'load' in st.session_state.strat:
            loaded_vector = st.session_state.strat['load']
            st.session_state.pore_pressures = np.interp(selected_depths, y_points_full, loaded_vector).tolist()
        else:
            st.session_state.pore_pressures = np.full(int(num_sliders), pore_pressure).tolist()

        st.session_state.last_pore_pressure = pore_pressure
        st.session_state.load_vector_uploaded = False

    max_slider_val = max(2 * np.max(st.session_state.pore_pressures), 1)

    for i, depth in enumerate(selected_depths):
        st.session_state.pore_pressures[i] = st.slider(
            f"Pore Pressure at {depth:.1f} m",
            0, int(max_slider_val), int(st.session_state.pore_pressures[i]), step=5, key=f"pp_{i}"
        )

with col2:
    # Interpolate sliders to full depth vector
    load_vector = np.interp(y_points_full, selected_depths, st.session_state.pore_pressures)
    st.session_state.strat['load'] = load_vector

    # Calculate max value for plot x-axis, locked to 2x max load (minimum 1)
    max_plot_val = max(1.1 * np.max(load_vector), 1)

    # Plot load vector vs depth
    fig, ax = plt.subplots()
    ax.plot(load_vector, y_points_full, marker='o')
    ax.set_xlabel("Interpolated Pore Pressure (kPa)")
    ax.set_ylabel("Depth (m)")
    ax.set_title("Pore Pressure Profile")
    ax.invert_yaxis()
    ax.set_xlim(0, max_plot_val)
    st.pyplot(fig)

    # Download button for current load vector JSON
    load_vector_json = json.dumps(load_vector.tolist(), indent=2)
    st.download_button(
        label="Download load_vector (JSON)",
        data=load_vector_json,
        file_name="load_vector.json",
        mime="application/json"
    )