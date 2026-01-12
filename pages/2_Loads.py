import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import json

st.set_page_config(layout="wide")

st.title("Loads Input (Pore Pressure)")

st.write("""
This feature allows selection of pore pressure either as a static distribution over depth 
or as a varying profile using interactive sliders. You can change the number of sliders to 
adjust the discretization resolution for pore pressure variation.
""")

# --- Session State Initialization ---
if 'visited_pages' not in st.session_state:
    st.session_state.visited_pages = set()
if 1 not in st.session_state.visited_pages:
    st.warning("You must complete Soil Layers before accessing Loads.")
    st.stop()
st.session_state.visited_pages.add(2)

# Depth info from stratigraphy in session state
depths_full = st.session_state.strat['depth']
depth_min, depth_max = np.min(depths_full), np.max(depths_full)

# --- File uploader for load_vector JSON ---
uploaded_file = st.file_uploader("Upload load_vector JSON file", type="json")
if uploaded_file is not None:
    try:
        uploaded_data = json.load(uploaded_file)
        loaded_vector = np.array(uploaded_data, dtype=float)
        if loaded_vector.shape == depths_full.shape:
            st.session_state.strat['load'] = loaded_vector
            st.session_state.load_vector_uploaded = True  # flag to trigger slider update
            st.success("load_vector successfully uploaded.")
        else:
            st.error(
                f"Uploaded data shape {loaded_vector.shape} "
                f"does not match expected shape {depths_full.shape}."
            )
    except Exception as e:
        st.error(f"Failed to parse JSON: {e}")

col1, col2 = st.columns(2)

with col1:
    # Number of interactive sliders for pore pressure
    num_sliders = st.number_input(
        "Number of Pore Pressure Points",
        min_value=2,
        max_value=20,
        value=5,
        step=1
    )
    selected_depths = np.linspace(depth_min, depth_max, num_sliders)

    # Static pore pressure default value input
    static_pore_pressure = st.number_input(
        "Static Pore Pressure (default value):",
        min_value=0,
        max_value=200,
        step=5,
        value=st.session_state.get('static_pore_pressure', 100),
        key='static_pore_pressure'
    )

    # Check if update is needed for pore pressures stored in session state
    last_static = st.session_state.get('last_static_pore_pressure', None)
    input_changed = (last_static != static_pore_pressure)
    need_update = (
        ('pore_pressures' not in st.session_state) or
        (len(st.session_state.pore_pressures) != num_sliders) or
        input_changed or
        st.session_state.get('load_vector_uploaded', False)
    )

    # Update pore pressures accordingly
    if need_update:
        if st.session_state.get('load_vector_uploaded', False) and 'load' in st.session_state.strat:
            loaded_vector = st.session_state.strat['load']
            interp_pressures = np.interp(selected_depths, depths_full, loaded_vector).tolist()
            st.session_state.pore_pressures = interp_pressures
        else:
            st.session_state.pore_pressures = [static_pore_pressure] * num_sliders

        st.session_state.last_static_pore_pressure = static_pore_pressure
        st.session_state.load_vector_uploaded = False

    max_slider_val = max(2 * max(st.session_state.pore_pressures), 1)

    # Render sliders to adjust pore pressures at selected depths
    for i, depth in enumerate(selected_depths):
        st.session_state.pore_pressures[i] = st.slider(
            f"Pore Pressure at {depth:.1f} m",
            min_value=0,
            max_value=int(max_slider_val),
            value=int(st.session_state.pore_pressures[i]),
            step=5,
            key=f"pp_{i}"
        )

with col2:
    # Interpolate from slider values to full depth vector
    load_vector = np.interp(depths_full, selected_depths, st.session_state.pore_pressures)
    st.session_state.strat['load'] = load_vector

    # Plot setup
    max_plot_val = max(1, 1.1 * np.max(load_vector))

    fig, ax = plt.subplots()
    ax.plot(load_vector, depths_full, marker='o')
    ax.invert_yaxis()
    ax.set_xlabel("Interpolated Pore Pressure (kPa)")
    ax.set_ylabel("Depth (m)")
    ax.set_title("Pore Pressure Profile")
    ax.set_xlim(0, max_plot_val)
    st.pyplot(fig)

    # Enable download of current load vector as JSON
    load_vector_json = json.dumps(load_vector.tolist(), indent=2)
    st.download_button(
        label="Download load_vector (JSON)",
        data=load_vector_json,
        file_name="load_vector.json",
        mime="application/json"
    )
