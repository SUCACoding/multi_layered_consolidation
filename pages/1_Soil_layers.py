# -*- coding: utf-8 -*-
"""
Created on Fri Dec 12 11:07:33 2025

@author: SUCA
"""
import streamlit as st
import pandas as pd
#import numpy as np
import json
from pore_pressure import stratigraphy

st.set_page_config(layout="wide")

# Initialize session state for visited pages and current page
if 'visited_pages' not in st.session_state:
    st.session_state.visited_pages = set()

st.session_state.visited_pages.add(1)
    
if "dZ" not in st.session_state:
    st.session_state.dZ = 0.2  # or None or your default
    
st.title("Soil Layer")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Input description")
    st.markdown("""
    Provide details for each soil layer including:
    
    - **Name**: Identifier for the soil layer (e.g., clay, sand, silt).
    - **Elevation**: Top and bottom elevations (meters) defining the thickness. Ensure no overlap
    - **Void Ratio (e)**: Measure of the volume of voids relative to solids in the soil.
    - **Permeability (k)**: Hydraulic conductivity (m/s), indicating how easily water flows through the soil.
    """)
    
with col2:
    # File Uploader and processing, BEFORE data_editor display
    st.subheader("Upload soil layers")
    st.markdown("If a previous soil layer stratigraphy has been saved, it can be uploaded below")
    uploaded_file = st.file_uploader("Drag and drop a JSON or CSV file or click to upload")

    if uploaded_file is not None:
        bytes_data = uploaded_file.read()
        try:
            if uploaded_file.name.endswith(".json"):
                data = json.loads(bytes_data.decode("utf-8"))
                soil_layers = data.get("soil_layers", [])
                if soil_layers:
                    st.session_state.edited_df = pd.DataFrame(soil_layers)

            elif uploaded_file.name.endswith(".csv"):
                df_upload = pd.read_csv(uploaded_file)
                st.session_state.edited_df = df_upload

            else:
                st.error("Unsupported file type! Please upload JSON or CSV.")
        except Exception as e:
            st.error(f"Error loading file: {e}")

    st.markdown("The soil increment, used to discritize the soil layers into multiple horizontal layers can be specified below")
    
    st.session_state.dZ = st.number_input("dZ:", min_value=0.1, max_value=1.0, step=0.1, value=0.2)

default_df = pd.DataFrame(
    [
        {"name": "mw_sand", "eoed": 60000, "void": 1.0, "k_perm": 1e-5, "top": 0, "bot": -5},
        {"name": "clay", "eoed": 100000, "void": 1.2, "k_perm": 1e-7, "top": -5, "bot": -20},
        {"name": "pal_clay", "eoed": 50000, "void": 1.3, "k_perm": 1e-10, "top": -20, "bot": -50},
    ]
)


# Initialize if not in session_state
st.subheader("Soil layer input table")
if "edited_df" not in st.session_state:
    st.session_state.edited_df = default_df


    
st.markdown("To use the table, either edit in the existing data or add a new row below.")
st.markdown("Existing rows can be deleted by checkmarking row on the left and pressing the delete button.")

# Display and edit table with key for state sync
edited_df = st.data_editor(st.session_state.edited_df, key="data_editor", num_rows="dynamic")

# Update session state when table changes
st.session_state.edited_df = edited_df

st.markdown(r"""
Some typical values have been provided below for common soils.

| Soil Type       | Void Ratio (e) | Permeability \(k\) (m/s)               |
|-----------------|----------------|----------------------------------------|
| Meltwater Sand  | 0.5 – 1.0      | $$1 \times 10^{-4} \text{ to } 1 \times 10^{-5}$$  |
| Clay Till       | 0.4 – 0.7      | $$1 \times 10^{-6} \text{ to } 1 \times 10^{-8}$$ |
| Soft Clay       | 1.0 – 1.5      | $$1 \times 10^{-8} \text{ to } 1 \times 10^{-11}$$ |

*Note:* Site-specific investigations should be used for precise analyses.
""")

# Prepare JSON for download
st.subheader("Download soil layer")
combined_data = {
    "soil_layers": edited_df.to_dict(orient="records")
}
soil_layers = st.session_state.edited_df.to_dict(orient='records')
st.session_state.strat = stratigraphy(soil_layers, st.session_state.dZ, 100)
combined_json = json.dumps(combined_data, indent=4)

st.download_button(
    label="Download JSON Data",
    data=combined_json,
    file_name="input_data_with_soil_layers.json",
    mime="application/json"
)
