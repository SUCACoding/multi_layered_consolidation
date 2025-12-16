# -*- coding: utf-8 -*-
"""
Created on Fri Dec 12 11:08:07 2025

@author: SUCA
"""

import streamlit as st

st.title("Multi Layered Consolidation - Tool")

st.write("""
          This tool allows engineers to simulate and predict time-dependent settlements in multi-layered soils 
          under applied loads. It accepts input parameters such as layer thickness, compressibility, permeability, 
          initial void ratio, and loading conditions. By solving consolidation equations for each layer while 
          accounting for varying drainage conditions and layer interactions, this tool provides accurate predictions 
          of settlement magnitude and rate.
          """)

with st.expander("Development"):
    st.write("""
             The tool originates from the Fehmarn Belt, where it was a module used in the settlement calculation tool developed
             directly for settlement and springs calculation for the entire project.
             
             It has later been adopted as a stand alone module and developed under the Geology and geotechnical Umbrella:
             """)
             
    st.markdown("[Geotechnical Umbrella](https://cowi.sharepoint.com/sites/c300995-community)")

with st.expander("Theory"):
    st.markdown(r"""
    When soil deposits consist of multiple horizontal layers with different properties, consolidation analysis must consider 
    each layer separately. The ground is divided into layers, each with thickness $$H_i$$, initial void ratio $$e_{0_i}$$, 
    permeability $$k_i$$, and compressibility.
    
    The vertical strain in the middle of layer *i* is:
    
    $$
    \epsilon_{v_i} = \frac{\Delta e_i}{1 + e_{0_i}}
    $$
    
    where $$\Delta e_i$$ is the change in void ratio for layer *i*.
    
    Settlement of layer *i* is:
    
    $$
    S_{c_i} = \epsilon_{v_i} \times H_i = \frac{\Delta e_i}{1 + e_{0_i}} H_i
    $$
    
    The total settlement is the sum of all layers' settlements:
    
    $$
    S_c = \sum_i S_{c_i}
    $$
    
    At the interface $$i$$ between layers $$n$$ and $$n+1$$, the flow continuity requires:
    
    $$
    k_n (1 + e_n) \frac{\partial u}{\partial z}\bigg|_i^{(n)} = k_{n+1} (1 + e_{n+1}) \frac{\partial u}{\partial z}\bigg|_i^{(n+1)}
    $$
    
    where $$u$$ is excess pore water pressure, and $$k$$ and $$e$$ represent permeability and void ratio of respective layers. 
    This ensures consistent flow of pore water across layer boundaries.
    
    For solving consolidation in a multi-layer deposit numerically, the finite difference explicit scheme is modified 
    at the layer interfaces to incorporate flow continuity.
    
    If node $$i$$ lies on the interface between layers $$n$$ and $$n+1$$, the finite difference update equation is:
    
    $$
    u_i^{j+1} = m_n u_{i-1}^j - [1 + (m_n + m_{n+1})] u_i^j + m_{n+1} u_{i+1}^j
    $$
    
    where coefficients $$m_n$$ and $$m_{n+1}$$ are defined as:
    
    $$
    m_n = \frac{2\alpha_n \alpha_{n+1}}{\alpha_n + \alpha_{n+1}}, \quad
    m_{n+1} = \frac{2\alpha_n \alpha_{n+1}}{\alpha_n + \alpha_{n+1}}
    $$
    
    and
    
    $$
    \alpha_n = \frac{k_n (1 + e_n) C_{v_n} \Delta t}{(\Delta z_n)^2}
    $$
    
    The corresponding implicit or $$\theta$$-method schemes apply similar substitutions to account for varying soil properties.
    
    This approach enables accurate modeling of consolidation behavior in stratified soils, accounting for differing soil compressibility, permeability, and drainage conditions in each layer, and properly handling flow continuity at layer boundaries.
    """)

with st.expander('Code'):
    st.markdown("The Python code from the theory has been shown below.")
    code = '''
    ```python
    def compute_explicit(upore: np.array, perm: np.array, eoed: np.array, void: np.array,
                         gamw: float, d_t: float, d_z: np.array, boundary: str) -> np.array:
        n = len(upore)
        abc_arr = np.zeros((3, n + 2))
        alpha = np.zeros(n + 1)
        beta = np.zeros(n + 1)
    
        # Pad upore to apply boundary conditions
        upore_pad = np.concatenate(([0], upore, [0]))
        abc_arr[1, [0, -1]] = 1  # Boundary rows main diagonal = 1
    
        beta[:-1] = perm * eoed / gamw * d_t / (d_z ** 2)
        alpha[:-1] = perm * (1 + void) / d_z
    
        beta[-1] = beta[-2]
        alpha[-1] = alpha[-2]
    
        numerator = alpha[:-1] / beta[:-1] + alpha[1:] / beta[1:]
        m_n0 = 2 * alpha[:-1] / numerator
        m_n1 = 2 * alpha[1:] / numerator
    
        abc_arr[0, 1:-1] = -m_n0
        abc_arr[1, 1:-1] = 1 + m_n0 + m_n1
        abc_arr[2, 1:-1] = -m_n1
    
        if boundary == "closed":
            abc_arr[0, -1] = -m_n1[-1] - m_n0[-1]
            abc_arr[1, -1] = 2 + m_n1[-1] + m_n0[-1]
            upore_pad[-1] = upore_pad[-2]
    
        # Prepare banded matrix for solve_banded (with proper shifts)
        ab = np.zeros_like(abc_arr)
        ab[0, 1:] = abc_arr[0, 1:]
        ab[1, :] = abc_arr[1, :]
        ab[2, :-1] = abc_arr[2, :-1]
    
        sol_banded = solve_banded((1, 1), ab, upore_pad)
    
        return sol_banded[1:-1]
    
    def stratigraphy(soil_layers, dZ, pressure):
        strat = {'depth':[], 'eoed':[], 'void':[], 'k_perm':[], 'load':[], 'dZ':[]}
        for  _, layer in enumerate(soil_layers):
            nodes = (layer['top']-layer['bot']) / dZ
            strat['dZ'].extend(np.ones(int(nodes))*dZ)
            strat['eoed'].extend(np.ones(int(nodes))*layer['eoed'])
            strat['void'].extend(np.ones(int(nodes))*layer['void'])
            strat['k_perm'].extend(np.ones(int(nodes))*layer['k_perm'])
            strat['load'].extend(np.ones(int(nodes))*pressure)
        strat['depth'] = np.cumsum(strat['dZ'])
        for key, value in strat.items():
            if isinstance(value, list):
                strat[key] = np.array(value)
        return strat
    
        for key, value in strat.items():
            if isinstance(value, list):
                strat[key] = np.array(value)
    
        return strat
    '''
    st.markdown(code)
    
    run_code = '''
    ```python
    soil_layer_1 = {'name': "meltwater", 'eoed': 60000, 'void': 1.3, 'k_perm': 1e-5, 'top': 0, 'bot': -10}
    soil_layer_2 = {'name': "Till", 'eoed': 100000, 'void': 1.3, 'k_perm': 1e-10, 'top': -10, 'bot': -30}
    soil_layer_3 = {'name': "Clay", 'eoed': 40000, 'void': 1.3, 'k_perm': 1e-10, 'top': -30, 'bot': -60}
    soil_layers = [soil_layer_1, soil_layer_2, soil_layer_3]
    
    consolidation_time = 1000  # days
    pressure = 100             # kPa applied on entire vector
    dZ = 0.2                   # soil increment in meters
    dT = 1                     # time increment in days
    boundary = "closed"          # boundary condition
    gamW = 10                  # kN/m^3 water density
    
    strat = stratigraphy(soil_layers, dZ, pressure)
    time_count = int(np.ceil(consolidation_time / dT))
    time_vec = np.cumsum(np.ones(time_count) * dT)
    UPore_Calc = np.zeros((len(strat['depth']), len(time_vec)))
    UPore_Calc[:, 0] = strat['load']
    
    for i in range(1, len(time_vec)):
        UPore_Calc[:, i] = compute_explicit(UPore_Calc[:, i - 1], strat['k_perm'], 
                                            strat['eoed'], strat['void'], gamW, 
                                            dT * 24 * 3600, strat['dZ'], boundary)
    '''
    st.markdown("An example of how to run this code has been shown below")
    st.markdown(run_code)


