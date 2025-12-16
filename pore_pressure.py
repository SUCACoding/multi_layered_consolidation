import numpy as np
from scipy.linalg import solve_banded

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

    if boundary == "Closed":
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

if __name__ == '__main__':
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

    dset_tot = strat['load']/strat['eoed']*strat['dZ']*1000
    set_tot = np.cumsum(dset_tot[::-1])[::-1]
    
    pow_n = 1.2
    i = 1
    set_u_calc = 0.1
    t_step_sum = np.zeros(1000)
    dT = 0.5
    
    while (set_u_calc/set_tot[0])<0.9:
        t_step = (i*dT)**pow_n
        UPore_Calc[:, i] = compute_explicit(
            UPore_Calc[:, i - 1], strat['k_perm'], strat['eoed'],
            strat['void'], gamW, t_step* 24 * 3600, strat['dZ'], boundary 
        )
        dset_calc = (strat['load']-UPore_Calc[:,i])/strat['eoed']*strat['dZ']*1000
        set_temp_calc = np.cumsum(dset_calc[::-1])[::-1]
        set_u_calc = set_temp_calc[0]
        t_step_sum[i] = t_step+t_step_sum[i-1]
        i += 1
    t_step_sum = t_step_sum[0:i-1]
    UPore_Calc = UPore_Calc[:,0:i-1]

    # for i in range(1, len(time_vec)):
    #     UPore_Calc[:, i] = compute_explicit(UPore_Calc[:, i - 1], strat['k_perm'], strat['eoed'],
    #                                        strat['void'], gamW, dT * 24 * 3600, strat['dZ'], boundary)
    set_frac = [0, 0.25, 0.50, 0.60, 0.70, 0.80, 0.90]
    cons_deg = (1 - UPore_Calc / strat['load'][:, np.newaxis])    
    idx_cons = np.zeros([len(strat['depth']), len(set_frac)], dtype = int)
    for i, val in enumerate(set_frac):
        test = (cons_deg>=val)
        temp = np.argmax(test, axis = 1)
        has_value = test.any(axis= 1)
        temp[~has_value] = len(t_step_sum)
        idx_cons[:,i] = temp
        
    
    
    #calc settlements
    d_eps = np.zeros_like(UPore_Calc)
    d_eps[:,1:] = (UPore_Calc[:,0:-1]-UPore_Calc[:,1:])/strat['eoed'][:, np.newaxis]
    eps = np.cumsum(d_eps,axis = 1)
    d_sett = strat['dZ'][:, np.newaxis]*eps*1000
    sett = np.cumsum(d_sett[::-1,:],axis=0)[::-1,:]