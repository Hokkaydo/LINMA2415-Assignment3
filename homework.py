import os
import json
from charset_normalizer import is_binary
import numpy as np
import cvxpy as cp
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch
from matplotlib.gridspec import GridSpec

def parse_json(path, T_max):
    with open(path) as fh:
        raw = json.load(fh)
    T_all = int(raw.get("time_periods", T_max))
    T = min(T_all, T_max)

    demand = np.array(raw.get("demand", 0.0), dtype=float)[:T]
    reserves = np.array(raw.get("reserves", 0.0), dtype=float)[:T]

    thermal_generators = []
    for i, (gname, g) in enumerate(raw.get("thermal_generators", {}).items()):
        name = g.get("name", gname)
        must_run = int(g.get("must_run", 0)) == 1
        power_output_minimum = g.get("power_output_minimum", 0.0)
        power_output_maximum = g.get("power_output_maximum", 0.0)
        pwl_mw = [float(pwl["mw"]) for pwl in g.get("piecewise_production", [])]
        pwl_cost = [float(pwl["cost"]) for pwl in g.get("piecewise_production", [])]
        ramp_up_limit = float(g.get("ramp_up_limit", 0.0))
        ramp_down_limit = float(g.get("ramp_down_limit", 0.0))
        ramp_startup_limit = float(g.get("ramp_startup_limit", 0.0))
        ramp_shutdown_limit = float(g.get("ramp_shutdown_limit", 0.0))
        time_up_minimum = int(g.get("time_up_minimum", 0))
        time_down_minimum = int(g.get("time_down_minimum", 0))
        power_output_t0 = float(g.get("power_output_t0", 0.0))
        unit_on_t0 = int(g.get("unit_on_t0", 0))
        time_up_t0 = int(g.get("time_up_t0", 0))
        time_down_t0 = int(g.get("time_down_t0", 0))

        startup_lag = [int(s.get("lag", 0)) for s in g.get("startup", [])]
        startup_cost = [float(s.get("cost", 0.0)) for s in g.get("startup", [])]

        thermal_generators.append(dict(
            idx=i,
            must_run=must_run,
            power_output_minimum=power_output_minimum,
            power_output_maximum=power_output_maximum,
            pwl_mw=pwl_mw,
            pwl_cost=pwl_cost,
            ramp_up_limit=ramp_up_limit,
            ramp_down_limit=ramp_down_limit,
            ramp_startup_limit=ramp_startup_limit,
            ramp_shutdown_limit=ramp_shutdown_limit,
            time_up_minimum=time_up_minimum,
            time_down_minimum=time_down_minimum,
            power_output_t0=power_output_t0,
            unit_on_t0=unit_on_t0,
            time_up_t0=time_up_t0,
            time_down_t0=time_down_t0,
            startup_lag=startup_lag,
            startup_cost=startup_cost,
            name=name,
        ))
        
    renewable_generators = []
    for i, (gname, g) in enumerate(raw.get("renewable_generators", {}).items()):
        name = g.get("name", gname)
        power_output_minimum = np.array(g.get("power_output_minimum", 0.0), dtype=float)[:T]
        power_output_maximum = np.array(g.get("power_output_maximum", 0.0), dtype=float)[:T]
        renewable_generators.append(dict(
            idx=i,
            power_output_minimum=power_output_minimum,
            power_output_maximum=power_output_maximum,
            name=name,
        ))
    
    return dict(
        T=T, 
        G=len(thermal_generators),
        W=len(renewable_generators),
        demand=demand, 
        reserves=reserves, 
        thermal_generators=pd.DataFrame(thermal_generators),
        renewable_generators=pd.DataFrame(renewable_generators)
    )

# Unit commitment problem

def uc_constraints(data, cg, pg, pw, rg, ug, vg, wg, dg, lg):
    """
    Add all UC generator constraints to a list and return (cstrs, objective).
    """
    T, G, W = data["T"], data["G"], data["W"]
    thermal = data["thermal_generators"]
    
    U = thermal["must_run"]
    Pmin = thermal["power_output_minimum"]
    Pmax = thermal["power_output_maximum"]
    Pl = thermal["pwl_mw"]
    CP = thermal["pwl_cost"]
    RU = thermal["ramp_up_limit"]
    RD = thermal["ramp_down_limit"]
    SU = thermal["ramp_startup_limit"]
    SD = thermal["ramp_shutdown_limit"]
    UT = thermal["time_up_minimum"]
    DT = thermal["time_down_minimum"]
    P0 = thermal["power_output_t0"]
    U0 = thermal["unit_on_t0"]
    UT0 = thermal["time_up_t0"]
    DT0 = thermal["time_down_t0"]
    TS = thermal["startup_lag"]
    CS = thermal["startup_cost"]
    
    cstrs = []
    
    obj = 0.0
        
    # sum_g (p_g(t) + Pd_g*u_g(t)) + sum_w p_w(t) = D(t) (2)
    bal_c = cp.sum(pg, axis=0) + cp.sum([Pmin[g] * ug[g] for g in range(G)], axis=0) + cp.sum(pw, axis=0) == data["demand"]
    cstrs += [bal_c]
    
    # sum_g r_g(t) >= R(t) (3)
    res_c = cp.sum(rg, axis=0) >= data["reserves"]
    cstrs += [res_c]

    
    for g in range(G):
        
        # sum_{t=1}^{DT_g - DT_g^0, T} (u_g(t)-1)=0 for g in G_{on}^0 (4)
        if U0[g] == 1:
            cstrs += [cp.sum([ug[g][t] - 1 for t in range(min(max(0, UT[g] - UT0[g]), T))]) == 0]
        # sum_{t=1}^{DT_g - DT_g^0, T} u_g(t)=0 for g in G_{off}^0 (5)
        if U0[g] == 0:
            cstrs += [cp.sum([ug[g][t] for t in range(min(max(0, DT[g] - DT0[g]), T))]) == 0]
        
        # ug(1) - Ug^0 = vg(1) - wg(1) (6)
        cstrs += [ug[g][0] - U0[g] == vg[g][0] - wg[g][0]]
        
        # sum_{s=1}^{Sg-1} sum_{t=\max(1,TS^{s+1}_g-DT_g^0+1)}^min(TS_g^(s+1)-1,T) delta_g^s(t) = 0 (7)
        Sg = len(TS[g])
        for s in range(Sg - 1):
            tmin = max(0, TS[g][s + 1] - DT0[g])
            tmax = min(TS[g][s + 1] - 1, T)
            cstrs += [
                cp.sum(dg[g][s, tmin:tmax]) == 0
            ]
        
        # pg(0) + rg(0) - Ug^0(Pg^0 - low Pg) <= RUg (8)
        cstrs += [pg[g][0] + rg[g][0] - U0[g] * (P0[g] - Pmin[g]) <= RU[g]]
        
        # Ug^0(Pg^0- low Pg) - pg(0) <= RDg (9)
        cstrs += [U0[g] * (P0[g] - Pmin[g]) - pg[g][0] <= RD[g]]
        
        # Ug^0(Pg^0-low Pg) <= (high Pg - low Pg)Ug^0 - max((high Pg - SDg), 0)wg(1) (10)
        cstrs += [U0[g] * (P0[g] - Pmin[g]) <= (Pmax[g] - Pmin[g]) * U0[g] - max(0.0, Pmax[g] - SD[g]) * wg[g][0]]
        
        # ug(t) >= Ug for all t (11)
        cstrs += [ug[g] >= U[g]]
        
        # ug(t) - ug(t-1) = vg(t) - wg(t) for all t in {2,...,T} (12)
        cstrs += [ug[g][1:] - ug[g][:-1] == vg[g][1:] - wg[g][1:]]
        
        # sum_{i=t-min(UTg, T)}^t vg(i) <= ug(t) for all t in {min(UTg, T),...,T} (13)
        UTg = min(UT[g], T)
        if UTg > 0:
            cs = cp.hstack([0, cp.cumsum(vg[g])])
            lhs = cs[UTg:T+1] - cs[:T-UTg+1] # sliding window sum of vg
            rhs = ug[g][UTg-1:T]
            cstrs += [lhs <= rhs]
            
        # sum_{i=t-min(DTg, T)}^t wg(i) <= 1 - ug(t) for all t in {min(DTg, T),...,T} (14)
        DTg = min(DT[g], T)
        if DTg > 0:
            cs = cp.hstack([0, cp.cumsum(wg[g])])
            lhs = cs[DTg:T+1] - cs[:T-DTg+1] # sliding window sum of wg
            rhs = 1 - ug[g][DTg-1:T]
            cstrs += [lhs <= rhs]
        
        # dg(s,t) <= sum_{i=TS^s_g}^{TS^{s+1}_g-1} wg(t-i) for t in {TS_g^{s+1}, ..., T}, for s in SG\{Sg-1} (15)
        for s in range(Sg - 1):
            # lag_s = startup[g][s]["lag"]
            # lag_next = startup[g][s + 1]["lag"]
            
            # k = np.zeros(lag_next)
            # k[lag_s:lag_next] = 1.0
            # print(wg[g].shape)
            # conv_w = cp.convolve(k, wg[g])[:T]  # convolution to compute sum of wg(t-i) over i in [lag_s, lag_next)
            # cstrs += [dg[g][s, lag_next:T] <= conv_w[lag_next:T]]
            for t in range(TS[g][s + 1] - 1, T):
                cstrs += [dg[g][s, t] <= cp.sum([wg[g][t - i] for i in range(TS[g][s], TS[g][s + 1])]),]
                
        # vg(t) = sum_{s=1}^{Sg} dg(s,t) for all t (16)
        cstrs += [vg[g] == cp.sum(dg[g], axis=0)]
        
        # pg(t) + rg(t) <= (high Pg - low Pg)ug(t) - max((high Pg - SUg), 0)vg(t) for all t (17)
        cstrs += [pg[g] + rg[g] <= (Pmax[g] - Pmin[g]) * ug[g] - cp.maximum(0.0, Pmax[g] - SU[g]) * vg[g]]
        
        # pg(t) + rg(t) <= (high Pg - low Pg)ug(t) - max((high Pg - SDg), 0)wg(t+1) for all t in {1,...,T-1} (18)
        cstrs += [pg[g][:-1] + rg[g][:-1] <= (Pmax[g] - Pmin[g]) * ug[g][:-1] - cp.maximum(0.0, Pmax[g] - SD[g]) * wg[g][1:]]
        
        # pg(t) + rg(t) - pg(t-1) <= RUg for all t in {2,...,T} (19)
        cstrs += [pg[g][1:] + rg[g][1:] - pg[g][:-1] <= RU[g]]
        
        # pg(t-1) - pg(t) <= RDg for all t in {2,...,T} (20)
        cstrs += [pg[g][:-1] - pg[g][1:] <= RD[g]]

        # pg(t) = sum_{l\in Lg}(Pg^l-Pg^1)lg^l(t) for all t (21)
        cstrs += [pg[g] == cp.sum([(Pl[g][l] - Pl[g][0]) * lg[g][l] for l in range(len(Pl[g]))], axis=0)]
        
        # cg(t) = sum_{l\in Lg}(CPg^l-CPg^1)lg^l(t) for all t (22)
        cstrs += [cg[g] == cp.sum([(CP[g][l] - CP[g][0]) * lg[g][l] for l in range(len(CP[g]))], axis=0)]
        
        # ug(t) = sum_{l\in Lg}lg^l(t) for all t (23)
        cstrs += [ug[g] == cp.sum(lg[g], axis=0)]        

        # objective terms
        obj += cp.sum(cg[g] + CP[g][0] * ug[g])
        for s, cost in enumerate(CS[g]):
            obj += cost * cp.sum(dg[g][s, :])
            
            
    renewable = data["renewable_generators"]
            
    cstrs += [pw[w] >= renewable["power_output_minimum"][w] for w in range(W)]
    cstrs += [pw[w] <= renewable["power_output_maximum"][w] for w in range(W)]

    return cstrs, obj


def solve_uc(data, is_binary=True):
    """
    Returns (total_cost, u_schedule, thermal_productions, renewable_productions).
    """
    T, G, W = data["T"], data["G"], data["W"]
    thermal = data["thermal_generators"]    
    
    cg = cp.Variable((G, T))
    pg = cp.Variable((G, T), nonneg=True)
    pw = cp.Variable((W, T), nonneg=True)
    rg = cp.Variable((G, T), nonneg=True)
    ug = cp.Variable((G, T), boolean=True) if is_binary else cp.Variable((G, T), bounds=[0, 1])
    vg = cp.Variable((G, T), boolean=True) if is_binary else cp.Variable((G, T), bounds=[0, 1])
    wg = cp.Variable((G, T), boolean=True) if is_binary else cp.Variable((G, T), bounds=[0, 1])
    dg = [cp.Variable((len(thermal["startup_lag"][g]), T), boolean=True) for g in range(G)] if is_binary else\
        [cp.Variable((len(thermal["startup_lag"][g]), T), bounds=[0, 1]) for g in range(G)]
    lg = [cp.Variable((len(thermal["pwl_mw"][g]), T), bounds=[0, 1]) for g in range(G)]        

    gen_c, objective = uc_constraints(data, cg, pg, pw, rg, ug, vg, wg, dg, lg)

    prob = cp.Problem(cp.Minimize(objective), gen_c)
    print(f"  Solving UC ({'MILP' if is_binary else 'LP relaxation'}) ...")
    prob.solve(solver=cp.GUROBI, verbose=False, MIPGap=1e-4, TimeLimit=120)
    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"UC failed: {prob.status}")
    if is_binary:
        # milp has no dual variables
        lmb, mu = None, None
    else:
        lmb = -gen_c[0].dual_value
        mu  = -gen_c[1].dual_value
    u_val   =  [np.round(ug[g].value).astype(int) if is_binary
              else ug[g].value for g in range(G)]
    pg_val  = [pg[g].value for g in range(G)]
    pw_val  = [pw[w].value for w in range(W)]
    print(f"    Status: {prob.status}   Cost: ${prob.value:,.2f}")
    return prob.value, u_val, pg_val, pw_val, lmb, mu


# Economic dispatch 

def solve_economic_dispatch(data, u_fixed):
    """
    Solve the ED LP
    Returns (total_cost, pg_schedule, pw_schedule, lambda_t, mu_t).
    """
    T, G, W = data["T"], data["G"], data["W"]
    thermal = data["thermal_generators"]

    pg = cp.Variable((G, T), nonneg=True)
    pw = cp.Variable((W, T), nonneg=True)
    cg = cp.Variable((G, T))
    lg = [cp.Variable((len(thermal["pwl_mw"][g]), T), bounds=[0, 1+1e-6]) for g in range(G)]
    rg = cp.Variable((G, T), nonneg=True)
        
    Pmin = thermal["power_output_minimum"]
    Pmax = thermal["power_output_maximum"]
    Pl = thermal["pwl_mw"]
    CP = thermal["pwl_cost"]
    
    cstrs = []
    u_fixed = np.array(u_fixed)
    bal_c = []
    for t in range(T):
        cb = cp.sum(pg[:, t]) + u_fixed[:, t].T @ Pmin + cp.sum(pw[:, t]) == data["demand"][t]
        bal_c.append(cb)
    cstrs += bal_c

    res_c = []
    for t in range(T):
        cr = (cp.sum(rg[:, t]) >= data["reserves"][t])
        res_c.append(cr)
    cstrs += res_c

    for g in range(G):
        ug = u_fixed[g]
        mw_diff = np.array(Pl[g]) - Pl[g][0]
        cost_diff = np.array(CP[g]) - CP[g][0]

        # pg(t) + rg(t) <= (high Pg - low Pg)ug(t) for all t (17)
        cstrs += [pg[g] + rg[g] <= (Pmax[g] - Pmin[g]) * ug]

        # pg(t) = sum_{l\in Lg}(Pg^l-Pg^1)lg^l(t) for all t (21)
        cstrs += [pg[g] == mw_diff @ lg[g]]

        # cg(t) = sum_{l\in Lg}(CPg^l-CPg^1)lg^l(t) for all t (22)
        cstrs += [cg[g] == cost_diff @ lg[g]]
        
        # ug_fixed(t) = sum_{l\in Lg}lg^l(t) for all t (23)
        cstrs += [u_fixed[g] == cp.sum(lg[g], axis=0)]


    renewable = data["renewable_generators"]

    for w in range(W):
        cstrs += [pw[w] >= renewable["power_output_minimum"][w]]
        cstrs += [pw[w] <= renewable["power_output_maximum"][w]]


    obj = cp.sum([cg[g] + CP[g][0] * u_fixed[g] for g in range(G)])

    prob = cp.Problem(cp.Minimize(cp.sum(obj)), cstrs)
    print("  Solving ED (LP with fixed commitment) ...")
    prob.solve(solver=cp.HIGHS, verbose=False)
    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"ED failed: {prob.status}")

    pg_val = [pg[g].value for g in range(G)]
    pw_val = [pw[w].value for w in range(W)]

    lam, mu = -np.array([c.dual_value for c in bal_c]), -np.array([c.dual_value for c in res_c])
    print(f"    Status: {prob.status}   Cost: ${prob.value:,.2f}")
    return prob.value, pg_val, pw_val, lam, mu


C = {"UC": "#1565C0", "ED": "#E65100"}
CLASS_COLOR = {"base": "#1A237E", "mid": "#1B5E20", "peak": "#B71C1C",
               "renewable": "#F57F17"}
 
 
def distinct_colors(n):
    cmap1 = plt.get_cmap("tab20",  20)
    cmap2 = plt.get_cmap("tab20b", 20)
    cmap3 = plt.get_cmap("tab20c", 20)
    palette = ([cmap1(i) for i in range(20)]
               + [cmap2(i) for i in range(20)]
               + [cmap3(i) for i in range(20)])
    return [palette[i % len(palette)] for i in range(n)]
 
 
def classify(G, u_uc):
    hours = [int(u_uc[g].sum()) for g in range(G)]
    vals  = sorted(hours, reverse=True)
    thr_b = vals[max(0, G // 3 - 1)]
    thr_p = vals[min(G - 1, 2 * G // 3)]
    return ["base" if hours[g] >= thr_b else "peak" if hours[g] <= thr_p else "mid" for g in range(G)]

def compute_uc_profits(data, u_uc, pg_uc, lam):
    T       = data["T"]
    th      = data["thermal_generators"]
    Pmin    = th["power_output_minimum"]
    Pl      = th["pwl_mw"]
    CP      = th["pwl_cost"]
    CS      = th["startup_cost"]
    U0      = th["unit_on_t0"]
 
    profits = []
    for g in range(data["G"]):
        u_g  = u_uc[g].astype(float)
        pg_g = pg_uc[g]
        CP0  = float(CP[g][0])
 
        u_prev = np.r_[float(U0[g]), u_g[:-1]]
        v_g    = np.maximum(0.0, u_g - u_prev)
 
        def variable_cost(pg_t):
            """uses pwl to compute variable cost of producing pg_t power, assuming generator is on"""
            rem = max(0.0, float(pg_t))
            cost = 0.0
            for l in range(1, len(Pl[g])):
                width = float(Pl[g][l]) - float(Pl[g][l-1])
                slope = (float(CP[g][l]) - float(CP[g][l-1])) / width
                seg   = min(rem, width)
                cost += slope * seg
                rem  -= seg
                if rem < 1e-9:
                    break
            return cost
 
        cheapest_su = float(min(CS[g])) if CS[g] else 0.0
 
        revenue = 0.0
        cost  = 0.0
        for t in range(T):
            # total production = pg + Pmin * u (pg = production above minimum if on, 0 if off)
            production  = float(pg_g[t]) + float(Pmin[g]) * float(u_g[t])
            revenue += lam[t] * production
 
            # cost = no-load + variable + startup
            cost  += (CP0 * float(u_g[t])
                        + variable_cost(float(pg_g[t]))
                        + cheapest_su * float(v_g[t]))
 
        profits.append(revenue - cost)
 
    return profits

def strip_to_1_generator(data, g_idx):
    T = data["T"]
    thermal = data["thermal_generators"]   # pandas DataFrame, rows = generators
 
    return {
        "T": T,
        "G": 1,
        "W": 0,                                        
        "demand":   np.zeros(T),                        
        "reserves": np.zeros(T),                        
        "thermal_generators": thermal.iloc[[g_idx]].reset_index(drop=True),
        "renewable_generators": data["renewable_generators"].iloc[[]].reset_index(drop=True), 
    }
    
def optimal_single_unit_profit(data, g_idx, lam, is_binary=True):
    """solve UC, dropping balance and reserve constraints to find optimal profit of single generator g_idx at prices lam"""
    T     = data["T"]
    d1    = strip_to_1_generator(data, g_idx)
    th1   = d1["thermal_generators"]
    Pmin0 = float(th1["power_output_minimum"][0])
 
    cg = cp.Variable((1, T))
    pg = cp.Variable((1, T), nonneg=True)
    rg = cp.Variable((1, T), nonneg=True)
    ug = cp.Variable((1, T), boolean=True) if is_binary else cp.Variable((1, T), bounds=[0, 1])
    vg = cp.Variable((1, T), boolean=True) if is_binary else cp.Variable((1, T), bounds=[0, 1])
    wg = cp.Variable((1, T), boolean=True) if is_binary else cp.Variable((1, T), bounds=[0, 1])
    dg = [cp.Variable((len(th1["startup_lag"][0]), T), boolean=True)] if is_binary else [cp.Variable((len(th1["startup_lag"][0]), T), bounds=[0, 1])]
    lg = [cp.Variable((len(th1["pwl_mw"][0]), T), nonneg=True)]
 
    pw_zero = np.zeros((1, T))
 
    all_cstrs, cost_obj = uc_constraints(d1, cg, pg, pw_zero, rg, ug, vg, wg, dg, lg)

    # drop balance and reserve constraints
    op_cstrs = all_cstrs[2:]
 
    total_output = pg[0] + Pmin0 * ug[0]   # shape (T,)
    revenue      = lam @ total_output       # scalar
 
    # min cost − revenue = - max profit
    prob = cp.Problem(cp.Minimize(cost_obj - revenue), op_cstrs)
    prob.solve(solver=cp.GUROBI, verbose=False, TimeLimit=120, MIPGap=1e-4)
 
    if prob.status in ("optimal", "optimal_inaccurate"):
        return -float(prob.value)
    return None
     
def compute_loc(data, u_uc, pg_uc, lam, label=""):
    G = data["G"]
 
    profits_uc  = compute_uc_profits(data, u_uc, pg_uc, lam)
    profits_opt = []
    loc         = []
 
    for g in range(G):
        pi_opt = optimal_single_unit_profit(data, g, lam)
        if pi_opt is None:
            pi_opt = profits_uc[g]   # loc = 0
 
        profits_opt.append(pi_opt)
        loc.append(max(0.0, pi_opt - profits_uc[g]))
 
    total_loc = sum(loc)
    if label:
        print(f"    LOC [{label:8s}]  total = ${total_loc:>12,.2f}")
 
    return loc, profits_uc, profits_opt, total_loc

  
def plot_part1(data, u_uc, pg_uc, pw_uc, pg_ed, pw_ed, cost_uc, cost_ed, save_path):
    T, G, W = data["T"], data["G"], data["W"]
    tt = np.arange(1, T + 1)
 
    gen_class = classify(G, u_uc)
    G_sorted  = sorted(range(G), key=lambda g: u_uc[g].sum(), reverse=True)
 
    # Pre-aggregate renewable totals
    ren_uc = np.sum(pw_uc, axis=0) if W > 0 else np.zeros(T)
    ren_ed = sum(pw_ed)
    ren_label = f"Renewables ({W} source{'s' if W > 1 else ''})"
 
    # Thermal totals
    th_uc = np.sum(pg_uc, axis=0)
    th_ed = np.sum(pg_ed, axis=0)
 
    gen_uc_total = th_uc + ren_uc
    gen_ed_total = th_ed + ren_ed
 
    fig = plt.figure(figsize=(16, 15))
    fig.suptitle("Part 1 – Unit Commitment vs Economic Dispatch", fontsize=14, fontweight="bold", y=0.995)
    gs = GridSpec(3, 2, figure=fig, hspace=0.60, wspace=0.42)
 
    # (a) Aggregate dispatch
    ax = fig.add_subplot(gs[0, :])
    ax.fill_between(tt, data["demand"], alpha=0.06, color="k")
    ax.plot(tt, data["demand"],  "k--", lw=2,   label="Demand", zorder=6)
    ax.plot(tt, gen_uc_total, color=C["UC"], lw=2.2, marker="o", ms=5, label=f"UC total  (${cost_uc:,.0f})")
    ax.plot(tt, gen_ed_total, color=C["ED"], lw=2.2, marker="s", ms=5, ls="--", label=f"ED total  (${cost_ed:,.0f})")
 
    ax.fill_between(tt, ren_uc, alpha=0.22, color=CLASS_COLOR["renewable"], label=ren_label, zorder=2)
 
    ax.set(xlabel="Period", ylabel="Power (MW)", title=f"Aggregate Generation  --  Δ cost (UC - ED) = ${cost_uc - cost_ed:+,.0f}")
    ax.legend(fontsize=8.5, loc="lower right")
    ax.grid(alpha=0.25)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(1))
 
    ax2 = ax.twinx()
    n_on = np.array([sum(int(u_uc[g][t]) for g in range(G)) for t in range(T)])
    ax2.step(tt, n_on, where="mid", color="gray", lw=1.0, ls=":", alpha=0.7)
    ax2.set_ylabel("# committed thermal", color="gray", fontsize=8)
    ax2.tick_params(axis="y", labelcolor="gray", labelsize=7)
    ax2.set_ylim(0, G * 1.5)
 
    # (b) Commitment heatmap 
    ax = fig.add_subplot(gs[1, 0])
    U  = np.array([[float(u_uc[g][t]) for t in range(T)] for g in G_sorted])
    im = ax.imshow(U, aspect="auto", cmap="Blues", vmin=0, vmax=1, extent=[0.5, T + 0.5, G - 0.5, -0.5], interpolation="nearest")
    ax.set_xticks(tt); ax.set_xticklabels(tt, fontsize=7)
    ax.set_xlabel("Period")
 
    if G <= 25:
        ax.set_yticks(range(G))
        ax.set_yticklabels(G_sorted, fontsize=max(4, 7 - G // 8))
    else:
        ax.set_yticks([])
        n_base = sum(1 for g in G_sorted if gen_class[g] == "base")
        n_mid  = sum(1 for g in G_sorted if gen_class[g] == "mid")
        n_peak = G - n_base - n_mid
        for y_pos, label, col in [
            (n_base,         f"▼ mid ({n_mid})",    CLASS_COLOR["mid"]),
            (n_base + n_mid, f"▼ peakers ({n_peak})", CLASS_COLOR["peak"]),
        ]:
            ax.axhline(y_pos - 0.5, color=col, lw=1.2, ls="--", alpha=0.8)
            ax.text(T + 0.7, y_pos - 0.5, label, va="center", color=col, fontsize=6.5, clip_on=False)
        for mid_y, label, col in [
            (n_base / 2,               f"base ({n_base})",  CLASS_COLOR["base"]),
            (n_base + n_mid / 2,       f"mid ({n_mid})",    CLASS_COLOR["mid"]),
            (n_base + n_mid + n_peak / 2, f"peak ({n_peak})", CLASS_COLOR["peak"]),
        ]:
            ax.text(-0.4, mid_y, label, va="center", ha="right", color=col, fontsize=6, rotation=90, clip_on=False)
 
    ax.set_title(
        f"Thermal Commitment  ({W} units)\n"
        f"[{ren_label} always active, not shown]",
        fontsize=8,
    )
 
    cb = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cb.set_label("u", fontsize=7); cb.ax.tick_params(labelsize=7)
 
    # (c) Stacked area / bar chart
    ax  = fig.add_subplot(gs[1, 1])
    bot = np.zeros(T)
 
    # Renewables as a single hatched band at the bottom
    ax.bar(tt, ren_uc, bottom=bot, width=0.8, color=CLASS_COLOR["renewable"], alpha=0.55, hatch="///", label=ren_label, zorder=3)
    bot = bot + ren_uc
 
    # Thermal generators stacked above in class colours
    colors = distinct_colors(G)
    for i, g in enumerate(G_sorted):
        vals = np.array([float(pg_uc[g][t]) for t in range(T)])
        ax.bar(tt, vals, bottom=bot, color=colors[i], alpha=0.88, width=0.8)
        bot += vals
 
    ax.plot(tt, data["demand"], "k--", lw=1.5)
 
    # Top-3 thermal annotation at peak
    t_pk = int(np.argmax(data["demand"]))
    top3 = sorted(range(G), key=lambda g: float(pg_uc[g][t_pk]), reverse=True)[:3]
    ren_at_pk = float(ren_uc[t_pk])
    ann = (f"Peak t={t_pk+1}:\n"
           + (f"  Renewables: {ren_at_pk:.0f} MW\n")
           + "Top thermal:\n"
           + "\n".join(f"  {g}: {float(pg_uc[g][t_pk]):.0f} MW" for g in top3))
    ax.annotate(
        ann, xy=(t_pk + 1, data["demand"][t_pk]),
        xytext=(max(1, t_pk - 2), data["demand"][t_pk] * 0.45),
        fontsize=6.5, va="top",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.90, ec="gray"),
        arrowprops=dict(arrowstyle="->", color="gray", lw=0.8),
        clip_on=False,
    )
 
    counts = {k: sum(1 for g in range(G) if gen_class[g] == k) for k in ("base", "mid", "peak")}
    legend_handles = []
    legend_handles.append(
        Patch(fc=CLASS_COLOR["renewable"], alpha=0.55, hatch="///",
                        label=ren_label)
    )
    for cls, label in [("base", "Base-load"), ("mid", "Mid-merit"), ("peak", "Peakers")]:
        legend_handles.append(
            Patch(fc=CLASS_COLOR[cls],
                           label=f"{label} ({counts[cls]})")
        )
    legend_handles.append(
        plt.Line2D([0], [0], color="k", ls="--", lw=1.5, label="Demand")
    )
    ax.legend(handles=legend_handles, fontsize=7, loc="upper left")
    ax.set(xlabel="Period", ylabel="Power (MW)", title="UC Stacked Dispatch")
    ax.grid(axis="y", alpha=0.25)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(1))
 
    # (d) Peak-period horizontal bar chart
    MAX_NAMED = 15   # show individual names only for top N thermal generators
 
    ax = fig.add_subplot(gs[2, :])
 
    # Aggregate renewable bar
    ren_uc_pk = float(ren_uc[t_pk])
    ren_ed_pk = float(ren_ed[t_pk])
 
    # Active thermal generators at peak
    active = [g for g in G_sorted if float(pg_uc[g][t_pk]) > 0.5 or float(pg_ed[g][t_pk]) > 0.5]
    active = sorted(active, key=lambda g: float(pg_uc[g][t_pk]), reverse=True)
 
    # Split into named (top MAX_NAMED) and "others"
    named  = active[:MAX_NAMED]
    others = active[MAX_NAMED:]
 
    # Build y-axis entries: renewables first, then named thermal, then "others"
    rows = []   # (label, uc_val, ed_val, color)
    rows.append((ren_label, ren_uc_pk, ren_ed_pk, CLASS_COLOR["renewable"]))
    for g in named:
        rows.append((g,
                     float(pg_uc[g][t_pk]),
                     float(pg_ed[g][t_pk]),
                     CLASS_COLOR[gen_class[g]]))
    if others:
        o_uc = sum(float(pg_uc[g][t_pk]) for g in others)
        o_ed = sum(float(pg_ed[g][t_pk]) for g in others)
        o_col = "#78909C"   # neutral gray for the aggregated "others" bar
        rows.append((f"others ({len(others)} units)", o_uc, o_ed, o_col))
 
    nR = len(rows)
    y  = np.arange(nR)
    h  = max(0.18, min(0.38, 9.0 / max(nR, 1)))
 
    bar_cols = [r[3] for r in rows]
    uc_vals  = np.array([r[1] for r in rows])
    ed_vals  = np.array([r[2] for r in rows])
    labels   = [r[0] for r in rows]
 
    ax.barh(y + h / 2, uc_vals, height=h, color=bar_cols, alpha=0.85)
    ax.barh(y - h / 2, ed_vals, height=h, color=bar_cols, alpha=0.38, hatch="//")
 
    # Value labels on UC bars (MW)
    for yi, val in zip(y, uc_vals):
        if val > 1:
            ax.text(val + 0.5, yi + h / 2, f"{val:.0f}", va="center", fontsize=5.5, color="k")
 
    fs_y = max(6, min(9, 130 // max(nR, 1)))
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=fs_y)
    ax.invert_yaxis()
 
    ax.set(
        xlabel="Power (MW)",
        title=(f"Dispatch at peak period t={t_pk+1}  "
               f"(demand = {data['demand'][t_pk]:.0f} MW,  "
               f"renewable = {ren_uc_pk:.0f} MW)"),
    )
    ax.grid(axis="x", alpha=0.25)
 
    legend_d = [
        plt.Line2D([0], [0], lw=0, marker="s", ms=9, mfc=CLASS_COLOR["base"], label="Base-load thermal"),
        plt.Line2D([0], [0], lw=0, marker="s", ms=9, mfc=CLASS_COLOR["mid"], label="Mid-merit thermal"),
        plt.Line2D([0], [0], lw=0, marker="s", ms=9, mfc=CLASS_COLOR["peak"], label="Peakers"),
        plt.Line2D([0], [0], lw=0, marker="s", ms=9, mfc=CLASS_COLOR["renewable"], label="Renewables"),
        Patch(fc="gray", alpha=0.85, label="UC dispatch"),
        Patch(fc="gray", alpha=0.38, hatch="//", label="ED dispatch"),
    ]
    ax.legend(handles=legend_d, fontsize=7.5, loc="lower right", ncol=2, framealpha=0.92)
    
    save_path = os.path.join(save_path, "part1_uc_vs_ed.pdf")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"{save_path} saved")

def main():
    
    import argparse
    parser = argparse.ArgumentParser(description="LINMA2415 – Assignment 3: Unit Commitment & Nonconvex Pricing")
    parser.add_argument("data_file", type=str, help="Path to the data JSON file")
    parser.add_argument("--save_path", type=str, default="./", help="Path to save the Part 1 plot")
    args = parser.parse_args()
    
    print("=" * 70)
    print("  LINMA2415 - Assignment 3: Unit Commitment & Nonconvex Pricing")
    print("=" * 70)
    
    data = parse_json(args.data_file, T_max=12)
    T, G = data["T"], data["G"]
    total_cap = data["thermal_generators"]["power_output_maximum"].sum() + sum(data["renewable_generators"]["power_output_maximum"].max())
    print(f"\n  Instance : {G} generators, T={T} periods")
    print(f"  Demand   : {data['demand'].min():.0f}-{data['demand'].max():.0f} MW")
    print(f"  Capacity : {total_cap:.0f} MW")

    # Part 1
    print("\n" + "─" * 70 + "\n  PART 1 - UC vs ED")
    cost_uc, u_uc, pg_uc, pw_uc, _, _ = solve_uc(data)  
    # profit for each generator
    print("\n    Generator profits at UC solution:")
    cost_ed, pg_ed, pw_ed, lmb_mp, mu_mp = solve_economic_dispatch(data, u_uc)
    print(f"    UC  total cost = ${cost_uc:>12,.2f}")
    print(f"    ED  total cost = ${cost_ed:>12,.2f}   (LP re-dispatch, fixed u)")
    print(f"    Δ              = ${cost_uc - cost_ed:>+12,.2f}")
    plot_part1(data, u_uc, pg_uc, pw_uc, pg_ed, pw_ed, cost_uc, cost_ed, save_path=args.save_path)
    
    print("\n" + "─"*70 + "\n  PART 2 – MP vs ACHP + LOC")
    cost_achp, u_achp, pg_achp, pw_achp, lmb_achp, mu_achp = solve_uc(data, is_binary=False)
    print(f"    λ_MP   : {lmb_mp.min():.3f} – {lmb_mp.max():.3f} $/MWh")
    print(f"    λ_ACHP : {lmb_achp.min():.3f} – {lmb_achp.max():.3f} $/MWh")
    
    loc_uc, profits_uc, profits_opt_uc, total_loc_uc =          compute_loc(data, u_uc, pg_uc, lmb_mp, label="UC")
    loc_achp, profits_achp, profits_opt_achp, total_loc_achp =  compute_loc(data, u_uc, pg_uc, lmb_achp, label="ACHP")
 
    print(f"    {'TOTAL LOC':>6}  {total_loc_uc:>12,.2f}  {total_loc_achp:>14,.2f}")
    
if __name__ == "__main__":
    main()
