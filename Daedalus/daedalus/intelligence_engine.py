import math
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import requests
from pulp import (
    LpProblem,
    LpMinimize,
    LpVariable,
    lpSum,
    LpStatus,
    PULP_CBC_CMD,
    value,
)

from engine_config import EngineData


# -----------------------------
# 1. Simulation Parameters (NO CHANGES)
# -----------------------------

DT_MIN = 5                              # time step size (minutes)
TIME_HORIZON_MIN = 120                  # total duration (minutes)
STEPS = TIME_HORIZON_MIN // DT_MIN

ALPHA = 0.8                             # EWMA (weight given to previous estimate)
CAPACITY_PER_BUS = 48                   # bus capacity (pax) per bus (MADRID AVG CAPACITY)
U_TARGET = 0.8                          # utilization target Λ/μ (not directly used here)
Kp = 0.25                               # proportional gain (not used in this version)
Ki = 0.02                               # integral gain (not used in this version)

F_MIN = 2.0                             # veh/h (max headway 30 min)
F_MAX = 15.0                            # veh/h (min headway 4 min)

TOTAL_DEMAND_TARGET_H = 50000.0         # total system demand (pax/h) for the simulation
MAX_LINES_ADJUSTED_PER_DISTRICT = 2     # Maximum number of lines to adjust (reinforce/sub-reinforce) per district
WEIGHT_LINE_ADJUSTMENT = 100.0          # Penalty for adjusting more than the limit (a large value)

# Observation/arrival simulation noise parameter
# σ (sigma): Fraction of base demand used as standard deviation
SIGMA_DEMAND_FRACTION = 0.15
RNG = np.random.default_rng(1234)


# -----------------------------
# 2. Load Network Data (NO CHANGES)
# -----------------------------

data = EngineData()  # uses default paths: salida_red_simulada/...

lines = data.lines
stops = data.stops
line_to_stops = data.line_to_stops
stop_to_distrito = data.stop_to_distrito
lambda_base = data.lambda_base
cycle_time_h = data.cycle_time_h
prior_freq = data.prior_freq

print("DEBUG_STOPS and network sizes")
print(f"Lines: {len(lines)}")
print(f"Stops: {len(stops)}")
print("Districts:", data.distritos)

# Total reference fleet (in "bus-hours", sum C_l * f_l)
B_tot = sum(cycle_time_h[ln] * prior_freq[ln] for ln in lines)
print(f"Total reference fleet (C * f): {B_tot:.1f} bus-hours")


# -----------------------------
# 2.b. Supabase (NO CHANGES)
# -----------------------------

SUPABASE_URL = "https://rrbuwzcoiyncvmzctvnr.supabase.co"
SUPABASE_SERVICE_KEY = (
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InJyYnV3emNvaXlu"
    "Y3ZtemN0dm5yIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc2MzE1NzE1MywiZXhwIjoyMDc4NzMzMTUzfQ."
    "7tJ32D_TQeQ9xWVW8lYTv_T2c0N6U08adn2Llk-8iVk"
)  # service_role from Settings → API


def supabase_headers():
    return {
        "apikey": SUPABASE_SERVICE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }


def create_run(scenario, description):
    payload = {"scenario": scenario, "description": description}
    resp = requests.post(
        f"{SUPABASE_URL}/rest/v1/runs",
        headers=supabase_headers(),
        json=payload,
    )
    resp.raise_for_status()
    data_json = resp.json()
    run_id = data_json[0]["run_id"]
    print("Run created in Supabase:", run_id)
    return run_id


def insert_run_line_freq(run_id, freq_before, freq_after, buses_before, buses_after):
    rows = []
    for line_id in freq_before.keys():
        rows.append(
            {
                "run_id": run_id,
                "line_id": line_id,
                "freq_before": float(freq_before[line_id]),
                "freq_after": float(freq_after.get(line_id, freq_before[line_id])),
                "buses_before": int(buses_before.get(line_id, 0)),
                "buses_after": int(buses_after.get(line_id, 0)),
            }
        )
    if not rows:
        return
    resp = requests.post(
        f"{SUPABASE_URL}/rest/v1/run_line_freq",
        headers=supabase_headers(),
        json=rows,
    )
    resp.raise_for_status()
    print("Line frequencies saved in Supabase:", len(rows))


def insert_run_stop_stats(run_id, lambda_obs, wait_time_min, crowding_index):
    """
    Inserts a row per stop into the run_stop_stats table:

      run_id, stop_id, lambda_obs, wait_time_min, crowding_index
    """
    rows = []
    for stop_id in lambda_obs.keys():
        rows.append(
            {
                "run_id": int(run_id),
                "stop_id": str(stop_id),
                "lambda_obs": float(lambda_obs[stop_id]),
                "wait_time_min": float(wait_time_min[stop_id]),
                "crowding_index": float(crowding_index[stop_id]),
            }
        )

    if not rows:
        return

    resp = requests.post(
        f"{SUPABASE_URL}/rest/v1/run_stop_stats",
        headers=supabase_headers(),
        json=rows,
    )
    resp.raise_for_status()
    print("Stop statistics saved in Supabase:", len(rows))


# -----------------------------
# 3. Prepare Auxiliary Maps
# -----------------------------

# Map stop -> [lines passing through]
stop_to_lines = {s: [] for s in stops}
for ln, s_list in line_to_stops.items():
    for s in s_list:
        if s not in stop_to_lines:
            stop_to_lines[s] = []
        stop_to_lines[s].append(ln)

# Normalize lambda_base to create a "true" simulation demand
sum_base = sum(lambda_base.values())
if sum_base == 0:
    raise RuntimeError("lambda_base sums to 0; check input data.")

scale = TOTAL_DEMAND_TARGET_H / sum_base
lambda_true_base = {s: lambda_base[s] * scale for s in stops}

# Time-of-day profile (smooth rise until stabilization)
t_axis = np.arange(STEPS)
tod_profile = 0.7 + 0.6 * (1 - np.exp(-t_axis / 3.0))  # between ~0.7 and ~1.3

# Select some districts as "hotter" to introduce peaks
hot_districts = {"01", "03", "08", "13"}  # example: Centro, Retiro, etc.


# =========================================================================
# 💡 MODIFICATION: Assign the main district to each line
# =========================================================================

def get_line_main_district(
    line_to_stops: dict, stop_to_distrito: dict
) -> tuple[dict, dict]:
    """
    Calculates the main district for each line (the one with the most stops).
    Returns:
      - line_to_main_district: {line_id: district_id}
      - district_to_lines: {district_id: [line_id, ...]}
    """
    line_to_main_district = {}
    district_to_lines = {d: [] for d in data.distritos} # Initialize with all districts

    for ln, stops_in_line in line_to_stops.items():
        district_counts = {}
        for s in stops_in_line:
            d = stop_to_distrito.get(s)
            if d:
                district_counts[d] = district_counts.get(d, 0) + 1

        if not district_counts:
            # Assign a default district or None if it has no valid stops
            line_to_main_district[ln] = None
            continue

        # Find the district with the maximum number of stops
        main_district = max(district_counts, key=district_counts.get)
        line_to_main_district[ln] = main_district
        if main_district:
             district_to_lines.setdefault(main_district, []).append(ln)
    
    # Clean up districts without assigned lines for the reverse mapping (optional)
    district_to_lines = {d: lns for d, lns in district_to_lines.items() if lns}
    
    return line_to_main_district, district_to_lines

# Calculate main district maps
line_to_main_district, district_to_lines = get_line_main_district(
    line_to_stops, stop_to_distrito
)
print("\nMapping of lines to main district calculated.")


# -----------------------------
# 4. Initial Simulation State (NO CHANGES)
# -----------------------------

# Frequencies per line (starting from simulated frequencies)
f = {ln: float(prior_freq[ln]) for ln in lines}

# Queues per stop
Q = {s: 0.0 for s in stops}
Q_prev = {s: 0.0 for s in stops}

# Arrival rate estimation λ_hat per stop (initial = base demand)
lambda_hat = {s: lambda_true_base[s] for s in stops}

# Integrator for each line (integral part of PI control) — not used here,
# but kept in case the controller is expanded.
I_err = {ln: 0.0 for ln in lines}


# -----------------------------
# 5. Auxiliary Functions (NO CHANGES)
# -----------------------------

def aggregate_lambda_per_line(lam_s: dict) -> dict:
    """Sums lambda_hat by stops to get Λ_l per line."""
    return {
        ln: float(sum(lam_s[s] for s in line_to_stops[ln]))
        for ln in lines
    }


def capacity_per_line(freq: dict) -> dict:
    """Capacity μ_l per line (pax/h)."""
    return {ln: CAPACITY_PER_BUS * freq[ln] for ln in lines}


# -----------------------------
# 5.bis. Fleet Optimization with PuLP (Integer Linear Programming)
# -----------------------------

def optimizar_frecuencias_pulp_subconjunto(
    lines_to_optimize: list[str],
    Lambda_line: dict,
    cycle_time_h: dict,
    prior_freq: dict,
    capacity_per_bus: float,
    F_min: float,
    F_max: float,
    f_current: dict,
    w_over: float = 10.0,
    w_dev_prior: float = 1.0,
    w_dev_step: float = 0.5,
    max_dev_prior_rel: float = 0.50,
    max_change_step_rel: float = 0.30,
    max_adjusted_lines: int = 3,  # New argument: N
    w_adjustment_limit: float = 100.0, # New argument: Penalty for exceeding N
):
    """
    Integer Linear Programming model to reassign buses ONLY
    among the provided subset of lines, while maintaining the TOTAL fleet
    and limiting the number of lines that change.
    """
    lines_loc = lines_to_optimize

    # 1) A priori buses and total fleet FOR THE SUBSET
    b0 = {}
    for ln in lines_loc:
        C = cycle_time_h[ln]
        f0 = prior_freq[ln]
        b0[ln] = C * f0

    B_tot_subconjunto = sum(b0.values())
    B_tot_int = int(round(B_tot_subconjunto))
    b_prev = {ln: cycle_time_h[ln] * f_current[ln] for ln in lines_loc}
    # ---------------------------------------------------------------

    # 2) Model
    prob = LpProblem(f"FleetReassignment_Subset_{len(lines_loc)}L", LpMinimize)

    # 3) Variables
    b = {}                  # nº buses per line
    o = {}                  # overload (pax/h)
    d_prior_plus = {}
    d_prior_minus = {}
    d_step_plus = {}
    d_step_minus = {}
    z = {}                  # Binary variable: 1 if the line changes, 0 if not
    M = B_tot_int * 2       # Large constant M for binary logic

    for ln in lines_loc:
        C = cycle_time_h[ln]

        # bus limits based on F_min and F_max (NO CHANGES)
        # ... [b_min and b_max limit calculation code] ...
        b_min_f = max(1, math.ceil(F_min * C))
        b_max_f = max(b_min_f, math.floor(F_MAX * C))
        b_min_prior = max(1.0, (1.0 - max_dev_prior_rel) * b0[ln])
        b_max_prior = (1.0 + max_dev_prior_rel) * b0[ln]
        b_min = max(b_min_f, math.floor(b_min_prior))
        b_max = min(b_max_f, math.ceil(b_max_prior))
        if b_max < b_min:
            b_max = b_min
        # ----------------------------------------
        
        b[ln] = LpVariable(f"b_{ln}", lowBound=b_min, upBound=b_max, cat="Integer")
        o[ln] = LpVariable(f"o_{ln}", lowBound=0.0, cat="Continuous")

        d_prior_plus[ln] = LpVariable(f"dprior_plus_{ln}", lowBound=0.0, cat="Continuous")
        d_prior_minus[ln] = LpVariable(f"dprior_minus_{ln}", lowBound=0.0, cat="Continuous")
        d_step_plus[ln] = LpVariable(f"dstep_plus_{ln}", lowBound=0.0, cat="Continuous")
        d_step_minus[ln] = LpVariable(f"dstep_minus_{ln}", lowBound=0.0, cat="Continuous")
        
        # 💡 Declare the binary variable
        z[ln] = LpVariable(f"z_{ln}", cat="Binary")

    # 4) Total fleet constraint FOR THE SUBSET (NO CHANGES)
    prob += lpSum(b[ln] for ln in lines_loc) == B_tot_int, "TotalFleet_Subset"

    # 5) Constraints per line (including the binary variable)
    for ln in lines_loc:
        C = cycle_time_h[ln]
        Lambda_l = Lambda_line.get(ln, 0.0)
        cap_factor = capacity_per_bus / C

        # 5.a Overload: Λ_l - μ_l <= o_l (NO CHANGES)
        prob += Lambda_l - cap_factor * b[ln] <= o[ln], f"Overload_{ln}"

        # 5.b Deviation from a priori buses (NO CHANGES)
        prob += (
            b[ln] - b0[ln] == d_prior_plus[ln] - d_prior_minus[ln]
        ), f"DevPrior_{ln}"

        # 5.c Deviation from previous step (NO CHANGES)
        prob += (
            b[ln] - b_prev[ln] == d_step_plus[ln] - d_step_minus[ln]
        ), f"DevStep_{ln}"

        # 5.d Hard bounds on change from previous step (NO CHANGES)
        max_step = max_change_step_rel * max(1.0, b_prev[ln])
        prob += b[ln] - b_prev[ln] <= max_step, f"DevStepMaxPos_{ln}"
        prob += b_prev[ln] - b[ln] <= max_step, f"DevStepMaxNeg_{ln}"
        
        # ------------------------------------------------------------------
        # 💡 M Constraint: Detect if there is a change
        # If b[ln] != b_prev[ln], then z[ln] MUST be 1.
        # If b[ln] == b_prev[ln], z[ln] can be 0 or 1, but the Objective Function will force it to 0.
        
        # | b[ln] - b_prev[ln] | <= M * z[ln]
        # Equivalent to:
        
        # 1. b[ln] - b_prev[ln] <= M * z[ln]
        prob += b[ln] - b_prev[ln] <= M * z[ln], f"ChangePos_{ln}"
        
        # 2. b_prev[ln] - b[ln] <= M * z[ln]
        prob += b_prev[ln] - b[ln] <= M * z[ln], f"ChangeNeg_{ln}"
        # ------------------------------------------------------------------

    # 6) Constraint for the limit of adjusted lines (Constraint)
    # A slack variable is added to penalize if the number of adjusted lines
    # (sum(z)) exceeds the limit N (MAX_LINES_ADJUSTED_PER_DISTRICT).
    s = LpVariable("slack_lines", lowBound=0.0, cat="Continuous")
    
    # sum(z) - N <= s
    prob += lpSum(z[ln] for ln in lines_loc) - max_adjusted_lines <= s, "MaxLinesAdjustedConstraint"

    # 7) Objective Function: Add penalty for slack
    # The slack 's' will be > 0 only if more than N lines were adjusted.
    prob += (
        w_over * lpSum(o[ln] for ln in lines_loc)
        + w_dev_prior
        * lpSum(d_prior_plus[ln] + d_prior_minus[ln] for ln in lines_loc)
        + w_dev_step
        * lpSum(d_step_plus[ln] + d_step_minus[ln] for ln in lines_loc)
        # 💡 Penalty if the limit of adjusted lines is exceeded
        + w_adjustment_limit * s 
    ), "MinimizeOverloadAndDeviations"
    
    # 8) Solve (NO CHANGES)
    prob.solve(PULP_CBC_CMD(msg=0))

    status = LpStatus[prob.status]
    
    f_opt_subconjunto = {}
    for ln in lines_loc:
        b_val = b[ln].varValue
        C = cycle_time_h[ln]
        f_opt_subconjunto[ln] = max(F_min, min(F_max, b_val / C))

    info = {
        "status": status,
        "B_tot_int": B_tot_int,
        "obj_value": value(prob.objective),
        "lines_adjusted_count": sum(value(z[ln]) for ln in lines_loc) if status == "Optimal" else -1
    }
    return f_opt_subconjunto, info


# -----------------------------
# 5.ter. Accumulators per stop for camera/hotspot stats (NO CHANGES)
# -----------------------------

# Total arrivals per stop (for lambda_obs)
total_arrivals_per_stop = {s: 0.0 for s in stops}

# Integral of the queue over time: sum Q(s,t) * dt_h (for Little's Law)
cum_Q_time_per_stop = {s: 0.0 for s in stops}

# Total service per stop (passengers served) for average capacity
total_service_per_stop = {s: 0.0 for s in stops}


# -----------------------------
# 6. Main Simulation Loop
# -----------------------------

history = []
debug_history = []

for step in range(STEPS):
    minute = step * DT_MIN
    dt_h = DT_MIN / 60.0

    # 6.1 Generate true demand and arrivals for this window (NO CHANGES)
    lambda_true = {}
    arrivals_window = {}
    # New map of the standard deviation (based on base demand)
    sigma_per_stop = {
        s: lambda_true_base[s] * SIGMA_DEMAND_FRACTION 
        for s in stops
    }
    
    for s in stops:
        base = lambda_true_base[s]
        # time-of-day factor
        factor_tod = tod_profile[step]

        # soft multiplicative noise
        noise = RNG.lognormal(mean=0.0, sigma=0.3)

        # additional spikes if the district is "hot"
        d = stop_to_distrito.get(s)
        spike = 1.0
        if d in hot_districts and RNG.random() < 0.10:
            spike = RNG.uniform(1.5, 2.5)

        lam_t = base * factor_tod * noise * spike  # true lambda (pax/h)
        lambda_true[s] = lam_t

        # ------------------------------------------------------------------
        #  CHANGE: Simulate observed arrivals with Normal noise
        # ------------------------------------------------------------------
        # 1. Mean arrival rate for this window
        mu_window = lam_t * dt_h
        
        # 2. Standard deviation for this window (using the base, adjusted for time)
        # Note: We use the 15% sigma of the base as a fixed measurement error value.
        sigma_window = sigma_per_stop[s] * dt_h
        
        # 3. Sample the number of arrivals (incorporating measurement error/stochasticity)
        # We generate a Normal sample, truncate it at 0, and round it to an integer.
        # This simulates a Poisson process with observation noise.
        arrivals = RNG.normal(loc=mu_window, scale=sigma_window)
        
        # Ensure the number of arrivals is not negative
        arrivals_window[s] = max(0, int(round(arrivals)))

        # Accumulate arrivals (which now have noise) for lambda_obs
        total_arrivals_per_stop[s] += arrivals_window[s]
        # ------------------------------------------------------------------

    # 6.2 Calculate available capacity and service per stop (NO CHANGES)
    mu_line = capacity_per_line(f)  # pax/h per line

    service_per_stop = {s: 0.0 for s in stops}
    for ln in lines:
        stops_ln = line_to_stops[ln]
        if not stops_ln:
            continue
        cap_window = mu_line[ln] * dt_h  # pax in this window
        per_stop = cap_window / len(stops_ln)
        for s in stops_ln:
            service_per_stop[s] += per_stop

    # Accumulate service per stop
    for s in stops:
        total_service_per_stop[s] += service_per_stop[s]
    # ------------------------------------

    # 6.3 Update queues (Q) at stops (NO CHANGES)
    Q_prev = Q.copy()
    for s in stops:
        Q[s] = max(0.0, Q[s] + arrivals_window[s] - service_per_stop[s])

        # Integral of the queue over time (Q * dt_h)
        cum_Q_time_per_stop[s] += Q[s] * dt_h
    # ------------------------------------

    # 6.4 Update lambda_hat based on queue increase (NO CHANGES)
    for s in stops:
        delta_Q = max(0.0, Q[s] - Q_prev[s])
        lambda_obs_inst = delta_Q / dt_h  # pax/h
        lambda_hat[s] = ALPHA * lambda_hat[s] + (1 - ALPHA) * lambda_obs_inst
    # ------------------------------------

    # 6.5 Line aggregates using λ_hat (NO CHANGES)
    Lambda_line = aggregate_lambda_per_line(lambda_hat)
    mu_line = capacity_per_line(f)
    # ------------------------------------

    # 6.6 KPI Calculation (wait time, utilization) (NO CHANGES)
    total_Lambda = sum(Lambda_line.values()) + 1e-9

    # Average demand-weighted wait time (h) ~ sum Λ/(2f) / sum Λ
    wait_h = (
        sum(Lambda_line[ln] / (2 * max(f[ln], 1e-6)) for ln in lines) / total_Lambda
    )
    wait_min = wait_h * 60.0

    # Average utilization
    utilizations = [Lambda_line[ln] / max(mu_line[ln], 1e-6) for ln in lines]
    avg_util = float(np.mean(utilizations))
    # ------------------------------------

    # 6.7 OPTIMIZATION BY DISTRICT (The main modification)
    f_new = f.copy()
    
    # Iterate through each line pool (each district)
    for district, lines_in_district in district_to_lines.items():
        if not lines_in_district:
            continue
            
        # Optimize only for the lines in this district
        f_opt_subconjunto, info_opt = optimizar_frecuencias_pulp_subconjunto(
            lines_to_optimize=lines_in_district,
            Lambda_line=Lambda_line,
            cycle_time_h=cycle_time_h,
            prior_freq=prior_freq,
            capacity_per_bus=CAPACITY_PER_BUS,
            F_min=F_MIN,
            F_max=F_MAX,
            f_current=f,
            w_over=10.0,
            w_dev_prior=1.0,
            w_dev_step=0.5,
            max_dev_prior_rel=0.50,
            max_change_step_rel=0.30,
            # 💡 New limit parameters
            max_adjusted_lines=MAX_LINES_ADJUSTED_PER_DISTRICT,
            w_adjustment_limit=WEIGHT_LINE_ADJUSTMENT,
        )
        
        f_new.update(f_opt_subconjunto)
        
        # Log the adjustment made
        if info_opt["status"] != "Optimal":
            print(f"[step={step}] PuLP status (District {district}) = {info_opt['status']}")
        else:
             print(f"[step={step}] District {district} - Lines adjusted: {info_opt['lines_adjusted_count']}")

    # Update frequencies with the optimal ones from all districts
    f = f_new

    # Recalculate capacity after reassignment
    mu_line = capacity_per_line(f)

    # 6.8 (DEBUG CODE) - NO CHANGES
    # -----------------------------
    total_Q = sum(Q.values())

    # aggregates by district
    Q_por_distrito = {}
    for s in stops:
        d = stop_to_distrito.get(s)
        if d is None:
            continue
        Q_por_distrito.setdefault(d, 0.0)
        Q_por_distrito[d] += Q[s]

    # build a debug row
    debug_row = {
        "step": step,
        "minute": minute,
        "total_Q": total_Q,
    }

    # add Q and lambda_hat from some key districts
    for d in ["01", "03", "08", "13"]:
        debug_row[f"Q_distrito_{d}"] = Q_por_distrito.get(d, 0.0)

    # info per stop for some example stops
    DEBUG_STOPS = stops[:5]
    for s in DEBUG_STOPS:
        debug_row[f"Q_{s}"] = Q.get(s, float("nan"))
        debug_row[f"arrivals_{s}"] = arrivals_window.get(s, float("nan"))
        debug_row[f"service_{s}"] = service_per_stop.get(s, float("nan"))
        debug_row[f"lambda_true_{s}"] = lambda_true.get(s, float("nan"))
        debug_row[f"lambda_hat_{s}"] = lambda_hat.get(s, float("nan"))

    debug_history.append(debug_row)
    # -----------------------------

    # 6.9 Save to history (NO CHANGES)
    row = {
        "step": step,
        "minute": minute,
        "wait_min": wait_min,
        "avg_util": avg_util,
        "total_Lambda": total_Lambda,
    }
    # Also save frequencies and utilizations per line (for later analysis)
    for ln in lines:
        row[f"f_{ln}"] = f[ln]
        row[f"u_{ln}"] = Lambda_line[ln] / max(mu_line[ln], 1e-6)

    history.append(row)

    # Log every so often
    if step % 3 == 0 or step == STEPS - 1:
        print(
            f"[t={minute:3d} min] avg wait={wait_min:5.2f} min, "
            f"avg utilization={avg_util:4.2f}"
        )


# -----------------------------
# 6.bis. Calculate stop metrics for run_stop_stats (NO CHANGES)
# -----------------------------

T_total_h = TIME_HORIZON_MIN / 60.0  # total duration in hours

lambda_obs_per_stop = {}
wait_time_min_per_stop = {}
crowding_index_per_stop = {}

EPS = 1e-6

for s in stops:
    # 1) Observed arrival rate (pax/h)
    lam_obs = total_arrivals_per_stop[s] / max(T_total_h, EPS)
    lambda_obs_per_stop[s] = lam_obs

    # 2) Average number in queue L(s)
    L_avg = cum_Q_time_per_stop[s] / max(T_total_h, EPS)

    if lam_obs > EPS:
        W_h = L_avg / lam_obs  # hours
        wait_time_min_per_stop[s] = W_h * 60.0
    else:
        wait_time_min_per_stop[s] = 0.0

    # 3) Average capacity offered at the stop (pax/h)
    mu_obs = total_service_per_stop[s] / max(T_total_h, EPS)

    if mu_obs > EPS:
        crowding_index_per_stop[s] = lam_obs / mu_obs
    else:
        crowding_index_per_stop[s] = 0.0


# -----------------------------
# 7. Save Results to CSV (NO CHANGES)
# -----------------------------

ts = datetime.now().strftime("%Y%m%d_%H%M%S")

df_hist = pd.DataFrame(history)
out_dir = Path("salida_red_simulada")
out_dir.mkdir(exist_ok=True)

out_csv = out_dir / f"intel_demo_history_{ts}.csv"
df_hist.to_csv(out_csv, index=False)

# Also save queue and camera debug data
df_debug = pd.DataFrame(debug_history)
out_debug_csv = out_dir / f"intel_debug_queues_{ts}.csv"
df_debug.to_csv(out_debug_csv, index=False)

print(f"Queue and camera debug saved to: {out_debug_csv}")
print("\nSimulation finished.")
print(f"History saved to: {out_csv}")


# -----------------------------
# 8. Prepare Frequency Data for Supabase (NO CHANGES)
# -----------------------------

final_row = df_hist.iloc[-1]
initial_f = prior_freq
final_f = {ln: final_row[f"f_{ln}"] for ln in lines}

# Create a dataframe with lines and relative frequency change
df_freq = pd.DataFrame(
    {
        "line": lines,
        "f_initial": [initial_f[ln] for ln in lines],
        "f_final": [final_f[ln] for ln in lines],
    }
)
df_freq["delta_abs"] = df_freq["f_final"] - df_freq["f_initial"]
df_freq["delta_rel_pct"] = 100 * df_freq["delta_abs"] / df_freq["f_initial"]

print("\nTop 10 lines by relative frequency increase:")
print(
    df_freq.sort_values("delta_rel_pct", ascending=False)
    .head(10)
    .to_string(
        index=False,
        formatters={
            "f_initial": "{:.2f}".format,
            "f_final": "{:.2f}".format,
            "delta_abs": "{:+.2f}".format,
            "delta_rel_pct": "{:+.1f}%".format,
        },
    )
)

print("\nTop 10 lines by relative frequency reduction:")
print(
    df_freq.sort_values("delta_rel_pct", ascending=True)
    .head(10)
    .to_string(
        index=False,
        formatters={
            "f_initial": "{:.2f}".format,
            "f_final": "{:.2f}".format,
            "delta_abs": "{:+.2f}".format,
            "delta_rel_pct": "{:+.1f}%".format,
        },
    )
)

# 8.1 Frequencies before / after
freq_before = {ln: float(initial_f[ln]) for ln in lines}
freq_after = {ln: float(final_f[ln]) for ln in lines}

# 8.2 Buses before / after based on line cycle time (cycle_time_h in hours)
buses_before = {}
buses_after = {}

for ln in lines:
    C = float(cycle_time_h[ln])             # hours per round trip
    fb = freq_before[ln]                    # veh/h before
    fa = freq_after[ln]                     # veh/h after
    buses_before[ln] = int(round(C * fb))
    buses_after[ln] = int(round(C * fa))


# -----------------------------
# 9. Save Run to Supabase (NO CHANGES)
# -----------------------------

try:
    scenario = "Simulación demo (Reasignación por Distrito)"
    description = (
        f"DT={DT_MIN} min, horizon={TIME_HORIZON_MIN} min, "
        f"TOTAL_DEMAND_TARGET_H={TOTAL_DEMAND_TARGET_H}"
    )

    run_id = create_run(scenario, description)

    # 1) Frequencies per line
    insert_run_line_freq(
        run_id,
        freq_before=freq_before,
        freq_after=freq_after,
        buses_before=buses_before,
        buses_after=buses_after,
    )
    print(f"\nRun {run_id} saved to Supabase (line frequencies).")

    # 2) Stop statistics (for v_hotspots)
    insert_run_stop_stats(
        run_id,
        lambda_obs=lambda_obs_per_stop,
        wait_time_min=wait_time_min_per_stop,
        crowding_index=crowding_index_per_stop,
    )
    print(f"Stop statistics for run {run_id} saved to run_stop_stats.")

except Exception as e:
    print("\n Error while saving run to Supabase:", e)