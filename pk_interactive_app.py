"""
pk_interactive_app.py

Final version (simplified):
- Real Theophylline data calibration (S, ka, ke, V, half-life)
- One-compartment oral absorption + elimination ODE model
- RK4 solver with fixed high-accuracy time step (dt = 0.02 h)
- User chooses:
    - Dose per administration
    - Dose interval
    - Number of doses
    - Total simulation time
- Computes:
    - Concentration curve
    - Time in therapeutic window (MEC-MTC)
- Recommendation engine:
    - Searches over candidate fixed intervals
    - Suggests best fixed-interval schedule (interval + #doses)
    - Gives explicit dose times and slider settings.
"""

import io
import math
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import streamlit as st

# ---------------- CONFIG ----------------
CSV_URL = "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/datasets/Theoph.csv"
OUTPUT_PLOT = "pk_simulation.png"

# Clinical bounds for theophylline (fixed)
MEC = 10.0  # mg/L (Minimum Effective Concentration)
MTC = 20.0  # mg/L (Maximum Tolerated Concentration)

# Fixed high-accuracy time step (in hours)
DEFAULT_DT = 0.02  # 0.02 h ≈ 1.2 minutes

# ---------------- HELPERS / MODEL FIT ----------------

def download_theoph_csv():
    """Download and preprocess Theophylline dataset."""
    r = requests.get(CSV_URL)
    r.raise_for_status()
    text = r.content.decode("utf-8")
    df = pd.read_csv(io.StringIO(text))
    df = df.rename(columns={c: c.lower() for c in df.columns})
    keep = [c for c in ["subject", "time", "conc", "dose", "wt"] if c in df.columns]
    return df[keep].copy()


def mean_concentration_by_time(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean concentration vs time across all subjects."""
    grouped = df.groupby("time")["conc"].agg(["mean", "std", "count"]).reset_index()
    grouped = grouped.rename(columns={"mean": "conc_mean"})
    return grouped


def one_comp_oral_model(t, S, ka, ke):
    """
    Analytical concentration-time model for a single oral dose
    in a one-compartment model with first-order absorption & elimination.

        C(t) = S * (ka / (ka - ke)) * (exp(-ke t) - exp(-ka t))

    S is a scale factor ∝ F * Dose / V.
    """
    t = np.asarray(t)
    eps = 1e-8
    if abs(ka - ke) < eps:
        ka = ke + eps
    return S * (ka / (ka - ke)) * (np.exp(-ke * t) - np.exp(-ka * t))


@st.cache_data(show_spinner=True)
def calibrate_model_from_data():
    """
    Fit S, ka, ke from the Theoph dataset and derive V, half-life.
    Returns:
        df_mean: mean concentration vs time
        params: dict with keys S, ka, ke, V, half_life
        dose_ref: mean dose in dataset (mg)
    """
    df = download_theoph_csv()
    df_mean = mean_concentration_by_time(df)

    t_data = df_mean["time"].values
    c_data = df_mean["conc_mean"].values

    mask = c_data > 0
    t_fit = t_data[mask]
    c_fit = c_data[mask]

    # Initial guesses
    S0 = float(c_fit.max() if len(c_fit) else 1.0)
    ka0 = 1.0
    ke0 = 0.1
    p0 = [S0, ka0, ke0]
    bounds = ([1e-6, 1e-3, 1e-3], [200.0, 5.0, 1.0])

    popt, _ = curve_fit(
        one_comp_oral_model,
        t_fit,
        c_fit,
        p0=p0,
        bounds=bounds,
        maxfev=20000,
    )
    S_fit, ka_fit, ke_fit = popt

    dose_ref = float(df["dose"].mean()) if "dose" in df.columns else 320.0
    V_est = dose_ref / S_fit if S_fit != 0 else 1.0
    half_life = math.log(2) / ke_fit if ke_fit > 0 else float("nan")

    params = {
        "S": float(S_fit),
        "ka": float(ka_fit),
        "ke": float(ke_fit),
        "V": float(V_est),
        "half_life": float(half_life),
    }
    return df_mean, params, float(dose_ref)


# ---------------- NUMERICAL INTEGRATOR (RK4) ----------------

def rk4_step(state, t, dt, ka, ke):
    """
    Perform one RK4 step for state = [A_g, A_c] at time t with step dt.
    """
    def deriv(y):
        A_g, A_c = y
        dA_g = -ka * A_g
        dA_c = ka * A_g - ke * A_c
        return np.array([dA_g, dA_c], dtype=float)

    k1 = deriv(state)
    k2 = deriv(state + 0.5 * dt * k1)
    k3 = deriv(state + 0.5 * dt * k2)
    k4 = deriv(state + dt * k3)
    new_state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return new_state


def simulate_multi_dose(ka, ke, V, dose_mg, tau_h, n_doses, t_end, dt=DEFAULT_DT):
    """
    Simulate multi-dose PK with fixed dosing schedule using RK4.
    Doses at times: 0, tau, 2*tau, ..., up to n_doses-1.

    Returns:
        t_values: array of time points
        C_values: array of concentrations (mg/L)
    """
    dose_times = [i * tau_h for i in range(n_doses)]
    dose_times = [t for t in dose_times if t <= t_end + 1e-9]

    # Initial state
    A_g = dose_mg if dose_times and abs(dose_times[0]) < 1e-9 else 0.0
    A_c = 0.0
    state = np.array([A_g, A_c], dtype=float)

    t_values = [0.0]
    A_g_values = [state[0]]
    A_c_values = [state[1]]

    applied = set()
    if dose_times and abs(dose_times[0]) < 1e-9:
        applied.add(0)

    t = 0.0
    n_steps = int(math.ceil(t_end / dt))

    for step in range(1, n_steps + 1):
        t_next = step * dt

        # Apply any doses that should occur by t_next
        for i, td in enumerate(dose_times):
            if i in applied:
                continue
            if td <= t_next + 1e-12:
                # Fraction of dt needed to reach td from t
                small_dt = td - t
                if small_dt > 1e-12:
                    state = rk4_step(state, t, small_dt, ka, ke)
                    t = td
                    t_values.append(t)
                    A_g_values.append(state[0])
                    A_c_values.append(state[1])
                # Apply dose
                state[0] += dose_mg
                applied.add(i)

        # Advance to t_next if needed
        if t_next > t + 1e-12:
            state = rk4_step(state, t, dt, ka, ke)
            t = t_next
            t_values.append(t)
            A_g_values.append(state[0])
            A_c_values.append(state[1])

    # Final correction to exactly t_end
    if t < t_end - 1e-9:
        final_dt = t_end - t
        state = rk4_step(state, t, final_dt, ka, ke)
        t = t_end
        t_values.append(t)
        A_g_values.append(state[0])
        A_c_values.append(state[1])

    C_values = np.array(A_c_values) / V
    return np.array(t_values), C_values


# ---------------- TIME-IN-WINDOW CALCS ----------------

def compute_time_above_threshold(t, C, threshold):
    """
    Compute total time where C(t) >= threshold using linear interpolation.
    """
    t = np.asarray(t)
    C = np.asarray(C)
    total = 0.0
    if len(t) < 2:
        return 0.0
    for i in range(len(t) - 1):
        t0, t1 = t[i], t[i+1]
        c0, c1 = C[i], C[i+1]
        dt = t1 - t0
        if dt <= 0:
            continue
        if c0 >= threshold and c1 >= threshold:
            total += dt
        elif (c0 - threshold) * (c1 - threshold) < 0:
            # crossing
            if (c1 - c0) != 0:
                frac = (threshold - c0) / (c1 - c0)
            else:
                frac = 0.0
            frac = max(0.0, min(1.0, frac))
            if c0 >= threshold:
                total += frac * dt
            else:
                total += (1.0 - frac) * dt
    return float(total)


def compute_time_in_window(t, C, low, high):
    """
    Time where low <= C(t) <= high.
    We compute:
        time_ge_low - time_ge_high
    """
    t_ge_low = compute_time_above_threshold(t, C, low)
    t_ge_high = compute_time_above_threshold(t, C, high)
    return max(0.0, t_ge_low - t_ge_high)


# ---------------- STREAMLIT APP ----------------

def main():
    st.set_page_config(
        page_title="PK Simulator (Fixed-interval + Recommendation)",
        layout="wide"
    )
    st.title("Pharmacokinetic Simulator — Data-Calibrated Fixed-Interval Model")

    st.markdown(
        """
        This simulator uses real **Theophylline** data to calibrate a one-compartment
        oral absorption model, then simulates multi-dose regimens with a **fixed interval**.

        - You choose: dose, interval, number of doses, total time.  
        - The model solves the ODE using RK4 and plots the concentration-time curve.  
        - A recommendation engine searches over intervals and suggests a safer and
          more effective fixed-interval schedule within the therapeutic window.
        """
    )

    with st.spinner("Calibrating model from the Theophylline dataset..."):
        df_mean, params, dose_ref = calibrate_model_from_data()

    S_fit = params["S"]
    ka_fit = params["ka"]
    ke_fit = params["ke"]
    V_est = params["V"]
    half_life = params["half_life"]

    # Sidebar controls
    st.sidebar.header("Simulation controls")

    dose_input = st.sidebar.number_input(
        "Dose per administration (mg)",
        min_value=1.0,
        max_value=600.0,
        value=float(round(dose_ref, 1)),
        step=1.0,
    )

    total_time = st.sidebar.slider(
        "Total simulation time (hours)",
        min_value=12.0,
        max_value=120.0,
        value=48.0,
        step=4.0,
    )

    interval = st.sidebar.slider(
        "Dose interval (hours)",
        min_value=2.0,
        max_value=24.0,
        value=8.0,
        step=0.5,
    )

    n_doses = st.sidebar.slider(
        "Number of doses",
        min_value=1,
        max_value=20,
        value=3,
        step=1,
    )

    st.sidebar.markdown(
        f"**Therapeutic window (fixed):** MEC = {MEC} mg/L, MTC = {MTC} mg/L"
    )

    # Main simulation (user-defined fixed schedule)
    t_sim, C_sim = simulate_multi_dose(
        ka_fit,
        ke_fit,
        V_est,
        dose_input,
        interval,
        n_doses,
        total_time,
        dt=DEFAULT_DT,
    )
    used_schedule = [i * interval for i in range(n_doses)]

    # Single-dose calibration curve
    t_single = np.linspace(0.0, total_time, 500)
    C_single = one_comp_oral_model(t_single, S_fit, ka_fit, ke_fit)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t_sim, C_sim, label="Simulated regimen (your settings)", linewidth=2)
    ax.plot(t_single, C_single, "--", label="Single-dose fit (calibrated)")
    ax.scatter(
        df_mean["time"],
        df_mean["conc_mean"],
        color="black",
        s=20,
        label="Real data (mean)",
        zorder=5,
    )

    for td in used_schedule:
        ax.axvline(td, color="gray", linestyle=":", alpha=0.5)

    ax.axhline(MEC, color="green", linestyle=":", label=f"MEC = {MEC} mg/L")
    ax.axhline(MTC, color="red", linestyle=":", label=f"MTC = {MTC} mg/L")

    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Concentration (mg/L)")
    ax.set_title("Drug Concentration vs Time")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)
    fig.savefig(OUTPUT_PLOT, dpi=150)
    st.info(f"Plot saved as **{OUTPUT_PLOT}** in the working directory.")

    # Key metrics for the current regimen
    if len(C_sim) > 0:
        Cmax = float(np.max(C_sim))
        Tmax = float(t_sim[np.argmax(C_sim)])
        Cmin = float(np.min(C_sim))
    else:
        Cmax = Tmax = Cmin = 0.0

    time_ge_MEC = compute_time_above_threshold(t_sim, C_sim, MEC)
    time_ge_MTC = compute_time_above_threshold(t_sim, C_sim, MTC)
    time_in_window = compute_time_in_window(t_sim, C_sim, MEC, MTC)

    st.header("Key Predictions for Your Current Regimen")
    st.write(f"- Maximum concentration: **{Cmax:.2f} mg/L** at **{Tmax:.2f} h**")
    st.write(f"- Minimum (trough) concentration: **{Cmin:.2f} mg/L**")
    st.write(f"- Therapeutic range: **{MEC:.0f}–{MTC:.0f} mg/L**")
    st.write(f"- Time with C(t) ≥ MEC: **{time_ge_MEC:.2f} hours**")
    st.write(f"- Time with C(t) ≥ MTC (potential toxicity): **{time_ge_MTC:.2f} hours**")
    st.write(
        f"- Time with {MEC:.0f} ≤ C(t) ≤ {MTC:.0f} mg/L (therapeutic window): "
        f"**{time_in_window:.2f} hours**"
    )

    # ---------------- RECOMMENDATION ENGINE ----------------
    st.header("Recommended Fixed-Interval Strategy")

    # Search best fixed-interval schedule over a grid of intervals
    intervals_to_try = np.arange(3.0, 13.0, 0.5)  # 3h to 12.5h in 0.5h steps
    best_fixed = None

    for tau in intervals_to_try:
        n_try = int(total_time // tau) + 1
        if n_try < 1:
            continue

        t_try, C_try = simulate_multi_dose(
            ka_fit,
            ke_fit,
            V_est,
            dose_input,
            float(tau),
            n_try,
            total_time,
            dt=DEFAULT_DT,
        )

        Cmax_try = float(np.max(C_try))
        if Cmax_try > MTC:
            continue  # toxic, reject

        time_window_try = compute_time_in_window(t_try, C_try, MEC, MTC)
        if best_fixed is None or time_window_try > best_fixed["time_window"]:
            best_fixed = {
                "tau": float(tau),
                "n": n_try,
                "Cmax": Cmax_try,
                "Cmin": float(np.min(C_try)),
                "time_window": time_window_try,
            }

    if best_fixed is None:
        st.error(
            "No safe fixed-interval schedule found in the scanned range "
            "(3–12.5 hours). Try lowering the dose."
        )
        return

    bf = best_fixed
    schedule_times = [i * bf["tau"] for i in range(bf["n"])]

    st.success("Recommended strategy: **Optimized fixed-interval schedule**")
    st.write(f"- Dose per administration: **{dose_input:.1f} mg**")
    st.write(f"- Interval: **every {bf['tau']:.1f} hours**")
    st.write(f"- Number of doses: **{bf['n']}**")
    st.write(
        "- Dose times (hours from t = 0): "
        + ", ".join(f"{t:.1f}" for t in schedule_times)
    )
    st.write(f"- Predicted peak concentration (Cmax): **{bf['Cmax']:.2f} mg/L**")
    st.write(
        f"- Time in therapeutic window ({MEC:.0f}–{MTC:.0f} mg/L): "
        f"**{bf['time_window']:.2f} hours**"
    )

    st.info(
        f"To follow this plan in the app, set:\n"
        f"- `Dose interval (hours)` → **{bf['tau']:.1f}**\n"
        f"- `Number of doses` → **{bf['n']}**\n"
        "and keep the same dose and total simulation time."
    )

if __name__ == "__main__":
    main()