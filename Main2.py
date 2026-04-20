from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

# Input files are expected in the same folder as this script.
OBS_FILENAME = "my_data_2.obs"
NAV_FILENAME = "samsung_nav.nav.rnx"

# Output files are written to the same folder as this script.
RAW_CSV_FILENAME = "gnss_solution_raw_weighted2.csv"
CLEAN_CSV_FILENAME = "gnss_solution_clean_weighted2.csv"
CLEAN_KML_FILENAME = "gnss_solution_clean_weighted2.kml"

SCRIPT_DIR = Path(__file__).resolve().parent
OBS_PATH = SCRIPT_DIR / OBS_FILENAME
NAV_PATH = SCRIPT_DIR / NAV_FILENAME
RAW_CSV_PATH = SCRIPT_DIR / RAW_CSV_FILENAME
CLEAN_CSV_PATH = SCRIPT_DIR / CLEAN_CSV_FILENAME
CLEAN_KML_PATH = SCRIPT_DIR / CLEAN_KML_FILENAME

# Physical constants.
C = 299792458.0
MU = 3.986005e14
OMEGA_E_DOT = 7.2921151467e-5
F_REL = -4.442807633e-10

# WGS84 constants.
WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_B = WGS84_A * (1.0 - WGS84_F)
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)
WGS84_EP2 = (WGS84_A * WGS84_A - WGS84_B * WGS84_B) / (WGS84_B * WGS84_B)

# RINEX float pattern.
FLOAT_RE = re.compile(r"[+-]?(?:\d+\.\d*|\d*\.\d+|\d+)(?:[EeDd][+-]?\d+)?")


@dataclass
class BroadcastEphemeris:
    system: str
    sv: str
    toc: datetime
    af0: float
    af1: float
    af2: float
    iode: float
    crs: float
    delta_n: float
    m0: float
    cuc: float
    e: float
    cus: float
    sqrt_a: float
    toe: float
    cic: float
    omega0: float
    cis: float
    i0: float
    crc: float
    omega: float
    omega_dot: float
    idot: float
    week: int
    tgd: float


def _parse_float_fields(line: str) -> List[float]:
    """Extract all numeric fields from one RINEX line."""
    values: List[float] = []
    for token in FLOAT_RE.findall(line.replace("D", "E")):
        try:
            values.append(float(token))
        except ValueError:
            pass
    return values


def parse_nav_rinex(nav_path: str | Path) -> Dict[str, List[BroadcastEphemeris]]:
    """Parse broadcast ephemeris from a navigation RINEX file."""
    nav_path = Path(nav_path)
    eph: Dict[str, List[BroadcastEphemeris]] = {}

    with nav_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "END OF HEADER" in line:
                break

        while True:
            line1 = f.readline()
            if not line1:
                break
            if not line1.strip():
                continue

            sv = line1[:3].strip()
            system = sv[:1]
            rest = [f.readline() for _ in range(7)]
            if any(r == "" for r in rest):
                break

            # Keep systems that share the GPS-like Kepler broadcast model.
            if system not in ("G", "E", "J"):
                continue

            nums1 = _parse_float_fields(line1[3:])
            if len(nums1) < 9:
                continue

            year, month, day, hour, minute = map(int, nums1[:5])
            sec = nums1[5]
            sec_int = int(sec)
            micro = int(round((sec - sec_int) * 1e6))
            toc = datetime(year, month, day, hour, minute, sec_int, micro, tzinfo=timezone.utc)
            af0, af1, af2 = nums1[6:9]

            fields = [_parse_float_fields(r) for r in rest]
            flat = [x for row in fields for x in row]
            if len(flat) < 23:
                continue

            rec = BroadcastEphemeris(
                system=system,
                sv=sv,
                toc=toc,
                af0=af0,
                af1=af1,
                af2=af2,
                iode=flat[0],
                crs=flat[1],
                delta_n=flat[2],
                m0=flat[3],
                cuc=flat[4],
                e=flat[5],
                cus=flat[6],
                sqrt_a=flat[7],
                toe=flat[8],
                cic=flat[9],
                omega0=flat[10],
                cis=flat[11],
                i0=flat[12],
                crc=flat[13],
                omega=flat[14],
                omega_dot=flat[15],
                idot=flat[16],
                week=int(round(flat[18])) if len(flat) > 18 else 0,
                tgd=flat[22] if len(flat) > 22 else 0.0,
            )
            eph.setdefault(sv, []).append(rec)

    for sv in eph:
        eph[sv].sort(key=lambda item: item.toc)
    return eph


def _parse_obs_field(field: str):
    # RINEX observation field: 16 chars = value(14) + LLI(1) + SSI(1)
    raw = field[:14].strip()
    if not raw:
        return None
    try:
        return float(raw.replace("D", "E"))
    except ValueError:
        return None


def parse_obs_rinex(obs_path: Path) -> list[dict]:
    # Reads RINEX OBS file and returns epochs with per-satellite observations.
    lines = Path(obs_path).read_text(encoding="utf-8", errors="ignore").splitlines()

    obs_types_by_sys = {}
    i = 0

    # -----------------------------
    # Parse header
    # -----------------------------
    while i < len(lines):
        line = lines[i]

        if "SYS / # / OBS TYPES" in line:
            sys_id = line[0]
            try:
                total_types = int(line[3:6].strip())
            except ValueError:
                total_types = 0

            obs_types = []
            # First header line contributes up to 13 observation types
            for start in range(7, 60, 4):
                token = line[start:start + 3].strip()
                if token:
                    obs_types.append(token)

            # Continuation header lines if needed
            while len(obs_types) < total_types:
                i += 1
                cont = lines[i]
                for start in range(7, 60, 4):
                    token = cont[start:start + 3].strip()
                    if token:
                        obs_types.append(token)

            obs_types_by_sys[sys_id] = obs_types[:total_types]

        elif "END OF HEADER" in line:
            i += 1
            break

        i += 1

    epochs = []

    # -----------------------------
    # Parse body
    # -----------------------------
    while i < len(lines):
        line = lines[i]

        if not line:
            i += 1
            continue

        # Epoch line
        if line.startswith(">"):
            try:
                year = int(line[2:6])
                month = int(line[7:9])
                day = int(line[10:12])
                hour = int(line[13:15])
                minute = int(line[16:18])
                sec = float(line[19:29])
                flag = int(line[30:32].strip() or 0)
                num_sats = int(line[32:35].strip() or 0)
            except ValueError:
                i += 1
                continue

            sec_int = int(sec)
            micro = int(round((sec - sec_int) * 1e6))

            epoch_time = pd.Timestamp(
                year=year,
                month=month,
                day=day,
                hour=hour,
                minute=minute,
                second=sec_int,
                microsecond=micro,
                tz="UTC",
            )

            # Skip non-regular epochs if needed
            if flag not in (0, 1):
                i += 1
                for _ in range(num_sats):
                    if i < len(lines):
                        i += 1
                continue

            i += 1
            obs_map = {}

            for _ in range(num_sats):
                if i >= len(lines):
                    break

                sat_line = lines[i]
                if len(sat_line) < 3:
                    i += 1
                    continue

                sv = sat_line[:3].strip()
                if len(sv) < 2:
                    i += 1
                    continue

                sys_id = sv[0]
                obs_types = obs_types_by_sys.get(sys_id, [])

                # Keep only systems you want
                if sys_id not in {"G", "E"}:
                    i += 1
                    continue

                # Number of observation lines for this satellite:
                # first line has cols 4..80, continuation lines same 16-char fields
                n_obs = len(obs_types)
                n_lines = max(1, math.ceil(n_obs / 5))

                sat_text = sat_line[3:]
                i += 1

                for _cont in range(1, n_lines):
                    if i >= len(lines):
                        break
                    sat_text += lines[i]
                    i += 1

                values = {}
                for k, obs_type in enumerate(obs_types):
                    start = k * 16
                    end = start + 16
                    field = sat_text[start:end]
                    values[obs_type] = _parse_obs_field(field)

                obs_map[sv] = values

            epochs.append({
                "time": epoch_time,
                "obs": obs_map,
            })

        else:
            i += 1

    return epochs

def gps_week_seconds(dt: datetime) -> Tuple[int, float]:
    """Convert UTC datetime to GPS week and seconds-of-week."""
    gps0 = datetime(1980, 1, 6, tzinfo=timezone.utc)
    delta = (dt - gps0).total_seconds()
    week = int(delta // 604800)
    sow = delta - week * 604800
    return week, sow


def wrap_gps_time(seconds: float) -> float:
    """Wrap time difference into the standard GPS half-week interval."""
    while seconds > 302400.0:
        seconds -= 604800.0
    while seconds < -302400.0:
        seconds += 604800.0
    return seconds


def solve_kepler(mk: float, e: float, tol: float = 1e-12, max_iter: int = 50) -> float:
    """Solve Kepler's equation with Newton iterations."""
    ek = mk
    for _ in range(max_iter):
        denom = 1.0 - e * math.cos(ek)
        if abs(denom) < 1e-14:
            break
        next_ek = ek - (ek - e * math.sin(ek) - mk) / denom
        if abs(next_ek - ek) < tol:
            return next_ek
        ek = next_ek
    return ek


def closest_ephemeris(eph_list: List[BroadcastEphemeris], t_rx: datetime) -> Optional[BroadcastEphemeris]:
    """Pick the navigation record closest in time to the epoch."""
    if not eph_list:
        return None
    return min(eph_list, key=lambda item: abs((t_rx - item.toc).total_seconds()))


def choose_best_pseudorange(system: str, obs_map: Dict[str, Optional[float]]) -> Optional[float]:
    """Select the best available pseudorange observable for a satellite."""
    candidate_map = {
        "G": ("C1C", "C1W", "C1X", "C1P", "C2W", "C2X", "C5Q", "C5X"),
        "E": ("C1C", "C1X", "C5Q", "C5X", "C7Q", "C7X", "C8Q", "C8X"),
        "J": ("C1C", "C1X", "C2L", "C2X", "C5Q", "C5X"),
    }
    for code in candidate_map.get(system, ("C1C",)):
        val = obs_map.get(code)
        if val is not None and val > 1e6:
            return val
    return None


def satellite_clock_bias(ep: BroadcastEphemeris, tx_time: datetime) -> float:
    """Compute the satellite clock correction in seconds."""
    _, sow_tx = gps_week_seconds(tx_time)
    tk = wrap_gps_time(sow_tx - ep.toe)
    a = ep.sqrt_a * ep.sqrt_a
    n0 = math.sqrt(MU / (a ** 3))
    n = n0 + ep.delta_n
    mk = ep.m0 + n * tk
    ek = solve_kepler(mk, ep.e)
    dtr = F_REL * ep.e * ep.sqrt_a * math.sin(ek)
    dt = (tx_time - ep.toc).total_seconds()
    return ep.af0 + ep.af1 * dt + ep.af2 * dt * dt + dtr - ep.tgd


def satellite_position_ecef(ep: BroadcastEphemeris, tx_time: datetime) -> np.ndarray:
    """Compute the broadcast satellite ECEF position at transmit time."""
    _, sow_tx = gps_week_seconds(tx_time)
    tk = wrap_gps_time(sow_tx - ep.toe)
    a = ep.sqrt_a * ep.sqrt_a
    n0 = math.sqrt(MU / (a ** 3))
    n = n0 + ep.delta_n
    mk = ep.m0 + n * tk
    ek = solve_kepler(mk, ep.e)

    vk = math.atan2(math.sqrt(1.0 - ep.e * ep.e) * math.sin(ek), math.cos(ek) - ep.e)
    phik = vk + ep.omega
    duk = ep.cus * math.sin(2.0 * phik) + ep.cuc * math.cos(2.0 * phik)
    drk = ep.crs * math.sin(2.0 * phik) + ep.crc * math.cos(2.0 * phik)
    dik = ep.cis * math.sin(2.0 * phik) + ep.cic * math.cos(2.0 * phik)

    uk = phik + duk
    rk = a * (1.0 - ep.e * math.cos(ek)) + drk
    ik = ep.i0 + dik + ep.idot * tk

    xk_prime = rk * math.cos(uk)
    yk_prime = rk * math.sin(uk)
    omega_k = ep.omega0 + (ep.omega_dot - OMEGA_E_DOT) * tk - OMEGA_E_DOT * ep.toe

    xk = xk_prime * math.cos(omega_k) - yk_prime * math.cos(ik) * math.sin(omega_k)
    yk = xk_prime * math.sin(omega_k) + yk_prime * math.cos(ik) * math.cos(omega_k)
    zk = yk_prime * math.sin(ik)
    return np.array([xk, yk, zk], dtype=float)


def earth_rotation_correction(pos: np.ndarray, travel_time: float) -> np.ndarray:
    """Compensate the satellite position for Earth rotation during signal travel."""
    angle = OMEGA_E_DOT * travel_time
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    rot = np.array([[cos_a, sin_a, 0.0], [-sin_a, cos_a, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    return rot @ pos


def ecef_to_lla(x: float, y: float, z: float) -> Tuple[float, float, float]:
    """Convert ECEF coordinates to latitude, longitude, and altitude."""
    lon = math.atan2(y, x)
    p = math.hypot(x, y)
    th = math.atan2(WGS84_A * z, WGS84_B * p)
    lat = math.atan2(
        z + WGS84_EP2 * WGS84_B * math.sin(th) ** 3,
        p - WGS84_E2 * WGS84_A * math.cos(th) ** 3,
    )
    n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * math.sin(lat) ** 2)
    alt = p / math.cos(lat) - n
    return math.degrees(lat), math.degrees(lon), alt


def ecef_to_enu_matrix(lat_deg: float, lon_deg: float) -> np.ndarray:
    """Build the rotation matrix from ECEF to local ENU coordinates."""
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    sin_lon = math.sin(lon)
    cos_lon = math.cos(lon)
    return np.array(
        [
            [-sin_lon, cos_lon, 0.0],
            [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
            [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat],
        ],
        dtype=float,
    )


def satellite_elevation_deg(rx_pos_ecef: np.ndarray, sat_pos_ecef: np.ndarray) -> float:
    """Compute satellite elevation above the receiver horizon."""
    if np.linalg.norm(rx_pos_ecef) < 1.0:
        return 30.0
    lat_deg, lon_deg, _ = ecef_to_lla(rx_pos_ecef[0], rx_pos_ecef[1], rx_pos_ecef[2])
    enu = ecef_to_enu_matrix(lat_deg, lon_deg)
    los_ecef = sat_pos_ecef - rx_pos_ecef
    los_enu = enu @ los_ecef
    east, north, up = los_enu
    horiz = math.hypot(east, north)
    return math.degrees(math.atan2(up, horiz))


def elevation_weight(elev_deg: float) -> float:
    """Assign a larger weight to satellites higher above the horizon."""
    elev = max(elev_deg, 5.0)
    sin_el = math.sin(math.radians(elev))
    return max(0.05, sin_el * sin_el)


def robust_residual_weight(residual_m: float, scale_m: float = 15.0) -> float:
    """Reduce the impact of measurements with large residuals."""
    a = abs(residual_m)
    if a <= scale_m:
        return 1.0
    return scale_m / a


def _build_measurement_model(
    t_rx: datetime,
    sat_subset: List[Tuple[str, float, BroadcastEphemeris]],
    state: np.ndarray,
) -> List[dict]:
    """Build the geometry, residuals, and weights for one LS iteration."""
    rx_pos = state[:3]
    rows: List[dict] = []

    for sv, pr, ep in sat_subset:
        tx_guess = t_rx - timedelta(seconds=pr / C)
        dt_sv = satellite_clock_bias(ep, tx_guess)
        tx_time = t_rx - timedelta(seconds=(pr / C) - dt_sv)

        sat_pos = satellite_position_ecef(ep, tx_time)
        sat_pos = earth_rotation_correction(sat_pos, pr / C)

        rho_vec = rx_pos - sat_pos
        rho = np.linalg.norm(rho_vec)
        if rho < 1.0:
            continue

        pred = rho + state[3] - C * dt_sv
        residual = pr - pred
        h_row = np.array([
            (rx_pos[0] - sat_pos[0]) / rho,
            (rx_pos[1] - sat_pos[1]) / rho,
            (rx_pos[2] - sat_pos[2]) / rho,
            1.0,
        ], dtype=float)
        elev_deg = satellite_elevation_deg(rx_pos, sat_pos)
        base_weight = elevation_weight(elev_deg)

        rows.append(
            {
                "sv": sv,
                "pseudorange": pr,
                "sat_pos": sat_pos,
                "dt_sv": dt_sv,
                "elev_deg": elev_deg,
                "pred": pred,
                "residual": residual,
                "h_row": h_row,
                "base_weight": base_weight,
            }
        )

    return rows


def weighted_least_squares(
    t_rx: datetime,
    sat_subset: List[Tuple[str, float, BroadcastEphemeris]],
    x0: Optional[np.ndarray],
    max_iter: int = 10,
) -> Optional[Tuple[np.ndarray, List[dict]]]:
    """Solve receiver position with elevation-weighted iterative least squares."""
    state = np.array(x0 if x0 is not None else [0.0, 0.0, 0.0, 0.0], dtype=float)
    last_rows: List[dict] = []

    for _ in range(max_iter):
        rows = _build_measurement_model(t_rx, sat_subset, state)
        if len(rows) < 4:
            return None

        # Use equal weights only in the very first coarse step.
        if np.linalg.norm(state[:3]) < 1.0:
            weights = np.ones(len(rows), dtype=float)
        else:
            weights = np.array(
                [row["base_weight"] * robust_residual_weight(row["residual"]) for row in rows],
                dtype=float,
            )

        sqrt_w = np.sqrt(weights)
        H = np.vstack([row["h_row"] for row in rows])
        y = np.array([row["residual"] for row in rows], dtype=float)
        Hw = H * sqrt_w[:, None]
        yw = y * sqrt_w

        try:
            dx, *_ = np.linalg.lstsq(Hw, yw, rcond=None)
        except np.linalg.LinAlgError:
            return None

        state += dx
        last_rows = rows

        if np.linalg.norm(dx[:3]) < 1e-3 and abs(dx[3]) < 1e-3:
            break

    final_rows = _build_measurement_model(t_rx, sat_subset, state)
    if len(final_rows) < 4:
        return None
    return state, final_rows


def solve_epoch_position(
    epoch: dict,
    nav: Dict[str, List[BroadcastEphemeris]],
    x0: Optional[np.ndarray] = None,
) -> Optional[dict]:
    """Solve one epoch and prune only clearly bad measurements."""
    t_rx = epoch["time"]
    sat_candidates: List[Tuple[str, float, BroadcastEphemeris]] = []

    for sv, obs_map in epoch["obs"].items():
        pr = choose_best_pseudorange(sv[0], obs_map)
        if pr is None:
            continue
        ep = closest_ephemeris(nav.get(sv, []), t_rx)
        if ep is None:
            continue
        sat_candidates.append((sv, pr, ep))

    if len(sat_candidates) < 5:
        return None

    working = sat_candidates[:]
    state0 = None if x0 is None else x0.copy()
    best_solution = None

    # Keep as many satellites as possible, but remove obvious outliers if needed.
    while len(working) >= 5:
        result = weighted_least_squares(t_rx, working, state0)
        if result is None:
            return None

        state, rows = result
        residuals = np.array([row["residual"] for row in rows], dtype=float)
        abs_res = np.abs(residuals)
        rms = float(np.sqrt(np.mean(residuals * residuals)))
        worst_idx = int(np.argmax(abs_res))
        worst = float(abs_res[worst_idx])
        min_elev = float(min(row["elev_deg"] for row in rows))
        mean_elev = float(np.mean([row["elev_deg"] for row in rows]))

        best_solution = {
            "time": t_rx,
            "x": state[0],
            "y": state[1],
            "z": state[2],
            "clock_bias_m": state[3],
            "num_sats": len(working),
            "satellites": ",".join([sv for sv, _, _ in working]),
            "rms_residual_m": rms,
            "worst_residual_m": worst,
            "min_elevation_deg": min_elev,
            "mean_elevation_deg": mean_elev,
            "state_vec": state,
            "rows": rows,
        }

        # Accept the solution if residuals are already reasonable.
        if rms <= 35.0 and worst <= 80.0:
            return best_solution

        # Keep enough satellites and only remove the most suspicious one.
        if len(working) <= 5:
            return best_solution

        bad_sv = rows[worst_idx]["sv"]
        working = [item for item in working if item[0] != bad_sv]
        state0 = state

    return best_solution


def add_lla_and_velocity(df: pd.DataFrame) -> pd.DataFrame:
    """Add geographic coordinates and speed to the output table."""
    out = df.copy()
    lla = out.apply(lambda r: ecef_to_lla(r["x"], r["y"], r["z"]), axis=1)
    out[["lat_deg", "lon_deg", "alt_m"]] = pd.DataFrame(lla.tolist(), index=out.index)

    velocities = [np.nan]
    for i in range(1, len(out)):
        dt = (out.loc[i, "time"] - out.loc[i - 1, "time"]).total_seconds()
        d = math.sqrt(
            (out.loc[i, "x"] - out.loc[i - 1, "x"]) ** 2
            + (out.loc[i, "y"] - out.loc[i - 1, "y"]) ** 2
            + (out.loc[i, "z"] - out.loc[i - 1, "z"]) ** 2
        )
        velocities.append(d / dt if dt > 0 else np.nan)

    out["velocity_mps"] = velocities
    return out


def clean_track(df: pd.DataFrame) -> pd.DataFrame:
    """Filter outliers and smooth the visible track for KML export."""
    out = df.copy().sort_values("time").reset_index(drop=True)
    out["ecef_radius_m"] = np.sqrt(out["x"] ** 2 + out["y"] ** 2 + out["z"] ** 2)

    valid = (
        out["ecef_radius_m"].between(6.2e6, 6.5e6)
        & (out["rms_residual_m"] < 45.0)
        & (out["worst_residual_m"] < 95.0)
        & (out["num_sats"] >= 5)
    )

    work = out[["time", "x", "y", "z"]].copy()
    for col in ["x", "y", "z"]:
        work.loc[~valid, col] = np.nan
        work[col] = work[col].interpolate(limit_direction="both")
        work[col] = work[col].rolling(7, center=True, min_periods=1).median()

    # Replace points that drift too far from the local path center.
    for window, dist_thr, jump_thr in [(9, 50.0, 75.0), (15, 38.0, 60.0), (21, 30.0, 48.0)]:
        med = pd.DataFrame(index=work.index)
        for col in ["x", "y", "z"]:
            med[col] = work[col].rolling(window, center=True, min_periods=1).median()

        dev = np.sqrt(((work[["x", "y", "z"]] - med[["x", "y", "z"]]) ** 2).sum(axis=1))
        step_prev = np.sqrt(((work[["x", "y", "z"]] - work[["x", "y", "z"]].shift(1)) ** 2).sum(axis=1))
        step_next = np.sqrt(((work[["x", "y", "z"]] - work[["x", "y", "z"]].shift(-1)) ** 2).sum(axis=1))

        bad = (dev > dist_thr) & ((step_prev > jump_thr) | (step_next > jump_thr))
        for col in ["x", "y", "z"]:
            work.loc[bad, col] = med.loc[bad, col]

    # Apply strong but smooth filtering to the visible path.
    for col in ["x", "y", "z"]:
        work[col] = work[col].ewm(span=9, adjust=False).mean()
        work[col] = work[col].rolling(9, center=True, min_periods=1).mean()
        work[col] = work[col].rolling(5, center=True, min_periods=1).mean()

    cleaned = pd.DataFrame(
        {
            "time": out["time"],
            "x": work["x"],
            "y": work["y"],
            "z": work["z"],
            "num_sats": out["num_sats"],
            "rms_residual_m": out["rms_residual_m"],
            "worst_residual_m": out["worst_residual_m"],
            "min_elevation_deg": out["min_elevation_deg"],
            "mean_elevation_deg": out["mean_elevation_deg"],
            "kept_measurement": valid,
        }
    )
    cleaned = add_lla_and_velocity(cleaned)

    # Final local jump guard for KML readability.
    coords = cleaned[["x", "y", "z"]].copy()
    step_prev = np.sqrt(((coords - coords.shift(1)) ** 2).sum(axis=1))
    step_next = np.sqrt(((coords - coords.shift(-1)) ** 2).sum(axis=1))
    jump_mask = (step_prev > 40.0) & (step_next > 40.0)
    for col in ["x", "y", "z"]:
        midpoint = (coords[col].shift(1) + coords[col].shift(-1)) / 2.0
        coords.loc[jump_mask, col] = midpoint.loc[jump_mask]
        coords[col] = coords[col].interpolate(limit_direction="both")

    cleaned[["x", "y", "z"]] = coords
    cleaned = add_lla_and_velocity(cleaned)
    cleaned["utc_time"] = cleaned["time"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return cleaned


def write_kml_points(df: pd.DataFrame, path: str | Path) -> None:
    """Write a KML with one point per second and one track line."""
    path = Path(path)
    point_blocks = []
    for i, row in df.reset_index(drop=True).iterrows():
        point_blocks.append(
            f"""    <Placemark>\n"
            f"      <name>Point {i + 1}</name>\n"
            f"      <description>{row['utc_time']}</description>\n"
            f"      <Point>\n"
            f"        <coordinates>{row['lon_deg']:.8f},{row['lat_deg']:.8f},{row['alt_m']:.3f}</coordinates>\n"
            f"      </Point>\n"
            f"    </Placemark>"""
        )

    line_coords = " ".join(
        f"{row['lon_deg']:.8f},{row['lat_deg']:.8f},{row['alt_m']:.3f}"
        for _, row in df.iterrows()
    )

    kml_text = f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <name>GNSS Track</name>
    <Style id="trackLineStyle">
      <LineStyle>
        <color>ff00ffff</color>
        <width>3</width>
      </LineStyle>
    </Style>
{chr(10).join(point_blocks)}
    <Placemark>
      <name>Track</name>
      <styleUrl>#trackLineStyle</styleUrl>
      <LineString>
        <tessellate>1</tessellate>
        <coordinates>{line_coords}</coordinates>
      </LineString>
    </Placemark>
  </Document>
</kml>
"""
    path.write_text(kml_text, encoding="utf-8")


def run_pipeline(
    obs_file: Path,
    nav_file: Path,
    raw_csv: Path,
    output_csv: Path,
    output_kml: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run the full offline GNSS solution pipeline."""
    print(f"Loading NAV: {nav_file}")
    nav = parse_nav_rinex(nav_file)
    print(f"Loaded navigation records for {len(nav)} satellites")

    print(f"Loading OBS: {obs_file}")
    epochs = parse_obs_rinex(obs_file)
    print(f"Loaded {len(epochs)} epochs")

    solutions = []
    prev_state = None

    for idx, epoch in enumerate(epochs, start=1):
        sol = solve_epoch_position(epoch, nav, prev_state)
        if sol is not None:
            prev_state = sol["state_vec"]
            solutions.append(sol)

        if idx % 50 == 0:
            print(f"Processed {idx}/{len(epochs)} epochs, valid solutions: {len(solutions)}")

    if not solutions:
        raise RuntimeError("No valid GNSS position solutions were produced.")

    raw_df = pd.DataFrame(
        {
            "time": [s["time"] for s in solutions],
            "x": [s["x"] for s in solutions],
            "y": [s["y"] for s in solutions],
            "z": [s["z"] for s in solutions],
            "clock_bias_m": [s["clock_bias_m"] for s in solutions],
            "num_sats": [s["num_sats"] for s in solutions],
            "satellites": [s["satellites"] for s in solutions],
            "rms_residual_m": [s["rms_residual_m"] for s in solutions],
            "worst_residual_m": [s["worst_residual_m"] for s in solutions],
            "min_elevation_deg": [s["min_elevation_deg"] for s in solutions],
            "mean_elevation_deg": [s["mean_elevation_deg"] for s in solutions],
        }
    )

    raw_df["time"] = pd.to_datetime(raw_df["time"], utc=True)
    raw_df = add_lla_and_velocity(raw_df)
    raw_df["utc_time"] = raw_df["time"].dt.strftime("%Y-%m-%d %H:%M:%S")
    raw_df.to_csv(raw_csv, index=False)

    clean_df = clean_track(raw_df)
    clean_df.to_csv(output_csv, index=False)
    write_kml_points(clean_df, output_kml)
    return raw_df, clean_df


def main() -> None:
    """Locate the input files next to the script and run the solver."""
    print(f"Using OBS: {OBS_PATH}")
    print(f"Using NAV: {NAV_PATH}")

    if not OBS_PATH.exists():
        raise FileNotFoundError(f"OBS file not found: {OBS_PATH}")
    if not NAV_PATH.exists():
        raise FileNotFoundError(f"NAV file not found: {NAV_PATH}")

    raw_df, clean_df = run_pipeline(
        obs_file=OBS_PATH,
        nav_file=NAV_PATH,
        raw_csv=RAW_CSV_PATH,
        output_csv=CLEAN_CSV_PATH,
        output_kml=CLEAN_KML_PATH,
    )

    print("\nDone.")
    print(f"Raw CSV:   {RAW_CSV_PATH}")
    print(f"Clean CSV: {CLEAN_CSV_PATH}")
    print(f"KML:       {CLEAN_KML_PATH}")
    print(f"Raw points:   {len(raw_df)}")
    print(f"Clean points: {len(clean_df)}")


if __name__ == "__main__":
    main()
