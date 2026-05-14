from __future__ import annotations

import itertools
import math
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from html import escape
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ============================================================
# Configuration
# ============================================================
OBS_FILENAME = "my_data_3.obs"
NAV_FILENAME = "samsung_nav.nav.rnx"

RAW_CSV_FILENAME = "gnss_solution_short_path.csv"
CLEAN_CSV_FILENAME = "gnss_solution_clean_short_path.csv"
CLEAN_KML_FILENAME = "gnss_solution_clean_short_path.kml"

SCRIPT_DIR = Path(__file__).resolve().parent
OBS_PATH = SCRIPT_DIR / OBS_FILENAME
NAV_PATH = SCRIPT_DIR / NAV_FILENAME
RAW_CSV_PATH = SCRIPT_DIR / RAW_CSV_FILENAME
CLEAN_CSV_PATH = SCRIPT_DIR / CLEAN_CSV_FILENAME
CLEAN_KML_PATH = SCRIPT_DIR / CLEAN_KML_FILENAME

RAW_CSV_COLUMNS = [
    "utc_time",
    "x",
    "y",
    "z",
    "satellites",
    "rms_residual_m",
    "pdop",
    "initial_guess_method",
    "lat_deg",
    "lon_deg",
    "velocity_mps",
    "kept_measurement",
]

CLEAN_CSV_COLUMNS = [
    "utc_time",
    "x",
    "y",
    "z",
    "satellites",
    "rms_residual_m",
    "pdop",
    "initial_guess_method",
    "lat_deg",
    "lon_deg",
    "velocity_mps",
]

# Physical constants
C = 299792458.0
MU = 3.986005e14
OMEGA_E_DOT = 7.2921151467e-5
F_REL = -4.442807633e-10

# WGS84 constants
WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_B = WGS84_A * (1.0 - WGS84_F)
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)
WGS84_EP2 = (WGS84_A * WGS84_A - WGS84_B * WGS84_B) / (WGS84_B * WGS84_B)

# Satellite systems to use.
# Keep Galileo-only by default because adding GPS made this recording worse.
# You can try {"E", "G"} again only if the raw result improves.
ENABLED_SYSTEMS = {"E"}

# Galileo tie bonus is irrelevant when using only Galileo, but kept for optional GPS tests.
GALILEO_TIE_BONUS = 0.04

# Signal strength has a noticeable but secondary effect.
# DOP/geometry must still dominate final satellite selection.
SIGNAL_PREFILTER_FACTOR = 3.0
DOP_SELECTION_SIGNAL_BONUS = 0.03
SIGNAL_WEIGHT_BASE = 0.65
SIGNAL_WEIGHT_STEP = 0.10

# DOP-based satellite selection settings.
MIN_SATS_FOR_FINAL = 4
TARGET_SATS_FOR_FINAL = 8
MAX_CANDIDATES_FOR_COMBINATION_SEARCH = 14

# Clean-track speed filter.
# 45 m/s is about 162 km/h, so it keeps normal walking, cycling, and city/intercity driving
# while rejecting GNSS jumps that create unrealistic point-to-point movement.
MAX_CLEAN_SPEED_MPS = 45.0
SPEED_FILTER_PASSES = 2

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
    """Extract numeric values from a RINEX line and normalize D/E exponent notation."""
    values: List[float] = []
    for token in FLOAT_RE.findall(line.replace("D", "E")):
        try:
            values.append(float(token))
        except ValueError:
            pass
    return values


def parse_nav_rinex(nav_path: str | Path) -> Dict[str, List[BroadcastEphemeris]]:
    """Read the RINEX navigation file and build ephemeris records per satellite."""
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

            # Keep only observations from enabled satellite systems.
            if system not in ENABLED_SYSTEMS:
                continue

            rest = [f.readline() for _ in range(7)]
            if any(r == "" for r in rest):
                break

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


def _parse_obs_field(field: str) -> Tuple[Optional[float], Optional[int]]:
    """
    Parses a RINEX observation field.

    The first 14 characters usually hold the observation value.
    The last character may hold SSI, which is a signal strength indicator.
    """
    raw = field[:14].strip()
    if not raw:
        return None, None

    try:
        value = float(raw.replace("D", "E"))
    except ValueError:
        return None, None

    ssi = None
    if len(field) >= 16:
        ssi_char = field[15].strip()
        if ssi_char.isdigit():
            ssi = int(ssi_char)

    return value, ssi


def parse_obs_rinex(obs_path: Path) -> list[dict]:
    """Read the RINEX observation file and collect pseudorange observations per epoch."""
    lines = Path(obs_path).read_text(encoding="utf-8", errors="ignore").splitlines()
    obs_types_by_sys = {}
    i = 0

    while i < len(lines):
        line = lines[i]

        if "SYS / # / OBS TYPES" in line:
            sys_id = line[0]
            try:
                total_types = int(line[3:6].strip())
            except ValueError:
                total_types = 0

            obs_types = []
            for start in range(7, 60, 4):
                token = line[start:start + 3].strip()
                if token:
                    obs_types.append(token)

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

    while i < len(lines):
        line = lines[i]
        if not line:
            i += 1
            continue

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

                # Keep only observations from enabled satellite systems.
                if sys_id not in ENABLED_SYSTEMS:
                    i += 1
                    continue

                obs_types = obs_types_by_sys.get(sys_id, [])
                n_obs = len(obs_types)
                n_lines = max(1, math.ceil(n_obs / 5))

                sat_text = sat_line[3:]
                i += 1

                for _cont in range(1, n_lines):
                    if i < len(lines):
                        sat_text += lines[i]
                        i += 1

                values = {}
                ssi_values = {}

                for k, obs_type in enumerate(obs_types):
                    start = k * 16
                    field = sat_text[start:start + 16]
                    value, ssi = _parse_obs_field(field)
                    values[obs_type] = value
                    ssi_values[obs_type] = ssi

                # Store SSI values inside the same observation map.
                values["ssi"] = ssi_values
                obs_map[sv] = values

            epochs.append({"time": epoch_time, "obs": obs_map})
        else:
            i += 1

    return epochs


# ============================================================
# Time, orbit, and coordinate helpers
# ============================================================

def gps_week_seconds(dt: datetime) -> Tuple[int, float]:
    """Convert UTC datetime to GPS week and seconds-of-week."""
    gps0 = datetime(1980, 1, 6, tzinfo=timezone.utc)
    delta = (dt - gps0).total_seconds()
    week = int(delta // 604800)
    sow = delta - week * 604800
    return week, sow


def wrap_gps_time(seconds: float) -> float:
    """Wrap GPS time differences into the standard half-week range."""
    while seconds > 302400.0:
        seconds -= 604800.0
    while seconds < -302400.0:
        seconds += 604800.0
    return seconds


def solve_kepler(mk: float, e: float, tol: float = 1e-12, max_iter: int = 50) -> float:
    """Solve Kepler's equation iteratively for the eccentric anomaly."""
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
    """Select the navigation record closest to the receiver time."""
    if not eph_list:
        return None
    return min(eph_list, key=lambda item: abs((t_rx - item.toc).total_seconds()))


def choose_best_pseudorange(system: str, obs_map: Dict[str, Optional[float]]) -> Tuple[Optional[float], Optional[int], Optional[str]]:
    """
    Selects the best pseudorange and returns its value, SSI, and observation code.
    """
    candidate_map = {
        "E": ("C1C", "C1X", "C5Q", "C5X", "C7Q", "C7X"),
        "G": ("C1C", "C1W", "C1X", "C2W", "C2X", "C5Q", "C5X"),
    }

    ssi_map = obs_map.get("ssi", {})

    for code in candidate_map.get(system, ("C1C",)):
        val = obs_map.get(code)
        if val is not None and val > 1e6:
            return val, ssi_map.get(code), code

    return None, None, None


def satellite_clock_bias(ep: BroadcastEphemeris, tx_time: datetime) -> float:
    """Compute satellite clock correction at the estimated transmit time."""
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
    """Compute satellite ECEF position from broadcast ephemeris."""
    _, sow_tx = gps_week_seconds(tx_time)
    tk = wrap_gps_time(sow_tx - ep.toe)

    a = ep.sqrt_a * ep.sqrt_a
    n0 = math.sqrt(MU / (a ** 3))
    n = n0 + ep.delta_n
    mk = ep.m0 + n * tk
    ek = solve_kepler(mk, ep.e)

    vk = math.atan2(
        math.sqrt(1.0 - ep.e * ep.e) * math.sin(ek),
        math.cos(ek) - ep.e,
    )
    phik = vk + ep.omega

    duk = ep.cus * math.sin(2.0 * phik) + ep.cuc * math.cos(2.0 * phik)
    drk = ep.crs * math.sin(2.0 * phik) + ep.crc * math.cos(2.0 * phik)
    dik = ep.cis * math.sin(2.0 * phik) + ep.cic * math.cos(2.0 * phik)

    uk = phik + duk
    rk = a * (1.0 - ep.e * math.cos(ek)) + drk
    ik = ep.i0 + dik + ep.idot * tk

    xk_p = rk * math.cos(uk)
    yk_p = rk * math.sin(uk)

    omega_k = ep.omega0 + (ep.omega_dot - OMEGA_E_DOT) * tk - OMEGA_E_DOT * ep.toe

    xk = xk_p * math.cos(omega_k) - yk_p * math.cos(ik) * math.sin(omega_k)
    yk = xk_p * math.sin(omega_k) + yk_p * math.cos(ik) * math.cos(omega_k)
    zk = yk_p * math.sin(ik)

    return np.array([xk, yk, zk], dtype=float)


def earth_rotation_correction(pos: np.ndarray, travel_time: float) -> np.ndarray:
    """Rotate satellite position to compensate for Earth rotation during signal travel."""
    angle = OMEGA_E_DOT * travel_time
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)

    return np.array(
        [
            [cos_a, sin_a, 0.0],
            [-sin_a, cos_a, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    ) @ pos


def ecef_to_lla(x: float, y: float, z: float) -> Tuple[float, float, float]:
    """Convert ECEF coordinates to geodetic latitude, longitude, and altitude."""
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
    """Build the ECEF-to-ENU rotation matrix for a local reference point."""
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)

    sl = math.sin(lat)
    cl = math.cos(lat)
    so = math.sin(lon)
    co = math.cos(lon)

    return np.array(
        [
            [-so, co, 0.0],
            [-sl * co, -sl * so, cl],
            [cl * co, cl * so, sl],
        ]
    )


# ============================================================
# Satellite geometry, weights, and DOP selection
# ============================================================

def get_az_el(rx_pos_ecef: np.ndarray, sat_pos_ecef: np.ndarray) -> Tuple[float, float]:
    """Computes azimuth and elevation for a satellite relative to the receiver."""
    if np.linalg.norm(rx_pos_ecef) < 1.0:
        return 0.0, 30.0

    lat, lon, _ = ecef_to_lla(rx_pos_ecef[0], rx_pos_ecef[1], rx_pos_ecef[2])
    enu_mat = ecef_to_enu_matrix(lat, lon)
    los_enu = enu_mat @ (sat_pos_ecef - rx_pos_ecef)

    e, n, u = los_enu
    el = math.degrees(math.atan2(u, math.hypot(e, n)))
    az = math.degrees(math.atan2(e, n)) % 360.0

    return az, el


def elevation_weight(elev_deg: float) -> float:
    """Convert satellite elevation angle into a measurement weight."""
    elev = max(elev_deg, 5.0)
    sin_el = math.sin(math.radians(elev))
    return max(0.05, sin_el * sin_el)


def signal_weight(ssi: Optional[int]) -> float:
    """
    Converts RINEX SSI into a measurement weight factor.

    Signal strength now has more influence than before, but geometry/DOP
    still remains the main factor in satellite selection.
    """
    if ssi is None:
        return 1.0

    ssi = max(1, min(9, int(ssi)))

    # Stronger than before: 1 -> 0.75, 9 -> 1.55
    return SIGNAL_WEIGHT_BASE + SIGNAL_WEIGHT_STEP * ssi


def robust_residual_weight(residual_m: float, scale_m: float = 15.0) -> float:
    """Downweight large residuals after the solution has had time to converge."""
    a = abs(residual_m)
    return 1.0 if a <= scale_m else scale_m / a


def compute_dops(rows: List[dict]) -> Tuple[float, float]:
    """
    Computes PDOP and GDOP from the geometry matrix.

    Lower PDOP/GDOP means better satellite geometry.
    """
    if len(rows) < 4:
        return float("inf"), float("inf")

    H = np.vstack([r["h_row"] for r in rows])

    try:
        q = np.linalg.inv(H.T @ H)
    except np.linalg.LinAlgError:
        try:
            q = np.linalg.pinv(H.T @ H)
        except np.linalg.LinAlgError:
            return float("inf"), float("inf")

    diag = np.diag(q)
    if np.any(diag < 0):
        return float("inf"), float("inf")

    pdop = float(math.sqrt(diag[0] + diag[1] + diag[2]))
    gdop = float(math.sqrt(diag[0] + diag[1] + diag[2] + diag[3]))

    return pdop, gdop


def _candidate_quality_for_prefilter(row: dict) -> float:
    """
    Prefilter score used only to keep the combination search small.
    Final selection is based mainly on PDOP/GDOP.
    """
    ssi = row["signal_strength"] if row["signal_strength"] is not None else 0
    galileo_bonus = 3.0 if row["sv"].startswith("E") else 0.0
    return row["elev_deg"] + SIGNAL_PREFILTER_FACTOR * ssi + galileo_bonus


def select_satellites_by_dop(rows: List[dict], target_count: int = TARGET_SATS_FOR_FINAL) -> Tuple[List[str], float, float]:
    """
    Selects satellites by checking combinations and choosing the lowest DOP.

    Geometry is the main criterion. Signal strength and Galileo preference are only tie-breakers.
    """
    if len(rows) < MIN_SATS_FOR_FINAL:
        return [], float("inf"), float("inf")

    usable = [r for r in rows if r["elev_deg"] >= 5.0]
    if len(usable) < MIN_SATS_FOR_FINAL:
        usable = rows[:]

    # Limit the number of combinations so the code stays practical on long recordings.
    if len(usable) > MAX_CANDIDATES_FOR_COMBINATION_SEARCH:
        usable = sorted(usable, key=_candidate_quality_for_prefilter, reverse=True)[:MAX_CANDIDATES_FOR_COMBINATION_SEARCH]

    max_k = min(target_count, len(usable))
    min_k = min(MIN_SATS_FOR_FINAL, max_k)

    best_score = float("inf")
    best_svs: List[str] = []
    best_pdop = float("inf")
    best_gdop = float("inf")

    for k in range(min_k, max_k + 1):
        for combo in itertools.combinations(usable, k):
            combo_rows = list(combo)
            pdop, gdop = compute_dops(combo_rows)
            if not np.isfinite(pdop) or not np.isfinite(gdop):
                continue

            avg_signal = np.mean([
                r["signal_strength"] if r["signal_strength"] is not None else 0
                for r in combo_rows
            ])
            galileo_ratio = sum(1 for r in combo_rows if r["sv"].startswith("E")) / len(combo_rows)

            # DOP dominates. Signal is stronger than before, but remains secondary.
            score = pdop + 0.05 * gdop - DOP_SELECTION_SIGNAL_BONUS * avg_signal - GALILEO_TIE_BONUS * galileo_ratio

            if score < best_score:
                best_score = score
                best_svs = [r["sv"] for r in combo_rows]
                best_pdop = pdop
                best_gdop = gdop

    return best_svs, best_pdop, best_gdop


# ============================================================
# Weighted least squares and motion prediction
# ============================================================

def _build_measurement_model(
    t_rx: datetime,
    sat_subset: List[Tuple[str, float, BroadcastEphemeris, Optional[int]]],
    state: np.ndarray,
) -> List[dict]:
    """Build residuals, geometry rows, and base weights for one WLS iteration."""
    rx_pos = state[:3]
    rows = []

    for sv, pr, ep, ssi in sat_subset:
        tx_guess = t_rx - timedelta(seconds=pr / C)
        dt_sv = satellite_clock_bias(ep, tx_guess)
        tx_time = t_rx - timedelta(seconds=(pr / C) - dt_sv)

        sat_pos = earth_rotation_correction(
            satellite_position_ecef(ep, tx_time),
            pr / C,
        )

        rho_vec = rx_pos - sat_pos
        rho = np.linalg.norm(rho_vec)
        if rho < 1.0:
            continue

        pred = rho + state[3] - C * dt_sv
        az, el = get_az_el(rx_pos, sat_pos)

        base_w = elevation_weight(el) * signal_weight(ssi)

        rows.append(
            {
                "sv": sv,
                "pseudorange": pr,
                "sat_pos": sat_pos,
                "dt_sv": dt_sv,
                "signal_strength": ssi,
                "elev_deg": el,
                "az_deg": az,
                "pred": pred,
                "residual": pr - pred,
                "h_row": np.array(
                    [
                        (rx_pos[0] - sat_pos[0]) / rho,
                        (rx_pos[1] - sat_pos[1]) / rho,
                        (rx_pos[2] - sat_pos[2]) / rho,
                        1.0,
                    ]
                ),
                "base_weight": base_w,
            }
        )

    return rows


def weighted_least_squares(
    t_rx: datetime,
    sat_subset: List[Tuple[str, float, BroadcastEphemeris, Optional[int]]],
    x0: Optional[np.ndarray],
    max_iter: int = 15,
    warmup_iters: int = 3,
) -> Optional[Tuple[np.ndarray, List[dict]]]:
    """
    Solves receiver position and clock bias using weighted least squares.

    Robust residual weights are applied only after a few warmup iterations.
    """
    state = np.array(x0 if x0 is not None else [0.0, 0.0, 0.0, 0.0], dtype=float)

    for iter_idx in range(max_iter):
        rows = _build_measurement_model(t_rx, sat_subset, state)
        if len(rows) < 4:
            return None

        if np.linalg.norm(state[:3]) < 1.0:
            weights = np.ones(len(rows))
        else:
            if iter_idx < warmup_iters:
                weights = np.array([r["base_weight"] for r in rows], dtype=float)
            else:
                weights = np.array(
                    [r["base_weight"] * robust_residual_weight(r["residual"]) for r in rows],
                    dtype=float,
                )

        sqrt_w = np.sqrt(weights)
        H = np.vstack([r["h_row"] for r in rows])
        y = np.array([r["residual"] for r in rows])

        try:
            dx, *_ = np.linalg.lstsq(H * sqrt_w[:, None], y * sqrt_w, rcond=None)
        except Exception:
            return None

        state += dx

        if np.linalg.norm(dx[:3]) < 1e-3:
            break

    final_rows = _build_measurement_model(t_rx, sat_subset, state)
    return (state, final_rows) if len(final_rows) >= 4 else None


def rms_from_rows(rows: List[dict]) -> float:
    """Compute RMS residual in meters from measurement model rows."""
    residuals = np.array([r["residual"] for r in rows], dtype=float)
    return float(np.sqrt(np.mean(residuals ** 2)))


def build_motion_prediction_state(
    t_rx: datetime,
    prev_state: Optional[np.ndarray],
    prev_prev_state: Optional[np.ndarray],
    prev_time: Optional[pd.Timestamp],
    prev_prev_time: Optional[pd.Timestamp],
) -> Optional[np.ndarray]:
    """
    Predicts the next receiver state using previous movement direction and speed.
    """
    if prev_state is None or prev_prev_state is None:
        return None

    if prev_time is None or prev_prev_time is None:
        return None

    dt_prev = (prev_time - prev_prev_time).total_seconds()
    dt_next = (t_rx - prev_time).total_seconds()

    if dt_prev <= 0 or dt_next <= 0:
        return None

    velocity_vec = (prev_state[:3] - prev_prev_state[:3]) / dt_prev

    predicted = prev_state.copy()
    predicted[:3] = prev_state[:3] + velocity_vec * dt_next
    predicted[3] = prev_state[3]

    return predicted


def solve_with_best_initial_guess(
    t_rx: datetime,
    sat_subset: List[Tuple[str, float, BroadcastEphemeris, Optional[int]]],
    regular_x0: Optional[np.ndarray],
    motion_x0: Optional[np.ndarray],
) -> Optional[Tuple[np.ndarray, List[dict], str]]:
    """
    Solves the same epoch using two possible initial guesses.
    """
    candidates = []

    regular_res = weighted_least_squares(t_rx, sat_subset, regular_x0)
    if regular_res:
        state, rows = regular_res
        candidates.append((rms_from_rows(rows), state, rows, "previous_point_guess"))

    if motion_x0 is not None:
        motion_res = weighted_least_squares(t_rx, sat_subset, motion_x0)
        if motion_res:
            state, rows = motion_res
            candidates.append((rms_from_rows(rows), state, rows, "motion_prediction_guess"))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0])
    _, best_state, best_rows, best_method = candidates[0]

    return best_state, best_rows, best_method


# ============================================================
# Per-epoch GNSS solution
# ============================================================

def solve_epoch_position(
    epoch: dict,
    nav: Dict[str, List[BroadcastEphemeris]],
    prev_state: Optional[np.ndarray] = None,
    prev_prev_state: Optional[np.ndarray] = None,
    prev_time: Optional[pd.Timestamp] = None,
    prev_prev_time: Optional[pd.Timestamp] = None,
) -> Optional[dict]:
    """Solve one epoch by selecting usable satellites and estimating receiver position."""
    t_rx = epoch["time"]
    sat_candidates = []

    for sv, obs_map in epoch["obs"].items():
        system = sv[:1]
        if system not in ENABLED_SYSTEMS:
            continue

        pr, ssi, obs_code = choose_best_pseudorange(system, obs_map)
        ep = closest_ephemeris(nav.get(sv, []), t_rx)

        if pr and ep:
            sat_candidates.append((sv, pr, ep, ssi))

    if len(sat_candidates) < MIN_SATS_FOR_FINAL:
        return None

    motion_x0 = build_motion_prediction_state(
        t_rx=t_rx,
        prev_state=prev_state,
        prev_prev_state=prev_prev_state,
        prev_time=prev_time,
        prev_prev_time=prev_prev_time,
    )

    # First solve uses all candidates to estimate receiver state and geometry.
    initial_res = solve_with_best_initial_guess(
        t_rx=t_rx,
        sat_subset=sat_candidates,
        regular_x0=prev_state,
        motion_x0=motion_x0,
    )

    if not initial_res:
        return None

    initial_state, initial_rows, _ = initial_res

    # Final subset is selected by PDOP/GDOP-based geometry quality.
    selected_svs, pdop, gdop = select_satellites_by_dop(initial_rows, target_count=TARGET_SATS_FOR_FINAL)
    working = [c for c in sat_candidates if c[0] in selected_svs]

    if len(working) < MIN_SATS_FOR_FINAL:
        return None

    final_res = solve_with_best_initial_guess(
        t_rx=t_rx,
        sat_subset=working,
        regular_x0=prev_state if prev_state is not None else initial_state,
        motion_x0=motion_x0,
    )

    if not final_res:
        return None

    state, rows, selected_method = final_res
    residuals = np.array([r["residual"] for r in rows], dtype=float)

    return {
        "time": t_rx,
        "x": state[0],
        "y": state[1],
        "z": state[2],
        "clock_bias_m": state[3],
        "num_sats": len(rows),
        "satellites": ",".join([r["sv"] for r in rows]),
        "rms_residual_m": np.sqrt(np.mean(residuals ** 2)),
        "pdop": pdop,
        "gdop": gdop,
        "initial_guess_method": selected_method,
        "state_vec": state,
    }


# ============================================================
# Track post-processing and output files
# ============================================================

def add_lla_and_velocity(df: pd.DataFrame) -> pd.DataFrame:
    """Add latitude, longitude, altitude, and point-to-point velocity columns."""
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


def build_speed_valid_mask(
    df: pd.DataFrame,
    initial_valid: pd.Series,
    max_speed_mps: float = MAX_CLEAN_SPEED_MPS,
    passes: int = SPEED_FILTER_PASSES,
) -> pd.Series:
    """Rejects points that require unrealistic speed from the previous kept point."""
    valid = initial_valid.copy().reset_index(drop=True)

    for _ in range(max(1, passes)):
        filtered = valid.copy()
        last_kept_idx = None

        for i in range(len(df)):
            if not valid.iloc[i]:
                continue

            if last_kept_idx is None:
                last_kept_idx = i
                continue

            dt = (df.loc[i, "time"] - df.loc[last_kept_idx, "time"]).total_seconds()
            if dt <= 0:
                filtered.iloc[i] = False
                continue

            distance_m = math.sqrt(
                (df.loc[i, "x"] - df.loc[last_kept_idx, "x"]) ** 2
                + (df.loc[i, "y"] - df.loc[last_kept_idx, "y"]) ** 2
                + (df.loc[i, "z"] - df.loc[last_kept_idx, "z"]) ** 2
            )
            speed_mps = distance_m / dt

            if speed_mps > max_speed_mps:
                filtered.iloc[i] = False
            else:
                last_kept_idx = i

        if filtered.equals(valid):
            break

        valid = filtered

    return valid


def clean_track(df: pd.DataFrame) -> pd.DataFrame:
    """Filter noisy points, smooth ECEF coordinates, and prepare the clean track table."""
    out = df.copy().sort_values("time").reset_index(drop=True)

    # Keep the RMS filter, add a mild DOP filter, and reject unrealistic speed jumps.
    valid = (out["rms_residual_m"] < 50.0) & (out["num_sats"] >= MIN_SATS_FOR_FINAL)

    if "pdop" in out.columns:
        valid = valid & (out["pdop"] < 15.0)

    valid = build_speed_valid_mask(out, valid)

    work = out[["time", "x", "y", "z"]].copy()

    for col in ["x", "y", "z"]:
        work.loc[~valid, col] = np.nan

        # Aggressive smoothing:
        # - interpolate missing/invalid points
        # - use a wider median window to remove jumps
        # - use a wider mean window to make the route smoother
        work[col] = work[col].interpolate()
        work[col] = work[col].rolling(7, center=True, min_periods=1).median()
        work[col] = work[col].rolling(11, center=True, min_periods=1).mean()

    cleaned = pd.concat(
        [
            out[["time"]],
            work[["x", "y", "z"]],
            out.drop(columns=["time", "x", "y", "z"]),
        ],
        axis=1,
    )

    cleaned["kept_measurement"] = valid
    cleaned = add_lla_and_velocity(cleaned)
    cleaned["utc_time"] = cleaned["time"].dt.strftime("%Y-%m-%d %H:%M:%S")

    return cleaned


def write_kml_points(df: pd.DataFrame, path: Path) -> None:
    """Writes a KML file with one route line and one marker per clean point."""
    if df.empty:
        path.write_text(
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<kml xmlns="http://www.opengis.net/kml/2.2"><Document><name>Clean GNSS Track</name></Document></kml>',
            encoding="utf-8",
        )
        return

    kml_df = df.dropna(subset=["lat_deg", "lon_deg", "alt_m"]).copy()
    if "time" in kml_df.columns:
        kml_df = kml_df.sort_values("time").reset_index(drop=True)

    line_coords = " ".join(
        f"{r['lon_deg']:.8f},{r['lat_deg']:.8f},{r['alt_m']:.3f}"
        for _, r in kml_df.iterrows()
    )

    point_placemarks = []
    for idx, r in kml_df.iterrows():
        time_text = str(r.get("utc_time", r.get("time", "")))
        satellites = escape(str(r.get("satellites", "")))
        description = (
            f"<![CDATA["
            f"<b>Time:</b> {escape(time_text)}<br/>"
            f"<b>Satellites:</b> {satellites}<br/>"
            f"<b>RMS residual:</b> {r.get('rms_residual_m', '')}<br/>"
            f"<b>PDOP:</b> {r.get('pdop', '')}<br/>"
            f"<b>GDOP:</b> {r.get('gdop', '')}<br/>"
            f"<b>Velocity:</b> {r.get('velocity_mps', '')} m/s"
            f"]]>"
        )
        point_placemarks.append(
            f"""
        <Placemark>
            <name>Point {idx + 1}</name>
            <description>{description}</description>
            <styleUrl>#red_point</styleUrl>
            <Point>
                <coordinates>{r['lon_deg']:.8f},{r['lat_deg']:.8f},{r['alt_m']:.3f}</coordinates>
            </Point>
        </Placemark>"""
        )

    kml_text = f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
<Document>
    <name>Clean GNSS Track</name>

    <Style id="route_line">
        <LineStyle>
            <color>ff0000ff</color>
            <width>4</width>
        </LineStyle>
    </Style>

    <Style id="red_point">
        <IconStyle>
            <scale>0.85</scale>
            <Icon>
                <href>http://maps.google.com/mapfiles/kml/paddle/red-circle.png</href>
            </Icon>
        </IconStyle>
        <LabelStyle>
            <scale>0</scale>
        </LabelStyle>
    </Style>

    <Placemark>
        <name>Clean route</name>
        <styleUrl>#route_line</styleUrl>
        <LineString>
            <tessellate>1</tessellate>
            <altitudeMode>clampToGround</altitudeMode>
            <coordinates>{line_coords}</coordinates>
        </LineString>
    </Placemark>

    <Folder>
        <name>Clean CSV points</name>
        {''.join(point_placemarks)}
    </Folder>
</Document>
</kml>
"""
    path.write_text(kml_text, encoding="utf-8")


def main() -> None:
    """Run the full GNSS pipeline and write raw CSV, clean CSV, and KML outputs."""
    print("--- GNSS processing: Galileo-only, DOP selection, and mild smoothing ---")

    if not OBS_PATH.exists():
        print(f"OBS file not found: {OBS_PATH}")
        return

    if not NAV_PATH.exists():
        print(f"NAV file not found: {NAV_PATH}")
        return

    nav = parse_nav_rinex(NAV_PATH)
    epochs = parse_obs_rinex(OBS_PATH)

    solutions = []

    prev_prev_state = None
    prev_state = None

    prev_prev_time = None
    prev_time = None

    for epoch in epochs:
        sol = solve_epoch_position(
            epoch=epoch,
            nav=nav,
            prev_state=prev_state,
            prev_prev_state=prev_prev_state,
            prev_time=prev_time,
            prev_prev_time=prev_prev_time,
        )

        if sol:
            # Shift history after accepting a valid solution.
            prev_prev_state = prev_state
            prev_prev_time = prev_time

            prev_state = sol["state_vec"]
            prev_time = sol["time"]

            solutions.append(sol)

    if not solutions:
        print("No valid GNSS solutions were found.")
        return

    raw_df = pd.DataFrame(solutions)
    raw_df["time"] = pd.to_datetime(raw_df["time"], utc=True)

    # Keep state_vec only internally. It is not useful in the output CSV.
    if "state_vec" in raw_df.columns:
        raw_df = raw_df.drop(columns=["state_vec"])

    raw_df = add_lla_and_velocity(raw_df)
    clean_df = clean_track(raw_df)

    raw_output_df = raw_df.copy()
    raw_output_df["kept_measurement"] = clean_df["kept_measurement"].values
    raw_output_df["utc_time"] = raw_output_df["time"].dt.strftime("%Y-%m-%d %H:%M:%S")
    raw_output_df[RAW_CSV_COLUMNS].to_csv(RAW_CSV_PATH, index=False)

    # The clean CSV and KML are generated from the exact same kept measurements.
    clean_output_df = clean_df[clean_df["kept_measurement"]].copy()
    clean_output_df[CLEAN_CSV_COLUMNS].to_csv(CLEAN_CSV_PATH, index=False)
    write_kml_points(clean_output_df, CLEAN_KML_PATH)

    print(f"Done. Files were saved in: {SCRIPT_DIR}")
    print(f"Raw CSV:   {RAW_CSV_PATH}")
    print(f"Clean CSV: {CLEAN_CSV_PATH}")
    print(f"KML:       {CLEAN_KML_PATH}")
    print(f"Raw points: {len(raw_df)}")
    print(f"Clean kept points: {int(clean_df['kept_measurement'].sum())}")


if __name__ == "__main__":
    main()
