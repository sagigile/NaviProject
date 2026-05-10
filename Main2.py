from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ==========================================
# 🎛️ הגדרות קבצים
# ==========================================
OBS_FILENAME = "my_data_2.obs"
NAV_FILENAME = "samsung_nav.nav.rnx"

RAW_CSV_FILENAME = "gnss_solution_raw_galileo.csv"
CLEAN_CSV_FILENAME = "gnss_solution_clean_galileo.csv"
CLEAN_KML_FILENAME = "gnss_solution_clean_galileo.kml"

SCRIPT_DIR = Path(__file__).resolve().parent
OBS_PATH = SCRIPT_DIR / OBS_FILENAME
NAV_PATH = SCRIPT_DIR / NAV_FILENAME
RAW_CSV_PATH = SCRIPT_DIR / RAW_CSV_FILENAME
CLEAN_CSV_PATH = SCRIPT_DIR / CLEAN_CSV_FILENAME
CLEAN_KML_PATH = SCRIPT_DIR / CLEAN_KML_FILENAME

# קבועים פיזיקליים
C = 299792458.0
MU = 3.986005e14
OMEGA_E_DOT = 7.2921151467e-5
F_REL = -4.442807633e-10

# WGS84 קבועי
WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_B = WGS84_A * (1.0 - WGS84_F)
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)
WGS84_EP2 = (WGS84_A * WGS84_A - WGS84_B * WGS84_B) / (WGS84_B * WGS84_B)

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
    values: List[float] = []
    for token in FLOAT_RE.findall(line.replace("D", "E")):
        try:
            values.append(float(token))
        except ValueError:
            pass
    return values


def parse_nav_rinex(nav_path: str | Path) -> Dict[str, List[BroadcastEphemeris]]:
    nav_path = Path(nav_path)
    eph: Dict[str, List[BroadcastEphemeris]] = {}
    with nav_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "END OF HEADER" in line: break
        while True:
            line1 = f.readline()
            if not line1: break
            if not line1.strip(): continue
            sv = line1[:3].strip()
            system = sv[:1]

            # --- שינוי: טוען רק לווייני גלילאו (E) ---
            if system != "E":
                continue

            rest = [f.readline() for _ in range(7)]
            if any(r == "" for r in rest): break
            nums1 = _parse_float_fields(line1[3:])
            if len(nums1) < 9: continue
            year, month, day, hour, minute = map(int, nums1[:5])
            sec = nums1[5]
            sec_int = int(sec)
            micro = int(round((sec - sec_int) * 1e6))
            toc = datetime(year, month, day, hour, minute, sec_int, micro, tzinfo=timezone.utc)
            af0, af1, af2 = nums1[6:9]
            fields = [_parse_float_fields(r) for r in rest]
            flat = [x for row in fields for x in row]
            if len(flat) < 23: continue
            rec = BroadcastEphemeris(
                system=system, sv=sv, toc=toc, af0=af0, af1=af1, af2=af2,
                iode=flat[0], crs=flat[1], delta_n=flat[2], m0=flat[3],
                cuc=flat[4], e=flat[5], cus=flat[6], sqrt_a=flat[7], toe=flat[8],
                cic=flat[9], omega0=flat[10], cis=flat[11], i0=flat[12], crc=flat[13],
                omega=flat[14], omega_dot=flat[15], idot=flat[16],
                week=int(round(flat[18])) if len(flat) > 18 else 0,
                tgd=flat[22] if len(flat) > 22 else 0.0,
            )
            eph.setdefault(sv, []).append(rec)
    for sv in eph: eph[sv].sort(key=lambda item: item.toc)
    return eph


def _parse_obs_field(field: str):
    raw = field[:14].strip()
    if not raw: return None
    try:
        return float(raw.replace("D", "E"))
    except ValueError:
        return None


def parse_obs_rinex(obs_path: Path) -> list[dict]:
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
                if token: obs_types.append(token)
            while len(obs_types) < total_types:
                i += 1
                cont = lines[i]
                for start in range(7, 60, 4):
                    token = cont[start:start + 3].strip()
                    if token: obs_types.append(token)
            obs_types_by_sys[sys_id] = obs_types[:total_types]
        elif "END OF HEADER" in line:
            i += 1
            break
        i += 1
    epochs = []
    while i < len(lines):
        line = lines[i]
        if not line: i += 1; continue
        if line.startswith(">"):
            try:
                year, month, day, hour, minute = int(line[2:6]), int(line[7:9]), int(line[10:12]), int(
                    line[13:15]), int(line[16:18])
                sec = float(line[19:29])
                flag, num_sats = int(line[30:32].strip() or 0), int(line[32:35].strip() or 0)
            except ValueError:
                i += 1; continue
            sec_int = int(sec)
            micro = int(round((sec - sec_int) * 1e6))
            epoch_time = pd.Timestamp(year=year, month=month, day=day, hour=hour, minute=minute, second=sec_int,
                                      microsecond=micro, tz="UTC")
            if flag not in (0, 1):
                i += 1
                for _ in range(num_sats):
                    if i < len(lines): i += 1
                continue
            i += 1
            obs_map = {}
            for _ in range(num_sats):
                if i >= len(lines): break
                sat_line = lines[i]
                if len(sat_line) < 3: i += 1; continue
                sv = sat_line[:3].strip()
                if len(sv) < 2: i += 1; continue
                sys_id = sv[0]

                # --- שינוי: שומר רק לווייני גלילאו (E) ---
                if sys_id != "E":
                    i += 1;
                    continue

                obs_types = obs_types_by_sys.get(sys_id, [])
                n_obs = len(obs_types)
                n_lines = max(1, math.ceil(n_obs / 5))
                sat_text = sat_line[3:]
                i += 1
                for _cont in range(1, n_lines):
                    if i < len(lines): sat_text += lines[i]; i += 1
                values = {}
                for k, obs_type in enumerate(obs_types):
                    start = k * 16
                    field = sat_text[start:start + 16]
                    values[obs_type] = _parse_obs_field(field)
                obs_map[sv] = values
            epochs.append({"time": epoch_time, "obs": obs_map})
        else:
            i += 1
    return epochs


# --- פונקציות מתמטיות וניווט ---

def gps_week_seconds(dt: datetime) -> Tuple[int, float]:
    gps0 = datetime(1980, 1, 6, tzinfo=timezone.utc)
    delta = (dt - gps0).total_seconds()
    week = int(delta // 604800)
    sow = delta - week * 604800
    return week, sow


def wrap_gps_time(seconds: float) -> float:
    while seconds > 302400.0: seconds -= 604800.0
    while seconds < -302400.0: seconds += 604800.0
    return seconds


def solve_kepler(mk: float, e: float, tol: float = 1e-12, max_iter: int = 50) -> float:
    ek = mk
    for _ in range(max_iter):
        denom = 1.0 - e * math.cos(ek)
        if abs(denom) < 1e-14: break
        next_ek = ek - (ek - e * math.sin(ek) - mk) / denom
        if abs(next_ek - ek) < tol: return next_ek
        ek = next_ek
    return ek


def closest_ephemeris(eph_list: List[BroadcastEphemeris], t_rx: datetime) -> Optional[BroadcastEphemeris]:
    if not eph_list: return None
    return min(eph_list, key=lambda item: abs((t_rx - item.toc).total_seconds()))


def choose_best_pseudorange(system: str, obs_map: Dict[str, Optional[float]]) -> Optional[float]:
    # Priority for Galileo: C1C is usually the primary L1 signal
    candidate_map = {"E": ("C1C", "C1X", "C5Q", "C5X", "C7Q", "C7X")}
    for code in candidate_map.get(system, ("C1C",)):
        val = obs_map.get(code)
        if val is not None and val > 1e6: return val
    return None


def satellite_clock_bias(ep: BroadcastEphemeris, tx_time: datetime) -> float:
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
    uk, rk, ik = phik + duk, a * (1.0 - ep.e * math.cos(ek)) + drk, ep.i0 + dik + ep.idot * tk
    xk_p, yk_p = rk * math.cos(uk), rk * math.sin(uk)
    omega_k = ep.omega0 + (ep.omega_dot - OMEGA_E_DOT) * tk - OMEGA_E_DOT * ep.toe
    xk = xk_p * math.cos(omega_k) - yk_p * math.cos(ik) * math.sin(omega_k)
    yk = xk_p * math.sin(omega_k) + yk_p * math.cos(ik) * math.cos(omega_k)
    zk = yk_p * math.sin(ik)
    return np.array([xk, yk, zk], dtype=float)


def earth_rotation_correction(pos: np.ndarray, travel_time: float) -> np.ndarray:
    angle = OMEGA_E_DOT * travel_time
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    return np.array([[cos_a, sin_a, 0.0], [-sin_a, cos_a, 0.0], [0.0, 0.0, 1.0]], dtype=float) @ pos


def ecef_to_lla(x: float, y: float, z: float) -> Tuple[float, float, float]:
    lon = math.atan2(y, x)
    p = math.hypot(x, y)
    th = math.atan2(WGS84_A * z, WGS84_B * p)
    lat = math.atan2(z + WGS84_EP2 * WGS84_B * math.sin(th) ** 3, p - WGS84_E2 * WGS84_A * math.cos(th) ** 3)
    n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * math.sin(lat) ** 2)
    return math.degrees(lat), math.degrees(lon), p / math.cos(lat) - n


def ecef_to_enu_matrix(lat_deg: float, lon_deg: float) -> np.ndarray:
    lat, lon = math.radians(lat_deg), math.radians(lon_deg)
    sl, cl, so, co = math.sin(lat), math.cos(lat), math.sin(lon), math.cos(lon)
    return np.array([[-so, co, 0], [-sl * co, -sl * so, cl], [cl * co, cl * so, sl]])


# --- פונקציות לחישוב פיזור לוויינים (Azimuth/Elevation) ---

def get_az_el(rx_pos_ecef: np.ndarray, sat_pos_ecef: np.ndarray) -> Tuple[float, float]:
    """מחשב עזמיות והגבהה עבור לוויין ביחס למקלט"""
    if np.linalg.norm(rx_pos_ecef) < 1.0: return 0.0, 30.0
    lat, lon, _ = ecef_to_lla(rx_pos_ecef[0], rx_pos_ecef[1], rx_pos_ecef[2])
    enu_mat = ecef_to_enu_matrix(lat, lon)
    los_enu = enu_mat @ (sat_pos_ecef - rx_pos_ecef)
    e, n, u = los_enu
    el = math.degrees(math.atan2(u, math.hypot(e, n)))
    az = math.degrees(math.atan2(e, n)) % 360
    return az, el


def elevation_weight(elev_deg: float) -> float:
    elev = max(elev_deg, 5.0)
    sin_el = math.sin(math.radians(elev))
    return max(0.05, sin_el * sin_el)


def robust_residual_weight(residual_m: float, scale_m: float = 15.0) -> float:
    a = abs(residual_m)
    return 1.0 if a <= scale_m else scale_m / a


def _build_measurement_model(t_rx: datetime, sat_subset: List[Tuple[str, float, BroadcastEphemeris]],
                             state: np.ndarray) -> List[dict]:
    rx_pos, rows = state[:3], []
    for sv, pr, ep in sat_subset:
        tx_guess = t_rx - timedelta(seconds=pr / C)
        dt_sv = satellite_clock_bias(ep, tx_guess)
        tx_time = t_rx - timedelta(seconds=(pr / C) - dt_sv)
        sat_pos = earth_rotation_correction(satellite_position_ecef(ep, tx_time), pr / C)
        rho_vec = rx_pos - sat_pos
        rho = np.linalg.norm(rho_vec)
        if rho < 1.0: continue
        pred = rho + state[3] - C * dt_sv
        az, el = get_az_el(rx_pos, sat_pos)
        rows.append({
            "sv": sv, "pseudorange": pr, "sat_pos": sat_pos, "dt_sv": dt_sv,
            "elev_deg": el, "az_deg": az, "pred": pred, "residual": pr - pred,
            "h_row": np.array(
                [(rx_pos[0] - sat_pos[0]) / rho, (rx_pos[1] - sat_pos[1]) / rho, (rx_pos[2] - sat_pos[2]) / rho, 1.0]),
            "base_weight": elevation_weight(el)
        })
    return rows


def weighted_least_squares(t_rx: datetime, sat_subset: List[Tuple[str, float, BroadcastEphemeris]],
                           x0: Optional[np.ndarray]) -> Optional[Tuple[np.ndarray, List[dict]]]:
    state = np.array(x0 if x0 is not None else [0.0, 0.0, 0.0, 0.0], dtype=float)
    for _ in range(15):
        rows = _build_measurement_model(t_rx, sat_subset, state)
        if len(rows) < 4: return None
        weights = np.ones(len(rows)) if np.linalg.norm(state[:3]) < 1.0 else np.array(
            [r["base_weight"] * robust_residual_weight(r["residual"]) for r in rows])
        sqrt_w = np.sqrt(weights)
        H, y = np.vstack([r["h_row"] for r in rows]), np.array([r["residual"] for r in rows])
        try:
            dx, *_ = np.linalg.lstsq(H * sqrt_w[:, None], y * sqrt_w, rcond=None)
        except:
            return None
        state += dx
        if np.linalg.norm(dx[:3]) < 1e-3: break
    final_rows = _build_measurement_model(t_rx, sat_subset, state)
    return (state, final_rows) if len(final_rows) >= 4 else None


# --- פונקציית בחירת לוויינים מפוזרים (Scattering) ---

def select_distributed_satellites(rows: List[dict], target_count: int = 8) -> List[str]:
    """בוחר לוויינים בצורה מפוזרת על פני 4 רבעים בשמיים כדי לשפר גיאומטריה"""
    quadrants = [[], [], [], []]  # 0-90, 90-180, 180-270, 270-360
    for r in rows:
        idx = int(r["az_deg"] // 90) % 4
        quadrants[idx].append(r)

    selected_svs = []
    # שלב א': קח את הלוויין הכי גבוה מכל רבע
    for q in quadrants:
        if q:
            best = max(q, key=lambda x: x["elev_deg"])
            selected_svs.append(best["sv"])

    # שלב ב': השלם לוויינים נוספים לפי גובה (הכי גבוהים) עד למטרה
    remaining = sorted([r for r in rows if r["sv"] not in selected_svs],
                       key=lambda x: x["elev_deg"], reverse=True)

    for r in remaining:
        if len(selected_svs) < target_count:
            selected_svs.append(r["sv"])

    return selected_svs


def solve_epoch_position(epoch: dict, nav: Dict[str, List[BroadcastEphemeris]], x0: Optional[np.ndarray] = None) -> \
Optional[dict]:
    t_rx, sat_candidates = epoch["time"], []
    for sv, obs_map in epoch["obs"].items():
        if not sv.startswith("E"): continue  # וידוא נוסף לגלילאו בלבד
        pr = choose_best_pseudorange("E", obs_map)
        ep = closest_ephemeris(nav.get(sv, []), t_rx)
        if pr and ep: sat_candidates.append((sv, pr, ep))

    if len(sat_candidates) < 5: return None

    # חישוב ראשוני כדי לקבל עזמיות והגבהה
    initial_res = weighted_least_squares(t_rx, sat_candidates, x0)
    if not initial_res: return None
    state, rows = initial_res

    # --- שיפור: בחירת לוויינים מפוזרים ---
    distributed_svs = select_distributed_satellites(rows)
    working = [c for c in sat_candidates if c[0] in distributed_svs]

    # חישוב סופי עם הפיזור הנבחר
    final_res = weighted_least_squares(t_rx, working, state)
    if not final_res: return None
    state, rows = final_res

    residuals = np.array([r["residual"] for r in rows])
    return {
        "time": t_rx, "x": state[0], "y": state[1], "z": state[2], "clock_bias_m": state[3],
        "num_sats": len(working), "satellites": ",".join([r["sv"] for r in rows]),
        "rms_residual_m": np.sqrt(np.mean(residuals ** 2)), "state_vec": state
    }


# --- שאר הפונקציות (ניקוי מסלול, KML וכו') ללא שינוי מהותי פרט לשמות קבצים ---

def add_lla_and_velocity(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    lla = out.apply(lambda r: ecef_to_lla(r["x"], r["y"], r["z"]), axis=1)
    out[["lat_deg", "lon_deg", "alt_m"]] = pd.DataFrame(lla.tolist(), index=out.index)
    velocities = [np.nan]
    for i in range(1, len(out)):
        dt = (out.loc[i, "time"] - out.loc[i - 1, "time"]).total_seconds()
        d = math.sqrt((out.loc[i, "x"] - out.loc[i - 1, "x"]) ** 2 + (out.loc[i, "y"] - out.loc[i - 1, "y"]) ** 2 + (
                    out.loc[i, "z"] - out.loc[i - 1, "z"]) ** 2)
        velocities.append(d / dt if dt > 0 else np.nan)
    out["velocity_mps"] = velocities
    return out


def clean_track(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().sort_values("time").reset_index(drop=True)
    valid = (out["rms_residual_m"] < 50.0) & (out["num_sats"] >= 4)
    work = out[["time", "x", "y", "z"]].copy()
    for col in ["x", "y", "z"]:
        work.loc[~valid, col] = np.nan
        work[col] = work[col].interpolate().rolling(7, center=True, min_periods=1).median()
        work[col] = work[col].rolling(11, center=True, min_periods=1).mean()
    cleaned = pd.concat([out[["time"]], work[["x", "y", "z"]], out.drop(columns=["time", "x", "y", "z"])], axis=1)
    cleaned["kept_measurement"] = valid
    cleaned = add_lla_and_velocity(cleaned)
    cleaned["utc_time"] = cleaned["time"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return cleaned


def write_kml_points(df: pd.DataFrame, path: Path) -> None:
    line_coords = " ".join(f"{r['lon_deg']:.8f},{r['lat_deg']:.8f},{r['alt_m']:.3f}" for _, r in df.iterrows())
    kml_text = f"""<?xml version="1.0" encoding="UTF-8"?><kml xmlns="http://www.opengis.net/kml/2.2"><Document><name>Galileo Track</name>
    <Style id="t"><LineStyle><color>ff00ff00</color><width>4</width></LineStyle></Style>
    <Placemark><name>Route</name><styleUrl>#t</styleUrl><LineString><tessellate>1</tessellate><coordinates>{line_coords}</coordinates></LineString></Placemark></Document></kml>"""
    path.write_text(kml_text, encoding="utf-8")


def main() -> None:
    print(f"--- 🛰️ ג'רוויס: עיבוד Galileo בלבד עם פיזור גיאומטרי ---")
    nav = parse_nav_rinex(NAV_PATH)
    epochs = parse_obs_rinex(OBS_PATH)
    solutions, prev_state = [], None
    for epoch in epochs:
        sol = solve_epoch_position(epoch, nav, prev_state)
        if sol: prev_state = sol["state_vec"]; solutions.append(sol)
    if not solutions: print("לא נמצאו פתרונות."); return
    raw_df = pd.DataFrame(solutions)
    raw_df["time"] = pd.to_datetime(raw_df["time"], utc=True)
    raw_df = add_lla_and_velocity(raw_df)
    clean_df = clean_track(raw_df)
    raw_df.to_csv(RAW_CSV_PATH, index=False)
    clean_df[clean_df["kept_measurement"]].to_csv(CLEAN_CSV_PATH, index=False)
    write_kml_points(clean_df, CLEAN_KML_PATH)
    print(f"סיום. קבצים נשמרו בתיקייה: {SCRIPT_DIR}")


if __name__ == "__main__":
    main()