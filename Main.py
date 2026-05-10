import georinex as gr
import pandas as pd
import numpy as np
import warnings
import sys
import subprocess

# השתקת אזהרות טכניות
warnings.filterwarnings("ignore")

# בדיקת התקנה של simplekml
try:
    import simplekml
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "simplekml"])
    import simplekml

# ==========================================
# 🎛️ הגדרות הפרויקט (Galileo Optimized)
# ==========================================
OBS_FILE = 'my_data.obs'
NAV_FILE = 'samsung_nav.nav.rnx'
C = 299792458.0
MIN_SATS = 4
SMOOTH_WINDOW = 7


# ==========================================

def calculate_satellite_position(nav_data, sv_id, transmit_time):
    """חישוב מיקום לוויין במרחב לפי פרמטרי קפלר"""
    try:
        sv_nav = nav_data.sel(sv=sv_id).dropna(dim='time', how='all')
        if len(sv_nav.time) == 0: return None
        diff = np.abs(sv_nav.time.values - transmit_time)
        best_toe_idx = np.argmin(diff)
        data = sv_nav.isel(time=best_toe_idx)

        def get_v(name):
            return data[name].item() if name in data else None

        e, sqrtA, M0, dn = get_v('Eccentricity'), get_v('sqrtA'), get_v('M0'), get_v('DeltaN')
        i0, idot, omega0, omega_dot = get_v('Io'), get_v('IDOT'), get_v('Omega0'), get_v('OmegaDot')
        arg_per, toe_val = get_v('omega'), get_v('Toe')

        if None in [e, sqrtA, M0, i0, omega0]: return None

        mu, omega_e = 3.986005e14, 7.2921151467e-5
        toe_ts = data.time.values.astype('datetime64[s]').astype(float)
        tk = transmit_time.astype('datetime64[s]').astype(float) - toe_ts

        A = sqrtA ** 2
        n = np.sqrt(mu / A ** 3) + dn
        M = M0 + n * tk
        E = M
        for _ in range(10): E = M + e * np.sin(E)
        v = np.arctan2(np.sqrt(1 - e ** 2) * np.sin(E), np.cos(E) - e)
        phi = v + arg_per
        u = phi + (get_v('Cuc') or 0) * np.cos(2 * phi) + (get_v('Cus') or 0) * np.sin(2 * phi)
        r = A * (1 - e * np.cos(E)) + (get_v('Crc') or 0) * np.cos(2 * phi) + (get_v('Crs') or 0) * np.sin(2 * phi)
        i = i0 + idot * tk + (get_v('Cic') or 0) * np.cos(2 * phi) + (get_v('Cis') or 0) * np.sin(2 * phi)
        x_p, y_p = r * np.cos(u), r * np.sin(u)
        omega_k = omega0 + (omega_dot - omega_e) * tk - omega_e * toe_val

        pos = np.array([x_p * np.cos(omega_k) - y_p * np.cos(i) * np.sin(omega_k),
                        x_p * np.sin(omega_k) + y_p * np.cos(i) * np.cos(omega_k),
                        y_p * np.sin(i)])
        return pos, get_v('SVclockBias')
    except:
        return None, None


def estimate_receiver_position(sat_positions, measured_ranges):
    """אלגוריתם Least Squares למציאת מיקום המקלט"""
    x = np.array([0, 0, 0, 0], dtype=float)
    for _ in range(20):
        G, y_res = [], []
        for i in range(len(sat_positions)):
            dist = np.linalg.norm(sat_positions[i] - x[:3])
            G.append([(x[0] - sat_positions[i][0]) / dist, (x[1] - sat_positions[i][1]) / dist,
                      (x[2] - sat_positions[i][2]) / dist, 1.0])
            y_res.append(measured_ranges[i] - (dist + x[3]))
        try:
            delta_x = np.linalg.inv(np.array(G).T @ np.array(G)) @ np.array(G).T @ np.array(y_res)
            x += delta_x
            if np.linalg.norm(delta_x[:3]) < 1e-3: break
        except:
            return None
    return x


def ecef_to_lla(x, y, z):
    """המרה מקואורדינטות כדור הארץ (X,Y,Z) לקווי אורך ורוחב"""
    a, f = 6378137.0, 1 / 298.257223563
    b = a * (1 - f)
    e2, ep2 = 1 - (b ** 2 / a ** 2), (a ** 2 - b ** 2) / b ** 2
    p = np.sqrt(x ** 2 + y ** 2)
    th = np.arctan2(a * z, b * p)
    lon = np.arctan2(y, x)
    lat = np.arctan2(z + ep2 * b * (np.sin(th) ** 3), p - e2 * a * (np.cos(th) ** 3))
    n = a / np.sqrt(1 - e2 * (np.sin(lat) ** 2))
    alt = p / np.cos(lat) - n
    return np.degrees(lat), np.degrees(lon), alt


if __name__ == "__main__":
    print(f"--- 🛰️ ג'רוויס: מעבד נתוני Galileo מתוך {OBS_FILE} ---")

    try:
        obs = gr.load(OBS_FILE)
        nav = gr.load(NAV_FILE, use=['E'])
    except Exception as e:
        print(f"❌ שגיאה בטעינת הקבצים: {e}")
        sys.exit(1)

    raw_ecef_list, times_list = [], []

    for t in obs.time.values:
        epoch_data = obs.sel(time=t).dropna(dim='sv', subset=['C1C'])
        valid_sats = []

        for sv in epoch_data.sv.values:
            sv_name = str(sv)
            if not sv_name.startswith('E'): continue

            pos, sv_bias = calculate_satellite_position(nav, sv_name, t)
            if pos is not None:
                snr = epoch_data.sel(sv=sv)['S1C'].item()
                valid_sats.append({
                    'id': sv_name,  # הוספת מזהה ייחודי למניעת שגיאת ההשוואה[cite: 1]
                    'pos': pos,
                    'range': epoch_data.sel(sv=sv)['C1C'].item() + (sv_bias * C),
                    'snr': snr
                })

        if len(valid_sats) >= MIN_SATS:
            selected_sats = []
            quadrants = [[], [], [], []]

            for s in valid_sats:
                if s['pos'][0] >= 0 and s['pos'][1] >= 0:
                    quadrants[0].append(s)
                elif s['pos'][0] >= 0 and s['pos'][1] < 0:
                    quadrants[1].append(s)
                elif s['pos'][0] < 0 and s['pos'][1] >= 0:
                    quadrants[2].append(s)
                else:
                    quadrants[3].append(s)

            for q in quadrants:
                if q: selected_sats.append(max(q, key=lambda x: x['snr']))

            # תיקון השגיאה: השוואה לפי ID ולא לפי האובייקט כולו[cite: 1]
            if len(selected_sats) < 4:
                selected_ids = [x['id'] for x in selected_sats]
                remaining = [s for s in valid_sats if s['id'] not in selected_ids]
                selected_sats.extend(remaining[:(4 - len(selected_sats))])

            pos_list = [s['pos'] for s in selected_sats]
            range_list = [s['range'] for s in selected_sats]

            res = estimate_receiver_position(pos_list, range_list)
            if res is not None:
                raw_ecef_list.append(res[:3])
                times_list.append(t)

    if not raw_ecef_list:
        print("⚠️ לא נמצאו מספיק לווייני Galileo בפיזור הנדרש.")
        sys.exit(1)

    df_ecef = pd.DataFrame(raw_ecef_list, columns=['x', 'y', 'z'])
    smoothed_ecef = df_ecef.rolling(window=SMOOTH_WINDOW, center=True, min_periods=1).median()

    results = []
    kml = simplekml.Kml()
    coords_for_kml = []

    for i in range(len(smoothed_ecef)):
        row = smoothed_ecef.iloc[i]
        lat, lon, alt = ecef_to_lla(row['x'], row['y'], row['z'])

        vel = 0.0
        if i > 0:
            vel = np.linalg.norm(smoothed_ecef.iloc[i] - smoothed_ecef.iloc[i - 1])
            if vel > 40: vel = results[-1]['Velocity (m/s)'] if results else 0.0

        results.append({
            'UTC_Time': pd.to_datetime(times_list[i]).strftime('%Y-%m-%d %H:%M:%S'),
            'Lat': lat, 'Lon': lon, 'Alt': alt, 'Velocity (m/s)': round(vel, 2)
        })
        coords_for_kml.append((lon, lat, alt))

    pd.DataFrame(results).to_csv('output_path.csv', index=False)
    ls = kml.newlinestring(name="High Precision Galileo Route")
    ls.coords = coords_for_kml
    ls.style.linestyle.color, ls.style.linestyle.width = simplekml.Color.green, 4
    kml.save('output_path.kml')
    print(f"✅ הצלחנו! חושבו {len(results)} נקודות מבוססות Galileo בפיזור מקסימלי.")