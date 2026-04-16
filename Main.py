import georinex as gr
import pandas as pd
import numpy as np
import warnings
import sys
import subprocess

warnings.filterwarnings("ignore")

try:
    import simplekml
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "simplekml"])
    import simplekml

# Final definitions
OBS_FILE = 'my_data.obs'
NAV_FILE = 'samsung_nav.nav.rnx'
C = 299792458.0
SMOOTH_WINDOW = 7  # Window for removing spikes


def calculate_satellite_position(nav_data, sv_id, transmit_time):
    try:
        # Selects the navigation record for that satellite
        sv_nav = nav_data.sel(sv=sv_id).dropna(dim='time', how='all')
        if len(sv_nav.time) == 0: return None
        # Selects the time record closest to the measurement time
        diff = np.abs(sv_nav.time.values - transmit_time)
        best_toe_idx = np.argmin(diff)
        data = sv_nav.isel(time=best_toe_idx)

        def get_v(name):
            return data[name].item() if name in data else None

        # Takes the satellite's trajectory description, and calculates where it is in space at the desired time
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
        return np.array([x_p * np.cos(omega_k) - y_p * np.cos(i) * np.sin(omega_k),
                         x_p * np.sin(omega_k) + y_p * np.cos(i) * np.cos(omega_k), y_p * np.sin(i)]), get_v(
            'SVclockBias') # Clock bias
    except:
        return None, None


# Main computation function — calculates the receiver position using an initial guess and iterative corrections
def estimate_receiver_position(sat_positions, measured_ranges):
    x = np.array([0, 0, 0, 0], dtype=float)
    for _ in range(20):
        G, y_res = [], []
        for i in range(len(sat_positions)):
            dist = np.linalg.norm(sat_positions[i] - x[:3])
            # The effect of shifting my guess in a certain direction on the fit with the satellites
            G.append([(x[0] - sat_positions[i][0]) / dist, (x[1] - sat_positions[i][1]) / dist,
                      (x[2] - sat_positions[i][2]) / dist, 1.0])
            #The size of the guess error for each satellite
            y_res.append(measured_ranges[i] - (dist + x[3]))
        try:
            #Updating the guess
            delta_x = np.linalg.inv(np.array(G).T @ np.array(G)) @ np.array(G).T @ np.array(y_res)
            x += delta_x
            if np.linalg.norm(delta_x[:3]) < 1e-3: break
        except:
            return None
    return x


# Converting from ECEF to LLA in order to obtain a geographic point
def ecef_to_lla(x, y, z):
    a, f = 6378137.0, 1 / 298.257223563
    b = a * (1 - f)
    e2, ep2 = 1 - (b ** 2 / a ** 2), (a ** 2 - b ** 2) / b ** 2
    p = np.sqrt(x ** 2 + y ** 2)
    th = np.arctan2(a * z, b * p)
    lon = np.arctan2(y, x)
    lat = np.arctan2(z + ep2 * b * (np.sin(th) ** 3), p - e2 * a * (np.cos(th) ** 3))
    n = a / np.sqrt(1 - e2 * (np.sin(lat) ** 2))
    return np.degrees(lat), np.degrees(lon), p / np.cos(lat) - n


if __name__ == "__main__":
    print(f"--- 🛰️ ג'רוויס מפיק את המסלול הסופי ---")
    obs = gr.load(OBS_FILE)
    nav = gr.load(NAV_FILE, use=['G', 'E'])

    raw_ecef_list = []
    times_list = []

    # Calculating all the raw points
    for t in obs.time.values:
        epoch_data = obs.sel(time=t).dropna(dim='sv', subset=['C1C'])
        sat_pos, ranges = [], []
        for sv in epoch_data.sv.values:
            pos, sv_bias = calculate_satellite_position(nav, str(sv), t)
            if pos is not None:
                sat_pos.append(pos)
                ranges.append(epoch_data.sel(sv=sv)['C1C'].item() + (sv_bias * C))

        if len(sat_pos) >= 4: # Minimum requirements: at least 4 satellites
            res = estimate_receiver_position(sat_pos, ranges)
            if res is not None:
                # Save the raw trajectory points that were obtained
                raw_ecef_list.append(res[:3])
                times_list.append(t)

    # Smoothing the trajectory by examining 7 neighboring points for each point and taking the median
    df_ecef = pd.DataFrame(raw_ecef_list, columns=['x', 'y', 'z']) # Converting the list of trajectory points into a DataFrame
    smoothed_ecef = df_ecef.rolling(window=SMOOTH_WINDOW, center=True, min_periods=1).median()

    # Conversion to LLA and output generation
    results = []
    kml = simplekml.Kml()
    coords_for_kml = []

    for i in range(len(smoothed_ecef)):
        row = smoothed_ecef.iloc[i]
        lat, lon, alt = ecef_to_lla(row['x'], row['y'], row['z']) # Converting from ECEF to LLA in order to obtain a geographic point

        # Calculating speed (meters per second)
        vel = 0.0
        if i > 0:
            p1 = smoothed_ecef.iloc[i - 1]
            p2 = smoothed_ecef.iloc[i]
            vel = np.linalg.norm(p2 - p1)
            if vel > 40: vel = results[-1]['Velocity (m/s)']  # Filtering out speed spikes for speeds greater than 40 meters per second

    # Converting the measurement time into a time object
        results.append({
            'UTC_Time': pd.to_datetime(times_list[i]).strftime('%Y-%m-%d %H:%M:%S'),
            'Lat': lat, 'Lon': lon, 'Alt': alt, 'Velocity (m/s)': round(vel, 2)
        })
        coords_for_kml.append((lon, lat, alt)) # Adding points to a separate list in a format compatible with KML

    pd.DataFrame(results).to_csv('output_path.csv', index=False) #CSV
    ls = kml.newlinestring(name="Final Smooth Route") #KML
    ls.coords = coords_for_kml
    ls.style.linestyle.color, ls.style.linestyle.width = simplekml.Color.blue, 4
    kml.save('output_path.kml')

    print(f"120 clean points were saved in the CSV and KML files.")