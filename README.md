# README — RINEX Navigation Exercise

## Purpose of the Code

The code computes a GNSS track from RINEX files:

### Input Files

- Observation file (`.obs`) — contains pseudorange measurements from satellites for each measurement time.
- Navigation file (`.nav.rnx`) — contains satellite orbit and clock data.

### Output Files

1. **Raw CSV** — a table containing all position points that the code was able to compute.
2. **Clean CSV** — a table containing points that passed filtering and cleaning for a cleaner track.
3. **KML** — a file for displaying the route and points on a map, for example in Google Earth.

By default, the code uses only Galileo satellites, computes a position for each epoch, selects satellites according to geometric quality using PDOP/GDOP, runs Weighted Least Squares, filters out outlier points, smooths the track, and finally exports CSV and KML files.


## General Pipeline Structure

```text
Start
  |
  v
Check that OBS and NAV files exist
  |
  v
Read NAV file
parse_nav_rinex()
  |
  v
Read OBS file
parse_obs_rinex()
  |
  v
For each epoch:
  |
  +--> Choose usable pseudoranges
  |
  +--> Match each satellite to closest ephemeris
  |
  +--> Build motion-based initial guess if possible
  |
  +--> First WLS solution using all candidate satellites
  |
  +--> Select best satellite subset by PDOP/GDOP
  |
  +--> Final WLS solution using selected satellites
  |
  +--> Save epoch solution
  |
  v
Build raw DataFrame
  |
  v
Add lat/lon/alt and velocity
add_lla_and_velocity()
  |
  v
Clean and smooth track
clean_track()
  |
  +--> Filter by RMS residual
  |
  +--> Filter by number of satellites
  |
  +--> Filter by PDOP
  |
  +--> Filter unrealistic speed jumps
  |
  +--> Interpolate invalid points
  |
  +--> Apply median smoothing
  |
  +--> Apply mean smoothing
  |
  v
Write Raw CSV
  |
  v
Write Clean CSV
  |
  v
Write KML route and point markers
write_kml_points()
  |
  v
End
```

---

## Preliminary Explanation of the Code

At the beginning of the file, the input and output file names, file paths, CSV columns, physical constants, WGS84 constants, satellite selection settings, and filtering settings are defined.

Important settings:

| Name | Meaning |

| `ENABLED_SYSTEMS = {"E"}` | Uses Galileo satellites only |
| `MIN_SATS_FOR_FINAL = 4` | Minimum of 4 satellites required to compute a 3D position and clock bias |
| `TARGET_SATS_FOR_FINAL = 8` | Target number of satellites for the final selection |
| `MAX_CLEAN_SPEED_MPS = 45.0` | Maximum reasonable speed before a point is considered an outlier jump |
| `RAW_CSV_COLUMNS` | Order and names of the columns in the Raw file |
| `CLEAN_CSV_COLUMNS` | Order and names of the columns in the Clean file |



## Functions

### `BroadcastEphemeris`

A data class (`dataclass`) that represents one satellite navigation record from the NAV file. It stores the parameters required to compute the satellite position and satellite clock correction.

---

### `_parse_float_fields(line)`

Receives a text line from a RINEX file and extracts numbers from it. The function also handles scientific notation using `D`, which sometimes appears in RINEX, and converts it to `E` so Python can read it as a number.

---

### `parse_nav_rinex(nav_path)`

Reads the navigation file. It skips the header, reads satellite ephemeris records, filters only the satellite systems defined in `ENABLED_SYSTEMS`, and builds a dictionary where each satellite has a list of navigation records sorted by time.

---

### `_parse_obs_field(field)`

Parses a single observation field from the OBS file. It returns the measurement value and the SSI value if it exists. SSI is a signal strength indicator from the RINEX file.

---

### `parse_obs_rinex(obs_path)`

Reads the observation file. First, it reads from the header which measurement types exist for each satellite system. Then it goes through each epoch and stores the measurements of the satellites that were received at each time.

---

### `gps_week_seconds(dt)`

Converts UTC time into a GPS week and the number of seconds within that week. This is required because satellite calculations in navigation files use GPS time.

---

### `wrap_gps_time(seconds)`

Corrects GPS time differences so they stay within the standard half-week range. This is important to avoid errors around GPS week transitions.

---

### `solve_kepler(mk, e, tol, max_iter)`

Solves Kepler's equation iteratively. The result is used to compute the satellite position in its orbit.

---

### `closest_ephemeris(eph_list, t_rx)`

For a specific satellite, selects the navigation record whose time is closest to the measurement time. This allows the code to use orbit data that is as suitable as possible for that epoch.

---

### `choose_best_pseudorange(system, obs_map)`

Selects the best pseudorange measurement from the available measurement types for that satellite. It also returns the SSI value and the selected measurement code.

---

### `satellite_clock_bias(ep, tx_time)`

Computes the satellite clock correction at the estimated transmission time. This correction is important because the pseudorange is affected by clock differences between the satellite and the receiver.

---

### `satellite_position_ecef(ep, tx_time)`

Computes the satellite position in the ECEF coordinate system using the ephemeris data and the transmission time. This is one of the central calculations in the code.

---

### `earth_rotation_correction(pos, travel_time)`

Corrects the satellite position for Earth's rotation during the time the signal travels from the satellite to the receiver. Without this correction, the satellite position relative to the receiver would not be accurate enough.

---

### `ecef_to_lla(x, y, z)`

Converts ECEF coordinates into latitude, longitude, and altitude. This output is used to display the points on a map.

---

### `ecef_to_enu_matrix(lat_deg, lon_deg)`

Builds a conversion matrix from ECEF to the local ENU coordinate system. ENU stands for East, North, Up, and is used to compute the satellite angle relative to the receiver.

---

### `get_az_el(rx_pos_ecef, sat_pos_ecef)`

Computes the azimuth and elevation angle of a satellite relative to the receiver. A higher elevation angle is usually better because satellites low on the horizon tend to be less reliable.

---

### `elevation_weight(elev_deg)`

Converts the satellite elevation angle into a weight. Higher satellites receive a higher measurement weight.

---

### `signal_weight(ssi)`

Converts the SSI signal strength into a weight. A stronger signal receives a higher weight, but in this code, geometry is still more important than signal strength.

---

### `robust_residual_weight(residual_m, scale_m)`

Assigns a lower weight to measurements with a large residual. The function is meant to reduce the influence of outlier measurements after the solution has already started to converge.

---

### `compute_dops(rows)`

Computes PDOP and GDOP from the satellite geometry matrix. Lower values indicate better satellite geometry.

---

### `_candidate_quality_for_prefilter(row)`

Gives an initial score to a satellite in order to reduce the number of candidates before checking all combinations. This is meant to improve efficiency when there are many satellites.

---

### `select_satellites_by_dop(rows, target_count)`

Selects the final satellite group according to PDOP/GDOP. It checks satellite combinations, computes geometric quality for each one, and chooses the group with the best score.

---

### `_build_measurement_model(t_rx, sat_subset, state)`

Builds the measurement model for one iteration of Weighted Least Squares. It computes satellite positions, clock corrections, residuals, geometry matrix rows, and base weights.

---

### `weighted_least_squares(t_rx, sat_subset, x0, max_iter, warmup_iters)`

Computes the receiver position and clock bias using Weighted Least Squares. At the beginning of the iterations, it lets the solution converge, and only afterward applies weights that reduce the effect of large residuals.

---

### `rms_from_rows(rows)`

Computes the RMS of the residuals. This is a measure of the fit quality between the measurements and the computed solution.

---

### `build_motion_prediction_state(...)`

Creates an initial guess for the next position using the two previous points. If there is enough history, it estimates movement direction and speed and uses them to guess the next position.

---

### `solve_with_best_initial_guess(...)`

Runs the WLS solution with two possible initial guesses: the previous point and motion prediction. It then chooses the solution with the lower RMS.

---

### `solve_epoch_position(epoch, nav, ...)`

Solves the position for a single epoch. It selects suitable measurements, matches each satellite with a navigation record, computes an initial solution, selects satellites according to PDOP/GDOP, computes a final solution, and returns the point data.

---

### `add_lla_and_velocity(df)`

Adds latitude, longitude, altitude, and velocity to the table. Velocity is computed as the distance between consecutive points divided by the time difference between them.

---

### `build_speed_valid_mask(df, initial_valid, max_speed_mps, passes)`

Filters points that require an unrealistic speed compared to the previous valid point. This allows the code to identify abnormal GNSS jumps in the track.

---

### `clean_track(df)`

Cleans the track. It filters by RMS, number of satellites, PDOP, and abnormal speed, and then smooths the ECEF coordinates using interpolation, median rolling, and mean rolling.

---

### `write_kml_points(df, path)`

Creates a KML file with a continuous route line and marked points. Each point includes a description with time, satellites, RMS, PDOP, GDOP, and velocity.

---

### `main()`

The main function that runs the full process. It checks that the input files exist, reads NAV and OBS, solves all epochs, creates Raw and Clean DataFrames, saves CSV files, creates a KML file, and prints a summary to the screen.


## Main Meaning of the Quality Metrics

| Metric | Meaning |

| `rms_residual_m` | How well the solution fits the pseudorange measurements. Lower is better |
| `pdop` | Quality of satellite geometry for 3D positioning. Lower is better |
| `gdop` | Similar to PDOP, but also includes the effect of receiver clock bias |
| `velocity_mps` | Speed between consecutive points. Used to identify abnormal jumps |
| `kept_measurement` | Whether the point passed the clean-track filtering |

---

## Important Notes for Running the Code

1. The input files must be in the same folder as the Python file.
2. The input file names are defined at the top of the code in `OBS_FILENAME` and `NAV_FILENAME`.
3. By default, the code uses only Galileo satellites (`E`).
4. If you want to also test GPS, you can change `ENABLED_SYSTEMS`, but this may improve or harm the result depending on the quality of the recording.
5. The KML file is built from the same points that are included in the Clean CSV.

---

## Short Summary

The code receives raw GNSS measurements, computes the receiver position for each measurement time, selects satellites according to geometric quality, runs a Weighted Least Squares solution, filters jumps and unreliable points, smooths the track, and creates CSV and KML files for display and analysis.

## Authors

- Sagi Rahat
- Taliya Levin
