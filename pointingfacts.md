# Pointing Variables and Initializations

## pointing.py

- **Variable:** `def generate_pointing_sphere(data_struct: Dict[str, Any], n_points: int, debug: bool`
  - **Initialization:** `False) -> None:`
  - **Location:** Line 18

- **Variable:** `data_struct['pointing_spheres'][n_points]`
  - **Initialization:** `\`
  - **Location:** Line 32

- **Variable:** `print(f"\n--- Debugging Pointing Sphere (n_points`
  - **Initialization:** `{n_points}) ---")`
  - **Location:** Line 36

- **Variable:** `generated_vectors`
  - **Initialization:** `data_struct['pointing_spheres'][n_points]`
  - **Location:** Line 37

- **Variable:** `def update_satellite_pointing(data_struct: Dict[str, Any], debug: bool`
  - **Initialization:** `False) -> None:`
  - **Location:** Line 51

- **Variable:** `pointing_state`
  - **Initialization:** `data_struct['satellites']['pointing_state']`
  - **Location:** Line 60

- **Variable:** `pointing_vectors_all`
  - **Initialization:** `data_struct['satellites']['pointing']`
  - **Location:** Line 61

- **Variable:** `count`
  - **Initialization:** `int(pointing_state[i, POINTING_COUNT_IDX])`
  - **Location:** Line 66

- **Variable:** `grid`
  - **Initialization:** `data_struct['pointing_spheres'][count]`
  - **Location:** Line 69

- **Variable:** `place`
  - **Initialization:** `int(pointing_state[i, POINTING_PLACE_IDX])`
  - **Location:** Line 71

- **Variable:** `pointing_vectors_all[i]`
  - **Initialization:** `grid[place]`
  - **Location:** Line 80

- **Variable:** `print(f"Satellite {i}: Pointing location {place}, Excluded: {excluded !`
  - **Initialization:** `0}")`
  - **Location:** Line 84

- **Variable:** `pointing_state[i, POINTING_PLACE_IDX]`
  - **Initialization:** `place`
  - **Location:** Line 88

- **Variable:** `pointing_state[i, POINTING_PLACE_IDX]`
  - **Initialization:** `place`
  - **Location:** Line 93

- **Variable:** `sim_data['satellites']['pointing_state'][0, POINTING_COUNT_IDX]`
  - **Initialization:** `n_points_sphere`
  - **Location:** Line 130

- **Variable:** `sim_data['satellites']['pointing_state'][0, POINTING_PLACE_IDX]`
  - **Initialization:** `0 # Start at the first point`
  - **Location:** Line 131

- **Variable:** `name`
  - **Initialization:** `'Initial Satellite Pointing'`
  - **Location:** Line 164

- **Variable:** `update_satellite_pointing(sim_data, debug`
  - **Initialization:** `False) # Turn off debug`
  - **Location:** Line 168

- **Variable:** `current_pointed_direction`
  - **Initialization:** `sim_data['satellites']['pointing'][0]`
  - **Location:** Line 169

- **Variable:** `name`
  - **Initialization:** `'Pointing History'`
  - **Location:** Line 190

- **Variable:** `name`
  - **Initialization:** `'Pointing Path'`
  - **Location:** Line 200

- **Variable:** `title`
  - **Initialization:** `"Satellite Pointing with Exclusion",`
  - **Location:** Line 204

- **Variable:** `p`
  - **Initialization:** `data_struct['satellites']['pointing'][satellite_indices]`
  - **Location:** Line 233

- **Variable:** `data_struct['satellites']['pointing'][satellite_indices]`
  - **Initialization:** `p_new`
  - **Location:** Line 248

## constellation.py

- **Variable:** `pointing_state_list`
  - **Initialization:** `[]`
  - **Location:** Line 39

- **Variable:** `pointing_state[POINTING_COUNT_IDX]`
  - **Initialization:** `grid_points`
  - **Location:** Line 60

- **Variable:** `pointing_state[POINTING_PLACE_IDX]`
  - **Initialization:** `random.randint(0,grid_points-1)`
  - **Location:** Line 61

- **Variable:** `'pointing': np.zeros((n, 3), dtype`
  - **Initialization:** `float),`
  - **Location:** Line 77

- **Variable:** `#        sim_data['satellites']['pointing_state']`
  - **Initialization:** `np.vstack([sim_data['satellites']['pointing_state'], pointing_state_array])`
  - **Location:** Line 91

- **Variable:** `pointing_state_list`
  - **Initialization:** `[]`
  - **Location:** Line 121

- **Variable:** `pointing_state[POINTING_COUNT_IDX]`
  - **Initialization:** `grid_points`
  - **Location:** Line 142

- **Variable:** `pointing_state[POINTING_PLACE_IDX]`
  - **Initialization:** `random.randint(0,grid_points-1)`
  - **Location:** Line 143

- **Variable:** `'pointing': np.zeros((n, 3), dtype`
  - **Initialization:** `float),`
  - **Location:** Line 161

- **Variable:** `sim_data['satellites']['pointing_state']`
  - **Initialization:** `np.vstack([sim_data['satellites']['pointing_state'], pointing_state_array])`
  - **Location:** Line 173

## simulation.py

No pointing variable initializations found in this file.

## propagation.py

- **Variable:** `'pointing': np.zeros((num_sats, 3), dtype`
  - **Initialization:** `float),`
  - **Location:** Line 42

- **Variable:** `'pointing_state': np.zeros((num_sats, 2), dtype`
  - **Initialization:** `int),`
  - **Location:** Line 43

- **Variable:** `data_struct[category]['pointing']`
  - **Initialization:** `positions / norms`
  - **Location:** Line 161

## observatories.py

- **Variable:** `'pointing': np.zeros((num_observatories, 3), dtype`
  - **Initialization:** `float),`
  - **Location:** Line 22

## demo_sky_scan.py

- **Variable:** `pointing_vector`
  - **Initialization:** `np.array([x, y, z])`
  - **Location:** Line 55

- **Variable:** `sim_data['satellites']['pointing'][0]`
  - **Initialization:** `pointing_vector`
  - **Location:** Line 58

## demo_pointing_sequence.py

- **Variable:** `dummy_tle_path`
  - **Initialization:** `"dummy_tle_pointing.txt"`
  - **Location:** Line 48

- **Variable:** `pointing_state`
  - **Initialization:** `sim_data['satellites']['pointing_state']`
  - **Location:** Line 58

- **Variable:** `pointing_state[0, POINTING_COUNT_IDX]`
  - **Initialization:** `100`
  - **Location:** Line 59

- **Variable:** `vectors`
  - **Initialization:** `sim_data['satellites']['pointing']`
  - **Location:** Line 77

- **Variable:** `vectors`
  - **Initialization:** `sim_data['satellites']['pointing']`
  - **Location:** Line 86

- **Variable:** `title`
  - **Initialization:** `"Satellite Pointing Sequence Over 30 Time Steps",`
  - **Location:** Line 150

- **Variable:** `fig`
  - **Initialization:** `demo_pointing_sequence()`
  - **Location:** Line 170

## demo_show_geo_search.py

- **Variable:** `fig1`
  - **Initialization:** `plot_pointing_vectors(sim_data, 'Initial Pointing Vectors', sim_start_time)`
  - **Location:** Line 27

- **Variable:** `p`
  - **Initialization:** `sim_data['satellites']['pointing'][0]`
  - **Location:** Line 33

- **Variable:** `fig2`
  - **Initialization:** `plot_pointing_vectors(sim_data, 'After 5 Updates', sim_start_time)`
  - **Location:** Line 47

- **Variable:** `fig3`
  - **Initialization:** `plot_pointing_vectors(sim_data, 'After 15 Updates', sim_start_time)`
  - **Location:** Line 54

## constants.py

- **Variable:** `POINTING_COUNT_IDX`
  - **Initialization:** `0         # Number of points in the pointing grid`
  - **Location:** Line 42

- **Variable:** `POINTING_PLACE_IDX`
  - **Initialization:** `1         # Current index in the pointing sequence`
  - **Location:** Line 43

## scandetectors.py

- **Variable:** `detectorVect`
  - **Initialization:** `sim_data['detector'] # detector pointings`
  - **Location:** Line 29

