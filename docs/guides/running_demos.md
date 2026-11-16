# Running VibeVolts Demos

This guide explains how to run, filter, and export the VibeVolts demonstration suites.

---

## 1. WebGL Context Limits & Grouped Execution

Plotly 3D scatter plots (`go.Scatter3d`) use **WebGL** via Three.js to render interactive
graphics. Most web browsers and IDE notebook renderers impose a strict limit of **~16 WebGL canvas
contexts per page/tab**.

Because VibeVolts includes 17 consolidated demonstration functions (producing 25 figures across
all suites), running all demos inline at once can exhaust WebGL contexts and display a
`"WebGL is not supported by your browser"` error.

To avoid this, `all_demos.py` organizes demos into three thematic groups:

| Group Name | Focus Area | Included Demo Functions |
| :--- | :--- | :--- |
| **`'orbits'`** | Satellite & Observatory Orbits | `demo2`, `demo_trajectories`, `demo_constellation`, `demo_observatories_only`, `run_observatory_animation` |
| **`'pointing'`** | Sensor Pointing & Sky Scans | `demo_pointing_plot`, `demo_sky_scan`, `demo_pointing_sequence`, `demo_show_geo_search`, `demo_exclusion_pointing` |
| **`'radiometry'`** | Signal, Background & Exclusion | `demo_fixedpoints`, `demo_exclusion_table`, `demo_lambertian`, `demo_requiredIntegrationTime`, `demo_vector_resorting_plot`, `demoFixed`, `demo_gap_time_histogram` |

---

## 2. Using `all_demos.py` in Python & Jupyter Notebooks

### Listing Demo Groups
To inspect available groups and functions from a Python script or notebook cell:

```python
import all_demos

all_demos.list_demo_groups()
```

### Running a Specific Group Inline
To render plots inline within a Jupyter Notebook without hitting WebGL context limits:

```python
import all_demos

# Run orbit geometry suite
all_demos.run_all_demos(group='orbits')

# Run sensor pointing suite
all_demos.run_all_demos(group='pointing')

# Run radiometry & target suite
all_demos.run_all_demos(group='radiometry')
```

### Running Specific Demos by Name
You can pass a list of function names or callables:

```python
all_demos.run_all_demos(demos=['demo2', 'demo_trajectories', 'demo_constellation'])
```

Or call any demo function directly on `all_demos` to inspect its Plotly `Figure`:

```python
fig = all_demos.demo_sky_scan()
fig.show()
```

### Exporting All Demos to HTML
To run all demos without hitting browser WebGL limits, export them to a single HTML file:

```python
all_demos.run_all_demos(group='all', save_html=True)
```
This generates `all_demo_plots.html` in the current working directory. Opening this file in your
system web browser allows full interactive 3D navigation across all figures.

---

## 3. Running Demos from the Command Line

To execute all demos from a terminal and write `all_demo_plots.html`:

```bash
python all_demos.py
```
