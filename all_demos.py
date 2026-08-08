import plotly.graph_objects as go
from typing import Union, List, Callable, Optional

# Import all demo functions
from demo1 import demo1
from demo2 import demo2
from demo3 import demo3
from demogeo import demogeo
from demo_fixedpoints import demo_fixedpoints
from demo_exclusion_table import demo_exclusion_table
from demo_pointing_plot import demo_pointing_plot
from radiometry_test import demoFixed

from demo_lambertian import demo_lambertian
from demo_sky_scan import demo_sky_scan
from demo_pointing_vectors import demo_pointing_vectors
from demo_pointing_sequence import demo_pointing_sequence
from demo_constellation import demo_constellation
from demo_show_geo_search import demo_show_geo_search
from demo_requiredIntegrationTime import demo_requiredIntegrationTime
from fibonacciSearch import test_vector_resorting
from pointing import demo_exclusion_pointing
from demo_gap_time_histogram import demo_gap_time_histogram
from demo_observatories_only import demo_observatories_only


def demo_vector_resorting_plot() -> go.Figure:
    """
    Runs the test_vector_resorting function and returns its figure.
    """
    print("\n--- Starting Demo: Vector Resorting ---")
    return test_vector_resorting()


DEMO_GROUPS = {
    'orbits': [
        ('demo1', demo1),
        ('demo2', demo2),
        ('demo3', demo3),
        ('demogeo', demogeo),
        ('demo_constellation', demo_constellation),
        ('demo_observatories_only', demo_observatories_only),
    ],
    'pointing': [
        ('demo_pointing_plot', demo_pointing_plot),
        ('demo_sky_scan', demo_sky_scan),
        ('demo_pointing_vectors', demo_pointing_vectors),
        ('demo_pointing_sequence', demo_pointing_sequence),
        ('demo_show_geo_search', demo_show_geo_search),
        ('demo_exclusion_pointing', demo_exclusion_pointing),
    ],
    'radiometry': [
        ('demo_fixedpoints', demo_fixedpoints),
        ('demo_exclusion_table', demo_exclusion_table),
        ('demo_lambertian', demo_lambertian),
        ('demo_requiredIntegrationTime', demo_requiredIntegrationTime),
        ('demo_vector_resorting_plot', demo_vector_resorting_plot),
        ('demoFixed', demoFixed),
        ('demo_gap_time_histogram', demo_gap_time_histogram),
    ],
}


def list_demo_groups() -> None:
    """Lists available demo groups and their included demo functions."""
    print("=== VibeVolts Demo Groups ===")
    for group_name, demo_list in DEMO_GROUPS.items():
        print(f"\nGroup: '{group_name}' ({len(demo_list)} demos)")
        for name, func in demo_list:
            print(f"  - {name}")
    print("\nUse run_all_demos(group='<name>') to run a specific group inline.")


def run_all_demos(
    group: str = 'all',
    demos: Optional[List[Union[str, Callable]]] = None,
    save_html: bool = False
) -> None:
    """
    Runs demo functions by suite group, specific list, or all demos.

    Args:
        group (str): Demo group to run: 'orbits', 'pointing', 'radiometry', or 'all'.
        demos (list): Optional list of demo names or functions to run.
        save_html (bool): If True, saves plots to HTML file.
    """
    funcs_to_run = []

    if demos is not None:
        all_name_map = {}
        for g_list in DEMO_GROUPS.values():
            for name, func in g_list:
                all_name_map[name] = func
        for item in demos:
            if callable(item):
                funcs_to_run.append((item.__name__, item))
            elif isinstance(item, str) and item in all_name_map:
                funcs_to_run.append((item, all_name_map[item]))
            else:
                print(f"Warning: Demo '{item}' not recognized. Skipping.")
    elif group in DEMO_GROUPS:
        funcs_to_run = DEMO_GROUPS[group]
    elif group == 'all':
        for g_list in DEMO_GROUPS.values():
            funcs_to_run.extend(g_list)
        if not save_html:
            print("Notice: Displaying all demos inline can exceed browser WebGL limits (16 max).")
            print("Defaulting to save_html=True. Set group='orbits', 'pointing', or 'radiometry' for inline notebook display.")
            save_html = True
    else:
        print(f"Unknown group '{group}'. Available groups: {list(DEMO_GROUPS.keys())} or 'all'.")
        return

    figs = []
    print(f"--- Running Demos (Group: {group}, Count: {len(funcs_to_run)}) ---")
    for name, func in funcs_to_run:
        print(f"\n\033[91m--- Executing {name} ---\033[0m")
        res = func()
        if isinstance(res, go.Figure):
            figs.append(res)
        elif isinstance(res, tuple):
            for item in res:
                if isinstance(item, go.Figure):
                    figs.append(item)

    if save_html:
        filename = "all_demo_plots.html" if group == 'all' else f"demo_plots_{group}.html"
        with open(filename, "w") as f:
            f.write("<html><head><title>VibeVolts Demos</title></head><body>")
            f.write(f"<h1>VibeVolts Demo Plots ({group.capitalize()})</h1>")
            for i, fig in enumerate(figs):
                title = fig.layout.title.text if fig.layout.title else f"Plot {i+1}"
                f.write(f"<h2>{title}</h2>")
                f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))
            f.write("</body></html>")
        print(f"\n--- Demos complete. {len(figs)} plots saved to {filename} ---")
    else:
        print(f"\n--- Displaying {len(figs)} demo plots inline ---")
        for fig in figs:
            fig.show()


if __name__ == '__main__':
    run_all_demos()
