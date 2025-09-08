import plotly.graph_objects as go

# Import all the demo functions
from demo1 import demo1
from demo2 import demo2
from demo3 import demo3
from demo4 import demo4
from demo_fixedpoints import demo_fixedpoints
from demo_exclusion_table import demo_exclusion_table
from demo_pointing_plot import demo_pointing_plot
from demo_exclusion_debug_print import demo_exclusion_debug_print
from demo_lambertian import demo_lambertian
from demo_sky_scan import demo_sky_scan
from demo_pointing_vectors import demo_pointing_vectors
from demo_pointing_sequence import demo_pointing_sequence

def run_all_demos():
    """
    Runs all demo functions and displays their plots.
    This function is designed to be called from a Jupyter notebook
    or another script.
    """
    demo_functions = [
        demo1,
        demo2,
        demo3,
        demo4,
        demo_fixedpoints,
        demo_exclusion_table,
        demo_pointing_plot,
        demo_lambertian,
        demo_sky_scan,
        demo_pointing_vectors,
        demo_pointing_sequence,
    ]

    print("--- Running All Demos ---")
    for func in demo_functions:
        print(f"\n... Executing {func.__name__} ...")
        result = func()
        if isinstance(result, go.Figure):
            result.show()

    # Also run the non-plotting demo
    demo_exclusion_debug_print()

    print("\n--- All demos complete. ---")

if __name__ == '__main__':
    run_all_demos()
