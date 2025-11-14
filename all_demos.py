import plotly.graph_objects as go
from IPython.display import display, HTML


# Import all the demo functions
from demo1 import demo1
from demo2 import demo2
from demo3 import demo3
from demo4 import demo4
from demo_fixedpoints import demo_fixedpoints
from demo_exclusion_table import demo_exclusion_table
from demo_pointing_plot import demo_pointing_plot

from demo_lambertian import demo_lambertian
from demo_sky_scan import demo_sky_scan
from demo_pointing_vectors import demo_pointing_vectors
from demo_pointing_sequence import demo_pointing_sequence
from demo_constellation import demo_constellation
from show_geo_search import show_geo_search
from demo_requiredIntegrationTime import demo_requiredIntegrationTime


def run_all_demos(save_html=False):
    """
    Runs all demo functions, and either shows them inline or saves them to a single HTML file.

    Args:
        save_html (bool): If True, saves plots to HTML. If False, displays plots inline.
    """
    demo_functions = [
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
        demo_constellation,
        show_geo_search,
        demo_requiredIntegrationTime,
    ]

    figs = [] # This is a list
    print("--- Running All Demos ---")
    for func in demo_functions:
        print(f"\n\033[91m--- Executing {func.__name__} ---\033[0m")
        res = func()
        if isinstance(res, go.Figure):
            figs.append(res)
            print("appending a single item")
        elif isinstance(res, tuple):
            print("I have a tuple")
            for fig in res:
                figs.append(fig)
                print("extracting an item from a tuple")

    if save_html:
        with open("all_demo_plots.html", "w") as f:
            f.write("<html><head><title>VibeVolts Demos</title></head><body>")
            f.write("<h1>VibeVolts Demo Plots</h1>")
            for i, fig in enumerate(figs):
                f.write(f"<h2>Demo {i+1}: {fig.layout.title.text if fig.layout.title else ''}</h2>")
                f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))
            f.write("</body></html>")
        print("\n--- All demos complete. Plots saved to all_demo_plots.html ---")
    else:
        print("\n--- Displaying all demo plots inline ---")
        print(len(figs)) 
        for fig in figs:
           fig.show() 

if __name__ == '__main__':
    run_all_demos()
