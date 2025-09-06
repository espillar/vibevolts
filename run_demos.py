import plotly.graph_objects as go
from datetime import datetime

# Import all the demo functions
from demos.demo1 import demo1
from demos.demo2 import demo2
from demos.demo3 import demo3
from demos.demo4 import demo4
from demos.demo_fixedpoints import demo_fixedpoints
from demos.demo_exclusion_table import demo_exclusion_table
from demos.demo_pointing_plot import demo_pointing_plot
from demos.demo_exclusion_debug_print import demo_exclusion_debug_print

def export_all_plots_to_html(output_filename: str = "demo_plots.html"):
    """
    Runs all demo functions that produce plots and saves them to a single
    self-contained HTML file.

    Args:
        output_filename: The name of the HTML file to create.
    """
    demo_functions = [
        demo1,
        demo2,
        demo3,
        demo4,
        demo_fixedpoints,
        demo_exclusion_table,
        demo_pointing_plot,
    ]

    print("--- Generating All Demo Figures for HTML Export ---")
    figures = []
    for func in demo_functions:
        print(f"\n... Executing {func.__name__} ...")
        result = func()
        if isinstance(result, go.Figure):
            figures.append(result)

    print(f"\n--- Exporting All Plots to {output_filename} ---")
    with open(output_filename, 'w') as f:
        f.write("<html><head><title>VibeVolts Demo Plots</title></head><body>\n")
        f.write("<h1>VibeVolts Demonstration Plots</h1>\n")
        f.write(f"<p>Generated on: {datetime.now().isoformat()}</p>\n")

        for i, fig in enumerate(figures):
            fig_title = fig.layout.title.text or f"Figure {i+1}"
            f.write(f"<hr><h2>{fig_title}</h2>\n")
            f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))

        f.write("</body></html>\n")

    print(f"--- Export complete. See {output_filename} for results. ---")

def run_all_demos_interactive():
    """
    Runs all demo functions and displays their plots interactively.
    """
    demo_functions = [
        demo1,
        demo2,
        demo3,
        demo4,
        demo_fixedpoints,
        demo_exclusion_table,
        demo_pointing_plot,
    ]

    print("--- Running All Demos for Interactive Display ---")
    figures = []
    for func in demo_functions:
        print(f"\n... Executing {func.__name__} ...")
        result = func()
        if isinstance(result, go.Figure):
            figures.append(result)

    print("\n--- Displaying Plots Interactively ---")
    print("Each plot will open in a new browser tab/window.")
    for fig in figures:
        fig.show()

    # Also run the non-plotting demo
    demo_exclusion_debug_print()

    print("\n--- All demos complete. ---")


if __name__ == '__main__':
    # To run interactively:
    # python run_demos.py
    #
    # To generate the HTML report:
    # python -c "import run_demos; run_demos.export_all_plots_to_html()"

    run_all_demos_interactive()
