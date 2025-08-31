import plotly.graph_objects as go

from vibevolts_demo import (
    demo1,
    demo2,
    demo3,
    demo4,
    demo_fixedpoints,
    demo_exclusion_table,
    demo_pointing_plot,
)

def generate_demo_html_report():
    """
    Runs all plotting demos and saves the output to a single HTML file.
    """
    print("Generating HTML report for all demos...")

    # A list of all demo functions that produce plots
    plotting_demos = [
        ("Demo 1: Satellite Positions", demo1),
        ("Demo 2: Satellite and Celestial Positions", demo2),
        ("Demo 3: Single LEO Satellite Trajectory", demo3),
        ("Demo 4: Single GEO Satellite Trajectory", demo4),
        ("Demo Fixed Points", demo_fixedpoints),
        ("Demo Exclusion Table", demo_exclusion_table),
        ("Demo Pointing Plot", demo_pointing_plot),
    ]

    # Collect all the figure objects
    figures = []
    for name, demo_func in plotting_demos:
        print(f"Running: {name}")
        fig = demo_func(show_plot=False)
        # Add a title to the figure layout if it doesn't have one
        if fig.layout.title.text is None or fig.layout.title.text == '':
            fig.update_layout(title_text=name)
        figures.append(fig)

    # Write all figures to a single HTML file
    report_filename = "demo_plots.html"
    print(f"Writing report to {report_filename}...")
    with open(report_filename, 'w') as f:
        f.write("<html><head><title>Vibevolts Demo Report</title></head><body>")
        f.write("<h1>Vibevolts Demo Report</h1>")
        # Include plotly.js from CDN
        f.write("<script src='https://cdn.plot.ly/plotly-latest.min.js'></script>")

        for i, fig in enumerate(figures):
            # Convert figure to an HTML div
            fig_html = fig.to_html(full_html=False, include_plotlyjs=False)
            f.write(f"<h2>{fig.layout.title.text}</h2>")
            f.write(fig_html)
            if i < len(figures) - 1:
                f.write("<hr>") # Add a horizontal rule between plots

        f.write("</body></html>")

    print("Report generation complete.")

if __name__ == '__main__':
    generate_demo_html_report()
