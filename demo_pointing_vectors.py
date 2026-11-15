import plotly.graph_objects as go
from fibonacciSearch import pointing_vectors, plot_vectors_on_sphere
from plotting_3d import tag_demo_figure

def demo_pointing_vectors() -> go.Figure:
    """
    Demonstrates the generation and plotting of pointing vectors.

    Returns:
        A Plotly figure object.
    """
    print("--- Running Pointing Vectors Demo ---")
    num_vectors = 1000
    print(f"Generating {num_vectors} pointing vectors...")
    vectors = pointing_vectors(num_vectors)
    print("Plotting vectors on a sphere...")
    fig = plot_vectors_on_sphere(vectors, f"{num_vectors} Pointing Vectors")
    return tag_demo_figure(fig, "demo_pointing_vectors")

if __name__ == '__main__':
    fig = demo_pointing_vectors()
    fig.show()
