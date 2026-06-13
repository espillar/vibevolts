import pandas as pd
import numpy as np
import plotly.graph_objects as go

class DataHandler:
    """
    A class to collect, manage, and export simulation results from VibeVolts.
    
    This handler facilitates collecting results from multiple calls to 
    `nextIntegration` (via `scandetectors`) and converting them into a 
    structured Pandas DataFrame for analysis, plotting, and export.

    Functions:
        - __init__(): Initializes an empty results collection.
        - add_results(results): Appends a dictionary of scan results to the collection.
        - get_dataframe(): Consolidates all results into a single Pandas DataFrame.
        - save_to_csv(filename): Exports the collected data to a CSV file.
        - save_to_parquet(filename): Exports the collected data to a Parquet file.
            import pandas as pd
            df = pd.read_parquet('my_data.parquet')
        - clear(): Resets the handler by removing all collected data.
    """

    def __init__(self):
        """Initializes an empty list to store result dictionaries."""
        self.results_list = []
        self._master_df = None

    def add_results(self, results: dict):
        """
        Appends a result dictionary from nextIntegration to the collection.
        
        Args:
            results (dict): The dictionary returned by nextIntegration/scandetectors.
                            Expected to contain 'time', 'sat_indices', 
                            'target_indices', 'signal', 'noise', and 'snr'.
        """
        # We only add if there are actual detections to keep the DataFrame concise.
        # If you need to track empty time steps, remove this check.
        if len(results['sat_indices']) > 0:
            # Converting to DataFrame immediately ensures consistent types
            df_step = pd.DataFrame(results)
            self.results_list.append(df_step)
            # Reset master_df so it's recalculated on next access
            self._master_df = None

    def get_dataframe(self) -> pd.DataFrame:
        """
        Combines all collected results into a single Pandas DataFrame.
        
        Returns:
            pd.DataFrame: A DataFrame with columns for time, satellite indices,
                          target indices, signal, noise, and SNR.
                          Returns an empty DataFrame if no results have been added.
        """
        if not self.results_list:
            return pd.DataFrame()
            
        if self._master_df is None:
            self._master_df = pd.concat(self.results_list, ignore_index=True)
            
        return self._master_df

    def save_to_csv(self, filename: str):
        """
        Saves the collected results to a CSV file.
        
        Args:
            filename (str): The path/name of the file to save (e.g., 'results.csv').
        """
        df = self.get_dataframe()
        if not df.empty:
            df.to_csv(filename, index=False)
            print(f"Results saved to {filename}")
        else:
            print("No data to save.")

    def save_to_parquet(self, filename: str):
        """
        Saves the collected results to a Parquet file (more efficient than CSV).
        
        Args:
            filename (str): The path/name of the file to save (e.g., 'results.parquet').
        """
        df = self.get_dataframe()
        if not df.empty:
            df.to_parquet(filename, index=False)
            print(f"Results saved to {filename}")
        else:
            print("No data to save.")

    def clear(self):
        """Clears all collected results to start a new run."""
        self.results_list = []
        self._master_df = None

    def calculate_gap_times(
        self, target_id: int = None, pooled: bool = False
    ) -> dict[int, np.ndarray] | np.ndarray:
        """
        Calculates the time gaps (in seconds) between consecutive observations of targets.

        If target_id is specified, returns a 1D numpy array of gap times (in seconds)
        for that target.
        If target_id is None and pooled is True, returns a single 1D numpy array containing
        the gap times of all targets concatenated.
        If target_id is None and pooled is False, returns a dictionary mapping target
        indices to their respective numpy arrays of gap times.

        Args:
            target_id (int, optional): The index of a specific target. Defaults to None.
            pooled (bool, optional): If True and target_id is None, returns a single
                                     concatenated array of all gap times. Defaults to False.

        Returns:
            np.ndarray or dict: Array of gap times or a dictionary mapping
                               target index -> numpy array of gap times.
        """
        df = self.get_dataframe()
        if df.empty:
            return np.array([], dtype=float) if target_id is not None else {}

        # Ensure sorted chronologically
        df_sorted = df.sort_values('time')

        if target_id is not None:
            target_df = df_sorted[df_sorted['target_indices'] == target_id]
            if len(target_df) < 2:
                return np.array([], dtype=float)
            time_diffs = target_df['time'].diff().dropna()
            return time_diffs.dt.total_seconds().values
        elif pooled:
            all_gaps = []
            for tid, group in df_sorted.groupby('target_indices'):
                if len(group) >= 2:
                    time_diffs = group['time'].diff().dropna()
                    all_gaps.extend(time_diffs.dt.total_seconds().tolist())
            return np.array(all_gaps, dtype=float)
        else:
            gaps_dict = {}
            for tid, group in df_sorted.groupby('target_indices'):
                if len(group) < 2:
                    gaps_dict[int(tid)] = np.array([], dtype=float)
                else:
                    time_diffs = group['time'].diff().dropna()
                    gaps_dict[int(tid)] = time_diffs.dt.total_seconds().values
            return gaps_dict

    def plot_gap_times_histogram(
        self,
        target_id: int = None,
        bins: int | str | None = "auto",
        show_plot: bool = True
    ) -> go.Figure:
        """
        Generates and optionally displays a Plotly histogram of the target gap times.

        If target_id is specified, plots the histogram of gap times for that target.
        If target_id is None, plots the histogram of pooled gap times across all targets.

        Args:
            target_id (int, optional): The index of a specific target. Defaults to None.
            bins (int or str, optional): Specification of histogram bins (e.g. 'auto', 10).
                                         Passed to Plotly's nbinsx. Defaults to 'auto'.
            show_plot (bool, optional): If True, displays the plot. Defaults to True.

        Returns:
            plotly.graph_objects.Figure: The generated Plotly figure object.
        """
        if target_id is not None:
            gaps = self.calculate_gap_times(target_id=target_id)
            title = f"Interobservation Gap Times for Target {target_id}"
        else:
            gaps = self.calculate_gap_times(pooled=True)
            title = "Pooled Interobservation Gap Times (All Targets)"

        if len(gaps) == 0:
            fig = go.Figure()
            fig.update_layout(
                title_text=f"{title} - No Gap Data Available",
                template="plotly_white"
            )
            if show_plot:
                fig.show()
            return fig

        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=gaps,
            nbinsx=bins if isinstance(bins, int) else None,
            marker=dict(
                color='#3B82F6',
                line=dict(color='white', width=1)
            ),
            opacity=0.85,
            name="Gap Times"
        ))

        fig.update_layout(
            title_text=title,
            xaxis_title="Gap Time (seconds)",
            yaxis_title="Observation Count",
            template="plotly_white",
            bargap=0.05,
            hovermode="x unified"
        )

        if show_plot:
            fig.show()

        return fig
