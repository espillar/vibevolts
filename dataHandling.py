import pandas as pd
import numpy as np

class DataHandler:
    """
    A class to collect and manage simulation results from VibeVolts.
    
    This handler facilitates collecting results from multiple calls to 
    nextIntegration and converting them into a structured Pandas DataFrame
    for analysis, plotting, and export.
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
