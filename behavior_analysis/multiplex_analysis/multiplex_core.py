import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import datetime
import json
import re


class MultiplexTrial:
    """
    Core class for analyzing individual multiplex trials with automatic CS+ detection
    and side-switching support.
    """
    
    def __init__(self) -> None:
        self.raw_data = None
        self.processed_data = None
        self.file_path = None

    def load_data(self, data_path):
        """
        Load a Multiplex log file (.csv) and convert it to pandas dataframe
        """
        self.file_path = data_path
        self.raw_data = pd.read_csv(data_path)
        self.processed_data = self.raw_data.copy()

    def select_test_period(self):
        """
        Return the period that corresponds to the 'Test' phase of the trial.
        """
        df = self.processed_data
        test_df = df[df['experiment_step'] == 'Test']
        test_df = test_df.filter(regex=r'^(Timestamp|chamber_\d+_loc)$')
        test_df.set_index('Timestamp', inplace=True)
        return test_df

    def select_initial_valence_period(self):
        """
        Return the period that corresponds to the 'Initial Valence' phase of the trial.
        """
        df = self.processed_data
        initial_valence_df = df[df['experiment_step'] == 'Initial Valence']
        initial_valence_df = initial_valence_df.filter(regex=r'^(Timestamp|chamber_\d+_loc)$')
        initial_valence_df.set_index('Timestamp', inplace=True)
        return initial_valence_df

    def filter_by_num_choices(self, midline_borders, threshold=1, filter='both'):
        """
        Filter flies based on the number of times they have crossed the midline.
        Values indicate how well the fly has explored the chamber during the initial valence/test period.
        """
        # Store filter parameters as instance variables for use in other methods
        self.midline_borders = midline_borders
        self.filter_threshold = threshold
        valence_df = self.select_initial_valence_period()
        test_df = self.select_test_period()

        df_mapping = {
            'both': [('valence_df', valence_df), ('test_df', test_df)],
            'test': [('test_df', test_df)],
            'valence': [('valence_df', valence_df)]
        }

        filtered_dfs = {}
        for key, df in df_mapping.get(filter, []):
            filtered_dfs[f"filtered_{key}"] = self.filter_by_midline(df, midline_borders=midline_borders, threshold=threshold)

        if filter == 'both':
            common_columns = filtered_dfs['filtered_valence_df'].columns.intersection(filtered_dfs['filtered_test_df'].columns)
            self.processed_data = filtered_dfs['filtered_valence_df'][common_columns], filtered_dfs['filtered_test_df'][common_columns]
        elif filter in ['test', 'valence']:
            key = f"filtered_{filter}_df"
            self.processed_data = filtered_dfs[key][filtered_dfs[key].columns]

    def filter_by_midline(self, df, midline_borders, threshold=1):
        """
        Filter out flies that have not crossed the midline threshold number of times.
        """
        crossing_counts = {}
        for col in df.columns:
            values = df[col]
            crossings = (
                ((values.shift(1) < midline_borders) & (values >= midline_borders)) | 
                ((values.shift(1) > midline_borders) & (values <= midline_borders)) |
                ((values.shift(1) > -midline_borders) & (values <= -midline_borders)) |
                ((values.shift(1) < -midline_borders) & (values >= -midline_borders))
            )
            crossing_counts[col] = crossings.sum()
        
        filtered_columns = [col for col, count in crossing_counts.items() if count >= threshold]
        filtered_df = df[filtered_columns]
        return filtered_df

    @staticmethod
    def time_spent(df, determine_side=10, sampling_rate=0.1):
        """
        Calculate time spent on each side of the chamber.
        Returns a dataframe showing for each fly the time spent on each side.
        """
        def process_counts(counts):
            df_transposed = counts.reset_index().T
            df_transposed.columns = df_transposed.iloc[0]
            return df_transposed.drop(df_transposed.index[0])

        mask_greater = df > determine_side
        mask_less = df < -determine_side

        count_greater = mask_greater.sum() * sampling_rate
        count_less = mask_less.sum() * sampling_rate

        count_greater_processed = process_counts(count_greater)
        count_less_processed = process_counts(count_less)

        df_combined = pd.concat([count_greater_processed, count_less_processed])
        df_combined.index = ['right_side', 'left_side']
        return df_combined

    def identify_cs_plus_odor(self):
        """
        Identify CS+ odor based on conditioning type and experimental rules:
        - Operant: Choice between MOIL and another odor → the other odor is CS+
        - Classical: Odor paired with electrical shock → that odor is CS+
        Returns: ('odor_name', 'side') indicating which odor and side is CS+
        """
        learning_shock_df = self.raw_data[self.raw_data['experiment_step'] == 'Learning Shock']
        
        if learning_shock_df.empty:
            print("Warning: No Learning Shock phase found in data")
            return ('mch', 'left')
        
        # Get protocol type from metadata
        metadata_path = os.path.join(os.path.dirname(self.file_path), 'experiment_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            protocol = metadata.get('protocol', '').lower()
        else:
            print("Warning: No metadata file found, defaulting to operant conditioning")
            protocol = 'operant'  # Default assumption
        
        # Get odor status during Learning Shock stage
        odor_columns = ['mch_right_status', 'oct_right_status', 'moil_right_status', 
                       'mch_left_status', 'oct_left_status', 'moil_left_status']
        odor_data = learning_shock_df[odor_columns]
        
        mch_left = odor_data['mch_left_status'].iloc[0]
        mch_right = odor_data['mch_right_status'].iloc[0]
        moil_left = odor_data['moil_left_status'].iloc[0]
        moil_right = odor_data['moil_right_status'].iloc[0]
        oct_left = odor_data['oct_left_status'].iloc[0]
        oct_right = odor_data['oct_right_status'].iloc[0]
        
        # Determine CS+ based on conditioning type
        if 'operant' in protocol:
            # Operant: Choice between MOIL and another odor → the other odor is CS+
            if (moil_left == 1 or moil_right == 1):
                if (mch_left == 1 or mch_right == 1):
                    cs_plus_odor = 'mch'
                    cs_plus_side = 'left' if mch_left == 1 else 'right'
                elif (oct_left == 1 or oct_right == 1):
                    cs_plus_odor = 'oct'
                    cs_plus_side = 'left' if oct_left == 1 else 'right'
                else:
                    print("Warning: MOIL present but no other odor found")
                    return ('mch', 'left')
            else:
                print("Warning: Operant conditioning but no MOIL found")
                return ('mch', 'left')
        
        elif 'classical' in protocol:
            # Classical: Find which odor is active during shock stage
            # Check all active odors and determine which one is CS+
            active_odors = []
            if mch_left == 1: active_odors.append(('mch', 'left'))
            if mch_right == 1: active_odors.append(('mch', 'right'))
            if oct_left == 1: active_odors.append(('oct', 'left'))
            if oct_right == 1: active_odors.append(('oct', 'right'))
            if moil_left == 1: active_odors.append(('moil', 'left'))
            if moil_right == 1: active_odors.append(('moil', 'right'))
            
            if len(active_odors) == 1:
                cs_plus_odor, cs_plus_side = active_odors[0]
            else:
                # Multiple odors active - try to determine CS+ from protocol name or experiment setup
                print("Warning: Multiple odors active during classical conditioning")
                
                # Try to infer CS+ from protocol name
                if 'oct' in protocol.lower():
                    # Protocol mentions OCT, so OCT is likely CS+
                    if oct_left == 1:
                        cs_plus_odor, cs_plus_side = ('oct', 'left')
                    elif oct_right == 1:
                        cs_plus_odor, cs_plus_side = ('oct', 'right')
                    else:
                        cs_plus_odor, cs_plus_side = ('mch', 'left')
                elif 'mch' in protocol.lower():
                    # Protocol mentions MCH, so MCH is likely CS+
                    if mch_left == 1:
                        cs_plus_odor, cs_plus_side = ('mch', 'left')
                    elif mch_right == 1:
                        cs_plus_odor, cs_plus_side = ('mch', 'right')
                    else:
                        cs_plus_odor, cs_plus_side = ('mch', 'left')
                else:
                    # If protocol name doesn't help, look for the odor that's active on both sides
                    # (indicating it's the main conditioning odor)
                    if oct_left == 1 and oct_right == 1:
                        cs_plus_odor, cs_plus_side = ('oct', 'left')  # Default to left side
                    elif mch_left == 1 and mch_right == 1:
                        cs_plus_odor, cs_plus_side = ('mch', 'left')  # Default to left side
                    elif active_odors:
                        cs_plus_odor, cs_plus_side = active_odors[0]
                    else:
                        cs_plus_odor, cs_plus_side = ('mch', 'left')
        
        else:
            print("Warning: Unknown protocol type, defaulting to operant logic")
            # Default to operant logic
            if (moil_left == 1 or moil_right == 1) and (mch_left == 1 or mch_right == 1):
                cs_plus_odor = 'mch'
                cs_plus_side = 'left' if mch_left == 1 else 'right'
            else:
                return ('mch', 'left')
        
        print(f"CS+ identified as {cs_plus_odor.upper()} on {cs_plus_side.upper()} side (protocol: {protocol})")
        return (cs_plus_odor, cs_plus_side)

    def detect_cs_plus_side_in_phase(self, phase_data, cs_plus_odor):
        """
        Detect which side the CS+ odor appears on in a specific phase
        Returns: 'left' or 'right'
        """
        odor_columns = ['mch_right_status', 'oct_right_status', 'moil_right_status', 
                       'mch_left_status', 'oct_left_status', 'moil_left_status']
        
        if not all(col in phase_data.columns for col in odor_columns):
            print(f"Warning: Odor status columns not found in phase data")
            return 'left'  # Default fallback
        
        odor_data = phase_data[odor_columns]
        
        # Check which side has the CS+ odor active
        left_odor_col = f'{cs_plus_odor}_left_status'
        right_odor_col = f'{cs_plus_odor}_right_status'
        
        if left_odor_col in odor_data.columns and right_odor_col in odor_data.columns:
            left_status = odor_data[left_odor_col].iloc[0] if not odor_data.empty else 0
            right_status = odor_data[right_odor_col].iloc[0] if not odor_data.empty else 0
            
            if left_status == 1 and right_status == 0:
                return 'left'
            elif right_status == 1 and left_status == 0:
                return 'right'
            elif left_status == 1 and right_status == 1:
                print(f"Warning: {cs_plus_odor} active on both sides, defaulting to left")
                return 'left'
            else:
                print(f"Warning: {cs_plus_odor} not active in this phase, defaulting to left")
                return 'left'
        else:
            print(f"Warning: {cs_plus_odor} status columns not found, defaulting to left")
            return 'left'

    def save_analysis_metadata(self, analysis_method, save_to_file=True, **kwargs):
        """
        Save analysis metadata including all parameters, version info, and analysis details.
        Creates a comprehensive JSON file for reproducibility and version control.
        Only saves if save_to_file is True
        
        Parameters:
        -----------
        analysis_method : str
            The analysis method used ('time', 'snapshot', 'learning_valence', 'valence_habituation')
        save_to_file : bool
            Whether to save the metadata file (default: True)
        **kwargs : dict
            All analysis parameters and metadata
        """
        if not save_to_file:
            return
        # Create metadata dictionary
        metadata = {
            "analysis_info": {
                "method": analysis_method,
                "timestamp": datetime.datetime.now().isoformat(),
                "version": "2.0.0",
                "description": "Multiplex learning behavior analysis for Drosophila"
            },
            "parameters": kwargs,
            "data_info": {
                "trial_folder": getattr(self, 'trial_folder', 'Unknown'),
                "experiment_step": getattr(self, 'experiment_step', 'Unknown'),
                "total_flies_loaded": len(self.raw_data.columns) - 1 if hasattr(self, 'raw_data') and self.raw_data is not None else 0,
                "filtering_applied": hasattr(self, 'midline_borders') and hasattr(self, 'filter_threshold')
            }
        }
        
        # Add method-specific information
        if analysis_method == 'time':
            metadata["analysis_info"]["description"] = "Time-based individual fly analysis - measures time spent on each side"
            metadata["data_info"]["filtering_details"] = {
                "midline_borders": getattr(self, 'midline_borders', 'Not set'),
                "filter_threshold": getattr(self, 'filter_threshold', 'Not set'),
                "filter_phase": kwargs.get('filter_phase', 'Not specified')
            }
        elif analysis_method == 'snapshot':
            metadata["analysis_info"]["description"] = "Snapshot population-level analysis - measures fly positions at end of phases"
            metadata["data_info"]["snapshot_details"] = {
                "seconds_before_end": 5,
                "snapshot_description": "Takes snapshot 5 seconds before end of each phase"
            }
        elif analysis_method == 'learning_valence':
            metadata["analysis_info"]["description"] = "Learning valence analysis - measures changes in valence with a learning session"
            metadata["data_info"]["filtering_details"] = {
                "midline_borders": getattr(self, 'midline_borders', 'Not set'),
                "filter_threshold": getattr(self, 'filter_threshold', 'Not set')
            }
        elif analysis_method == 'valence_habituation':
            metadata["analysis_info"]["description"] = "Valence habituation analysis - measures valence across multiple repeated exposures"
            metadata["data_info"]["filtering_details"] = {
                "midline_borders": getattr(self, 'midline_borders', 'Not set'),
                "filter_threshold": getattr(self, 'filter_threshold', 'Not set')
            }
            if 'valence_step_numbers' in kwargs:
                metadata["data_info"]["valence_steps"] = kwargs['valence_step_numbers']
        
        # Use the timestamped output folder created by save_analysis_results
        if hasattr(self, 'output_folder'):
            output_folder = self.output_folder
        else:
            # Fallback: create timestamped output folder
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_folder = os.path.join(os.path.dirname(self.file_path), f'output_{timestamp}')
            os.makedirs(output_folder, exist_ok=True)
        
        # Save metadata to JSON file
        metadata_filename = f"analysis_metadata_{analysis_method}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        metadata_path = os.path.join(output_folder, metadata_filename)
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Analysis metadata saved to: {metadata_path}")
        return metadata_path

    def save_analysis_results(self, results_df, save_to_file=True):
        """
        Save analysis results to CSV file in timestamped output folder
        Only saves if save_to_file is True
        """
        if not save_to_file:
            return
        
        # Create timestamped output folder
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_folder = os.path.join(os.path.dirname(self.file_path), f'output_{timestamp}')
        os.makedirs(output_folder, exist_ok=True)
        
        # Create filename
        base_name = os.path.splitext(os.path.basename(self.file_path))[0]
        csv_filename = f"{base_name}_learned_index_analysis_{timestamp}.csv"
        csv_path = os.path.join(output_folder, csv_filename)
        
        # Save CSV
        results_df.to_csv(csv_path, index=False)
        print(f"Analysis results saved to: {csv_path}")
        
        # Store the output folder path for metadata saving
        self.output_folder = output_folder

    def plot_trial(self):
        """
        Plot individual fly trajectories during test period
        """
        df = self.processed_data

        # Select only the test period (may vary if I will use other protocols)
        test_df = df[(df['LEFTODOR2'] == 1) & (df['RIGHTODOR1'] == 1)]

        # Extract time and location columns of flies
        test_df_location_only = test_df.filter(regex=r'^(Timestamp|cX\d{3})$')

        # Replace 0 values with NaN, to not include during the calculation 0, which happens when the system is not detectiing the flies
        test_df_location_only.replace(0, np.nan, inplace=True)

        # Set the 'Time' column as the index
        test_df_location_only.set_index('Time', inplace=True)
        # Plot data
        cX_columns = [col for col in test_df_location_only.columns if col.startswith('cX')]

        # Set up the figure and axis for subplots
        fig, axes = plt.subplots(len(cX_columns), 1, figsize=(10, 0.5 * len(cX_columns)), sharex=True)

        # If there's only one subplot, wrap the axes in a list for consistency
        if len(cX_columns) == 1:
            axes = [axes]

        # Plot each cX column in its own subplot
        for i, col in enumerate(cX_columns):
            sns.lineplot(ax=axes[i], x=test_df_location_only.index, y=test_df_location_only[col])
            axes[i].set_ylim(-1, 1)  # Set Y-axis limits from -1 to 1
            axes[i].set_ylabel('')  # Remove the Y-axis label
            axes[i].set_yticks([-1, 0, 1])  # Optionally, set specific y-ticks
            
            # Place the column name on the left side
            axes[i].annotate(col, xy=(0, 0.5), xytext=(-axes[i].yaxis.labelpad - 10, 0),
                             xycoords=axes[i].yaxis.label, textcoords='offset points',
                             size='large', ha='right', va='center', rotation=0)

            axes[i].grid(True)

        # Set the common x-label
        axes[-1].set_xlabel('Time')

        # Adjust the layout to prevent overlap
        plt.tight_layout()

        # Display the plot
        plt.show()

    def _detect_valence_odor(self, phase_data):
        """
        Detect which odor is being tested in the valence experiment.
        """
        odor_columns = ['mch_right_status', 'oct_right_status', 'moil_right_status', 
                       'mch_left_status', 'oct_left_status', 'moil_left_status']
        
        if not all(col in phase_data.columns for col in odor_columns):
            print("Warning: Odor status columns not found, defaulting to MCH")
            return 'mch'
        
        odor_data = phase_data[odor_columns]
        
        # Check which odors are active
        active_odors = []
        if odor_data['mch_left_status'].iloc[0] == 1 or odor_data['mch_right_status'].iloc[0] == 1:
            active_odors.append('mch')
        if odor_data['oct_left_status'].iloc[0] == 1 or odor_data['oct_right_status'].iloc[0] == 1:
            active_odors.append('oct')
        if odor_data['moil_left_status'].iloc[0] == 1 or odor_data['moil_right_status'].iloc[0] == 1:
            active_odors.append('moil')
        
        if len(active_odors) == 1:
            return active_odors[0]
        elif len(active_odors) > 1:
            print(f"Warning: Multiple odors active, defaulting to {active_odors[0]}")
            return active_odors[0]
        else:
            print("Warning: No odors detected, defaulting to MCH")
            return 'mch'
