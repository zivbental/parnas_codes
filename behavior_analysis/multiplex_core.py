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

    def analyse_time(self, determine_side=None, min_valence_seconds=30, save_to_file=True):
        """
        Calculate learned index with automatic CS+/CS- detection and side-switching handling
        
        Parameters:
        -----------
        determine_side : float, optional
            Threshold for determining which side flies are on (0.0 to 100.0). Default: 10
        min_valence_seconds : float, optional
            Minimum number of seconds a fly must spend in initial valence period to be included in analysis. Default: 30
        save_to_file : bool, optional
            Whether to save results and metadata to files. Default: True
        """
        # 1. Identify CS+ odor and side from Learning Shock phase
        cs_plus_odor, learning_cs_side = self.identify_cs_plus_odor()
        
        # 2. Get phase data from raw_data (before filtering)
        valence_phase_data = self.raw_data[self.raw_data['experiment_step'] == 'Initial Valence']
        test_phase_data = self.raw_data[self.raw_data['experiment_step'] == 'Test']
        
        # 3. Detect CS+ side in each phase
        valence_cs_side = self.detect_cs_plus_side_in_phase(valence_phase_data, cs_plus_odor)
        test_cs_side = self.detect_cs_plus_side_in_phase(test_phase_data, cs_plus_odor)
        
        # 4. Check if sides switched
        sides_switched = (valence_cs_side != test_cs_side)
        print(f"CS+ side in Initial Valence: {valence_cs_side.upper()}")
        print(f"CS+ side in Test: {test_cs_side.upper()}")
        print(f"Sides switched: {sides_switched}")
        
        # 5. Calculate time spent for both phases using filtered data
        if determine_side is None:
            determine_side = 10  # Default value
        valence_df = self.time_spent(self.processed_data[0], determine_side)
        test_df = self.time_spent(self.processed_data[1], determine_side)

        # 6. Apply existing filtering logic
        valence_denominator = valence_df.iloc[1] + valence_df.iloc[0]
        test_denominator = test_df.iloc[0] + test_df.iloc[1]
        combined_mask = (valence_denominator != 0) & (test_denominator != 0)

        filtered_valence_df = valence_df.loc[:, combined_mask]
        filtered_test_df = test_df.loc[:, combined_mask]

        # Filter out flies with < min_valence_seconds in initial valence
        initial_val_filter = filtered_valence_df.iloc[1] >= min_valence_seconds
        filtered_valence_df = filtered_valence_df.loc[:, initial_val_filter]
        filtered_test_df = filtered_test_df.loc[:, initial_val_filter]

        # 7. Calculate CS+ preference for each phase (adaptive to actual side)
        if valence_cs_side == 'left':
            initial_val = (filtered_valence_df.iloc[1]) / (filtered_valence_df.iloc[0] + filtered_valence_df.iloc[1])  # left/total
        else:
            initial_val = (filtered_valence_df.iloc[0]) / (filtered_valence_df.iloc[0] + filtered_valence_df.iloc[1])  # right/total
            
        if test_cs_side == 'left':
            end_valence = (filtered_test_df.iloc[1]) / (filtered_test_df.iloc[0] + filtered_test_df.iloc[1])  # left/total
        else:
            end_valence = (filtered_test_df.iloc[0]) / (filtered_test_df.iloc[0] + filtered_test_df.iloc[1])  # right/total

        # 8. Calculate learned index
        learned_index = (end_valence - initial_val) * 100
        
        # 9. Create enhanced results dataframe
        results_data = []
        for i, col in enumerate(filtered_valence_df.columns):
            fly_id = col.replace('chamber_', '').replace('_loc', '')
            results_data.append({
                'fly_id': fly_id,
                'cs_plus_odor': cs_plus_odor,
                'initial_valence_cs_side': valence_cs_side,
                'test_cs_side': test_cs_side,
                'sides_switched': sides_switched,
                'initial_valence_cs_preference': initial_val.iloc[i] * 100,  # Convert to percentage
                'test_cs_preference': end_valence.iloc[i] * 100,  # Convert to percentage
                'learned_index': learned_index.iloc[i],
                'valid_fly': True
            })
        
        results_df = pd.DataFrame(results_data)
        
        # 10. Print summary
        print(f"\nCS+ Odor: {cs_plus_odor.upper()}")
        print(f"Number of valid flies: {len(results_df)}")
        print(f"Mean learned index: {learned_index.mean():.2f}")
        print(f"Std learned index: {learned_index.std():.2f}")
        
        # 11. Save to CSV (only if save_to_file is True)
        self.save_analysis_results(results_df, save_to_file)
        
        # 12. Save analysis metadata (only if save_to_file is True)
        if save_to_file:
            metadata_params = {
                'determine_side': determine_side,
                'min_valence_seconds': min_valence_seconds,
                'cs_plus_odor': cs_plus_odor,
                'valence_cs_side': valence_cs_side,
                'test_cs_side': test_cs_side,
                'sides_switched': sides_switched,
                'num_valid_flies': len(results_df),
                'mean_learned_index': learned_index.mean(),
                'std_learned_index': learned_index.std()
            }
            self.save_analysis_metadata('time', save_to_file, **metadata_params)
        
        return results_df

    def analyse_snapshot(self, determine_side=None, time_window=None, save_to_file=True):
        """
        Calculate learned index using population snapshot at end of phases.
        Returns population-level PI rather than individual fly metrics.
        
        This method takes a snapshot of fly positions N seconds before the end of 
        Initial Valence and Test periods, counts flies on each side, and calculates
        Preference Index (PI) for each phase. The learned index is the difference
        between Test PI and Initial Valence PI.
        
        Parameters:
        -----------
        determine_side : float, optional
            Threshold for determining which side flies are on (0.0 to 100.0). Default: 60
        time_window : list of two floats, optional
            Time range [start_seconds, end_seconds] from phase start to average positions. Default: [0, 5]
        save_to_file : bool, optional
            Whether to save results and metadata to files. Default: True
        """
        # Set default time window if not provided
        if time_window is None:
            time_window = [0, 5]
        
        # 1. Identify CS+ odor and side from Learning Shock phase
        cs_plus_odor, learning_cs_side = self.identify_cs_plus_odor()
        
        # 2. Get phase data from raw_data (before filtering)
        valence_phase_data = self.raw_data[self.raw_data['experiment_step'] == 'Initial Valence']
        test_phase_data = self.raw_data[self.raw_data['experiment_step'] == 'Test']
        
        # 3. Detect CS+ side in each phase
        valence_cs_side = self.detect_cs_plus_side_in_phase(valence_phase_data, cs_plus_odor)
        test_cs_side = self.detect_cs_plus_side_in_phase(test_phase_data, cs_plus_odor)
        
        # 4. Check if sides switched
        sides_switched = (valence_cs_side != test_cs_side)
        print(f"CS+ side in Initial Valence: {valence_cs_side.upper()}")
        print(f"CS+ side in Test: {test_cs_side.upper()}")
        print(f"Sides switched: {sides_switched}")
        
        # 5. Get snapshot data from time window of each phase (configurable)
        valence_snapshot = self._get_snapshot_data(valence_phase_data, time_window)
        test_snapshot = self._get_snapshot_data(test_phase_data, time_window)
        
        # 6. Apply filtering to get valid flies (same as analyse_time)
        if hasattr(self, 'processed_data') and len(self.processed_data) >= 2:
            # Use the filtered flies from filter_by_num_choices
            valid_flies = set(self.processed_data[0].columns) & set(self.processed_data[1].columns)
            valence_snapshot = valence_snapshot[list(valid_flies)]
            test_snapshot = test_snapshot[list(valid_flies)]
        else:
            print("Warning: No filtering applied, using all flies")
        
        # 7. Calculate fly counts on each side for each phase
        valence_counts = self._count_flies_by_side(valence_snapshot, valence_cs_side, determine_side)
        test_counts = self._count_flies_by_side(test_snapshot, test_cs_side, determine_side)
        
        # 8. Calculate Preference Index (PI) for each phase
        pi_initial_valence = self._calculate_pi(valence_counts)
        pi_test = self._calculate_pi(test_counts)
        
        # 9. Calculate learned index
        learned_index = pi_test - pi_initial_valence
        
        # 10. Create results dataframe (trial-level, not fly-level)
        results_data = [{
            'trial_id': os.path.splitext(os.path.basename(self.file_path))[0],
            'cs_plus_odor': cs_plus_odor,
            'initial_valence_cs_side': valence_cs_side,
            'test_cs_side': test_cs_side,
            'sides_switched': sides_switched,
            'initial_valence_cs_preference': pi_initial_valence,
            'test_cs_preference': pi_test,
            'learned_index': learned_index,
            'num_flies': len(valence_snapshot.columns),
            'analysis_method': 'snapshot'
        }]
        
        results_df = pd.DataFrame(results_data)
        
        # 11. Print summary
        print(f"\nCS+ Odor: {cs_plus_odor.upper()}")
        print(f"Number of flies in snapshot: {len(valence_snapshot.columns)}")
        print(f"Initial Valence - CS+ side ({valence_cs_side.upper()}): {valence_counts['flies_on_cs_plus']} flies")
        print(f"Initial Valence - CS- side ({'RIGHT' if valence_cs_side == 'left' else 'LEFT'}): {valence_counts['flies_on_cs_minus']} flies")
        print(f"Test - CS+ side ({test_cs_side.upper()}): {test_counts['flies_on_cs_plus']} flies")
        print(f"Test - CS- side ({'RIGHT' if test_cs_side == 'left' else 'LEFT'}): {test_counts['flies_on_cs_minus']} flies")
        print(f"Initial Valence PI: {pi_initial_valence:.2f}")
        print(f"Test PI: {pi_test:.2f}")
        print(f"Learned Index: {learned_index:.2f}")
        
        # 12. Save to CSV (only if save_to_file is True)
        self.save_analysis_results(results_df, save_to_file)
        
        # 13. Save analysis metadata (only if save_to_file is True)
        if save_to_file:
            metadata_params = {
                'determine_side': determine_side,
                'time_window': time_window,
                'cs_plus_odor': cs_plus_odor,
                'valence_cs_side': valence_cs_side,
                'test_cs_side': test_cs_side,
                'sides_switched': sides_switched,
                'num_flies_in_snapshot': valence_counts['total_flies'],
                'valence_pi': pi_initial_valence,
                'test_pi': pi_test,
                'learned_index': learned_index,
                'valence_cs_plus_count': valence_counts['flies_on_cs_plus'],
                'valence_cs_minus_count': valence_counts['flies_on_cs_minus'],
                'test_cs_plus_count': test_counts['flies_on_cs_plus'],
                'test_cs_minus_count': test_counts['flies_on_cs_minus']
            }
            self.save_analysis_metadata('snapshot', save_to_file, **metadata_params)
        
        return results_df
    
    def _get_snapshot_data(self, phase_data, time_window=[0, 5]):
        """
        Get averaged snapshot data from a time window within a phase.
        
        Parameters:
        -----------
        phase_data : DataFrame
            Data for the phase
        time_window : list of two floats
            Time range [start_seconds, end_seconds] from phase start
        """
        # Get location columns only
        location_cols = [col for col in phase_data.columns if 'chamber_' in col and '_loc' in col]
        snapshot_data = phase_data[['Timestamp'] + location_cols].copy()
        
        # Convert timestamp to datetime
        if not pd.api.types.is_datetime64_any_dtype(snapshot_data['Timestamp']):
            snapshot_data['Timestamp'] = pd.to_datetime(snapshot_data['Timestamp'])
        
        # Get phase start time and calculate absolute time range
        start_time = snapshot_data['Timestamp'].min()
        window_start = start_time + pd.Timedelta(seconds=time_window[0])
        window_end = start_time + pd.Timedelta(seconds=time_window[1])
        
        # Filter to time window
        windowed_data = snapshot_data[
            (snapshot_data['Timestamp'] >= window_start) & 
            (snapshot_data['Timestamp'] <= window_end)
        ]
        
        # Calculate mean position for each fly across the time window
        averaged_snapshot = windowed_data[location_cols].mean().to_frame().T
        
        return averaged_snapshot
    
    def _count_flies_by_side(self, snapshot_data, cs_plus_side, determine_side=None):
        """
        Count flies on each side based on their positions and CS+ side.
        """
        if determine_side is None:
            if hasattr(self, 'determine_side'):
                determine_side = self.determine_side
            else:
                print("Warning: determine_side not set, using default value 60")
                determine_side = 60
        
        flies_on_cs_plus = 0
        flies_on_cs_minus = 0
        
        for col in snapshot_data.columns:
            position = snapshot_data[col].iloc[0]
            
            # Determine which side the fly is on
            if position > determine_side:
                fly_side = 'right'
            elif position < -determine_side:
                fly_side = 'left'
            else:
                # Fly is in the middle zone, skip
                continue
            
            # Count based on CS+ side
            if fly_side == cs_plus_side:
                flies_on_cs_plus += 1
            else:
                flies_on_cs_minus += 1
        
        return {
            'flies_on_cs_plus': flies_on_cs_plus,
            'flies_on_cs_minus': flies_on_cs_minus,
            'total_flies': flies_on_cs_plus + flies_on_cs_minus
        }
    
    def _calculate_pi(self, counts):
        """
        Calculate Preference Index: ((flies_on_cs_plus - flies_on_cs_minus) / total_flies) * 100
        """
        if counts['total_flies'] == 0:
            return 0.0
        
        pi = ((counts['flies_on_cs_plus'] - counts['flies_on_cs_minus']) / counts['total_flies']) * 100
        return pi

    def save_analysis_metadata(self, analysis_method, save_to_file=True, **kwargs):
        """
        Save analysis metadata including all parameters, version info, and analysis details.
        Creates a comprehensive JSON file for reproducibility and version control.
        Only saves if save_to_file is True
        
        Parameters:
        -----------
        analysis_method : str
            The analysis method used ('time' or 'snapshot')
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

    def analyse_learning_valence(self, determine_side=None, save_to_file=True):
        """
        Calculate learning valence scores - measures changes in valence with a learning session.
        
        Measures how fly preference for an odor changes after a learning/shock period by comparing 
        behavior during two 60-second exposures before and after shock where the odor switches sides. 
        Controls for immobility and side bias.
        
        Protocol phases:
        - Valence Before Shock Odor Right (odor on right)
        - Valence Before Shock Odor Left (odor on left) 
        - Valence After Shock Odor Right (odor on right)
        - Valence After Shock Odor Left (odor on left)
        
        Valence calculation:
        Valence = ((Time_right_when_odor_right / Total) - (Time_right_when_odor_left / Total)) * 100
        Results range from -100 (strong aversion) to +100 (strong attraction)
        Learned index = valence_after - valence_before (measures change due to learning)
        
        Parameters:
        -----------
        determine_side : float, optional
            Threshold for determining which side flies are on (0.0 to 100.0). Default: 10
        save_to_file : bool, optional
            Whether to save results and metadata to files. Default: True
            
        Returns:
        --------
        DataFrame with columns: fly_id, odor, valence_before, valence_after, 
        time_right_before_right, time_right_before_left, time_right_after_right, 
        time_right_after_left, learned_index, valid_fly
        """
        if determine_side is None:
            determine_side = 10  # Default value
            
        # 1. Identify the 4 valence phases
        phase_names = [
            'Valence Before Shock Odor Right',
            'Valence Before Shock Odor Left', 
            'Valence After Shock Odor Right',
            'Valence After Shock Odor Left'
        ]
        
        phase_data = {}
        for phase_name in phase_names:
            phase_df = self.raw_data[self.raw_data['experiment_step'] == phase_name]
            if phase_df.empty:
                print(f"Warning: Phase '{phase_name}' not found in data")
                return pd.DataFrame()
            phase_data[phase_name] = phase_df
        
        # 2. Detect odor from first phase (all phases should have same odor)
        odor = self._detect_valence_odor(phase_data['Valence Before Shock Odor Right'])
        print(f"Detected odor: {odor.upper()}")
        
        # 3. Apply filtering to each phase (flies must pass threshold in ALL phases)
        valid_flies = None
        for phase_name, phase_df in phase_data.items():
            # Get location columns for this phase
            location_cols = [col for col in phase_df.columns if 'chamber_' in col and '_loc' in col]
            phase_locations = phase_df[location_cols]
            
            # Apply midline filtering
            filtered_phase = self.filter_by_midline(phase_locations, self.midline_borders, self.filter_threshold)
            
            if valid_flies is None:
                valid_flies = set(filtered_phase.columns)
            else:
                valid_flies = valid_flies.intersection(set(filtered_phase.columns))
        
        if not valid_flies:
            print("Warning: No flies passed filtering in all phases")
            return pd.DataFrame()
            
        print(f"Number of valid flies: {len(valid_flies)}")
        
        # 4. Calculate time spent on right side for each phase
        phase_times = {}
        for phase_name, phase_df in phase_data.items():
            # Get location columns and filter to valid flies
            location_cols = [col for col in phase_df.columns if 'chamber_' in col and '_loc' in col]
            phase_locations = phase_df[location_cols]
            phase_locations = phase_locations[list(valid_flies)]
            
            # Calculate time spent on each side
            time_df = self.time_spent(phase_locations, determine_side)
            phase_times[phase_name] = time_df
        
        # 5. Calculate valence scores
        results_data = []
        for fly_id in valid_flies:
            fly_id_clean = fly_id.replace('chamber_', '').replace('_loc', '')
            
            # Get time on right side for each phase
            time_right_before_right = phase_times['Valence Before Shock Odor Right'].loc['right_side', fly_id]
            time_right_before_left = phase_times['Valence Before Shock Odor Left'].loc['right_side', fly_id]
            time_right_after_right = phase_times['Valence After Shock Odor Right'].loc['right_side', fly_id]
            time_right_after_left = phase_times['Valence After Shock Odor Left'].loc['right_side', fly_id]
            
            # Calculate total time for each phase
            total_before_right = (phase_times['Valence Before Shock Odor Right'].loc['right_side', fly_id] + 
                                phase_times['Valence Before Shock Odor Right'].loc['left_side', fly_id])
            total_before_left = (phase_times['Valence Before Shock Odor Left'].loc['right_side', fly_id] + 
                               phase_times['Valence Before Shock Odor Left'].loc['left_side', fly_id])
            total_after_right = (phase_times['Valence After Shock Odor Right'].loc['right_side', fly_id] + 
                               phase_times['Valence After Shock Odor Right'].loc['left_side', fly_id])
            total_after_left = (phase_times['Valence After Shock Odor Left'].loc['right_side', fly_id] + 
                              phase_times['Valence After Shock Odor Left'].loc['left_side', fly_id])
            
            # Calculate valence scores (scaled to -100 to 100)
            if total_before_right > 0 and total_before_left > 0:
                valence_before = ((time_right_before_right / total_before_right) - (time_right_before_left / total_before_left)) * 100
            else:
                valence_before = 0.0
                
            if total_after_right > 0 and total_after_left > 0:
                valence_after = ((time_right_after_right / total_after_right) - (time_right_after_left / total_after_left)) * 100
            else:
                valence_after = 0.0
            
            # Calculate learned index (valence after - valence before)
            learned_index = valence_after - valence_before
            
            results_data.append({
                'fly_id': fly_id_clean,
                'odor': odor,
                'valence_before': valence_before,
                'valence_after': valence_after,
                'learned_index': learned_index,
                'time_right_before_right': time_right_before_right,
                'time_right_before_left': time_right_before_left,
                'time_right_after_right': time_right_after_right,
                'time_right_after_left': time_right_after_left,
                'valid_fly': True
            })
        
        results_df = pd.DataFrame(results_data)
        
        # 6. Print summary
        print(f"\nLearning Valence Analysis Summary:")
        print(f"Odor: {odor.upper()}")
        print(f"Number of valid flies: {len(results_df)}")
        print(f"Mean valence before: {results_df['valence_before'].mean():.1f}")
        print(f"Mean valence after: {results_df['valence_after'].mean():.1f}")
        print(f"Mean learned index: {results_df['learned_index'].mean():.1f}")
        
        # 7. Save to CSV (only if save_to_file is True)
        self.save_analysis_results(results_df, save_to_file)
        
        # 8. Save analysis metadata (only if save_to_file is True)
        if save_to_file:
            metadata_params = {
                'determine_side': determine_side,
                'odor': odor,
                'num_valid_flies': len(results_df),
                'mean_valence_before': results_df['valence_before'].mean(),
                'mean_valence_after': results_df['valence_after'].mean(),
                'mean_learned_index': results_df['learned_index'].mean(),
                'analysis_method': 'learning_valence'
            }
            self.save_analysis_metadata('learning_valence', save_to_file, **metadata_params)
        
        return results_df
    
    def analyse_valence_habituation(self, determine_side=None, save_to_file=True):
        """
        Calculate valence scores for habituation test - measures valence across multiple repeated exposures.
        
        Detects all "Valence X Right" and "Valence X Left" steps in the trial where X is a number (1, 2, 3, etc.).
        Each "Valence X" step is subdivided into two sub-phases: Right and Left.
        Calculates valence for each step using the same formula as learning valence analysis.
        
        Valence calculation for each step:
        Valence_X = ((Time_right_when_odor_right / Total_right) - (Time_right_when_odor_left / Total_left)) * 100
        Results range from -100 (strong aversion) to +100 (strong attraction)
        
        Parameters:
        -----------
        determine_side : float, optional
            Threshold for determining which side flies are on (0.0 to 100.0). Default: 10
        save_to_file : bool, optional
            Whether to save results and metadata to files. Default: True
            
        Returns:
        --------
        DataFrame with columns: fly_id, odor, valid_fly, plus dynamic columns:
        - valence_1, valence_2, valence_3, ... (valence score for each step)
        - time_right_1_right, time_right_1_left, time_right_2_right, time_right_2_left, ... (time data)
        
        Raises:
        -------
        ValueError: If a "Valence X" step is missing its Right or Left phase
        """
        if determine_side is None:
            determine_side = 10  # Default value
        
        # 1. Find all "Valence X Right" and "Valence X Left" steps
        all_steps = self.raw_data['experiment_step'].unique()
        valence_steps = {}
        
        for step_name in all_steps:
            if pd.notna(step_name):
                match = re.match(r'Valence (\d+) (Right|Left)', step_name)
                if match:
                    step_num = int(match.group(1))
                    side = match.group(2).lower()
                    
                    if step_num not in valence_steps:
                        valence_steps[step_num] = {}
                    valence_steps[step_num][side] = step_name
        
        # Sort by step number
        step_numbers = sorted(valence_steps.keys())
        
        if not step_numbers:
            print("Error: No Valence X Right/Left steps found in data")
            return pd.DataFrame()
        
        print(f"Found {len(step_numbers)} valence step(s): {step_numbers}")
        
        # 2. Validate that each step has both Right and Left phases
        for step_num in step_numbers:
            if 'right' not in valence_steps[step_num]:
                raise ValueError(f"Valence {step_num} is missing Right phase - discarding trial")
            if 'left' not in valence_steps[step_num]:
                raise ValueError(f"Valence {step_num} is missing Left phase - discarding trial")
        
        # 3. Detect odor from first Right phase
        first_right_phase_name = valence_steps[step_numbers[0]]['right']
        first_right_phase_data = self.raw_data[self.raw_data['experiment_step'] == first_right_phase_name]
        odor = self._detect_valence_odor(first_right_phase_data)
        print(f"Detected odor: {odor.upper()}")
        
        # 4. Extract phase data for all steps
        phase_data = {}
        for step_num in step_numbers:
            phase_data[step_num] = {
                'right': self.raw_data[self.raw_data['experiment_step'] == valence_steps[step_num]['right']],
                'left': self.raw_data[self.raw_data['experiment_step'] == valence_steps[step_num]['left']]
            }
        
        # 5. Apply filtering to each phase (flies must pass threshold in ALL phases)
        valid_flies = None
        for step_num in step_numbers:
            for side in ['right', 'left']:
                phase_df = phase_data[step_num][side]
                location_cols = [col for col in phase_df.columns if 'chamber_' in col and '_loc' in col]
                phase_locations = phase_df[location_cols]
                
                # Apply midline filtering
                filtered_phase = self.filter_by_midline(phase_locations, self.midline_borders, self.filter_threshold)
                
                if valid_flies is None:
                    valid_flies = set(filtered_phase.columns)
                else:
                    valid_flies = valid_flies.intersection(set(filtered_phase.columns))
        
        if not valid_flies:
            print("Warning: No flies passed filtering in all phases")
            return pd.DataFrame()
        
        print(f"Number of valid flies: {len(valid_flies)}")
        
        # 6. Calculate time spent on right side for each phase
        phase_times = {}
        for step_num in step_numbers:
            phase_times[step_num] = {}
            for side in ['right', 'left']:
                phase_df = phase_data[step_num][side]
                location_cols = [col for col in phase_df.columns if 'chamber_' in col and '_loc' in col]
                phase_locations = phase_df[location_cols]
                phase_locations = phase_locations[list(valid_flies)]
                
                # Calculate time spent on each side
                time_df = self.time_spent(phase_locations, determine_side)
                phase_times[step_num][side] = time_df
        
        # 7. Calculate valence scores for each step
        results_data = []
        for fly_id in valid_flies:
            fly_id_clean = fly_id.replace('chamber_', '').replace('_loc', '')
            
            result_row = {
                'fly_id': fly_id_clean,
                'odor': odor,
                'valid_fly': True
            }
            
            # Calculate valence for each step
            for step_num in step_numbers:
                # Get time on right side for this step's Right and Left phases
                time_right_right = phase_times[step_num]['right'].loc['right_side', fly_id]
                time_right_left = phase_times[step_num]['left'].loc['right_side', fly_id]
                
                # Calculate total time for each phase
                total_right = (phase_times[step_num]['right'].loc['right_side', fly_id] + 
                             phase_times[step_num]['right'].loc['left_side', fly_id])
                total_left = (phase_times[step_num]['left'].loc['right_side', fly_id] + 
                            phase_times[step_num]['left'].loc['left_side', fly_id])
                
                # Store time data
                result_row[f'time_right_{step_num}_right'] = time_right_right
                result_row[f'time_right_{step_num}_left'] = time_right_left
                result_row[f'time_left_{step_num}_right'] = phase_times[step_num]['right'].loc['left_side', fly_id]
                result_row[f'time_left_{step_num}_left'] = phase_times[step_num]['left'].loc['left_side', fly_id]
                
                # Calculate valence score (scaled to -100 to 100)
                if total_right > 0 and total_left > 0:
                    valence = ((time_right_right / total_right) - (time_right_left / total_left)) * 100
                else:
                    valence = 0.0
                
                result_row[f'valence_{step_num}'] = valence
            
            results_data.append(result_row)
        
        results_df = pd.DataFrame(results_data)
        
        # 8. Print summary
        print(f"\nValence Habituation Analysis Summary:")
        print(f"Odor: {odor.upper()}")
        print(f"Number of valid flies: {len(results_df)}")
        print(f"Valence steps detected: {step_numbers}")
        for step_num in step_numbers:
            col_name = f'valence_{step_num}'
            if col_name in results_df.columns:
                print(f"Mean valence {step_num}: {results_df[col_name].mean():.1f}")
        
        # 9. Save to CSV (only if save_to_file is True)
        self.save_analysis_results(results_df, save_to_file)
        
        # 10. Save analysis metadata (only if save_to_file is True)
        if save_to_file:
            metadata_params = {
                'determine_side': determine_side,
                'odor': odor,
                'num_valid_flies': len(results_df),
                'valence_step_numbers': step_numbers,
                'analysis_method': 'valence_habituation'
            }
            # Add mean valence for each step
            for step_num in step_numbers:
                col_name = f'valence_{step_num}'
                if col_name in results_df.columns:
                    metadata_params[f'mean_valence_{step_num}'] = results_df[col_name].mean()
            
            self.save_analysis_metadata('valence_habituation', save_to_file, **metadata_params)
        
        return results_df
    
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
