import pandas as pd
import os


class SnapshotAnalysisMixin:
    """
    Mixin class for snapshot-based analysis methods
    """
    
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
