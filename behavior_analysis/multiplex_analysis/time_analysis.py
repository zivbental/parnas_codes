import pandas as pd


class TimeAnalysisMixin:
    """
    Mixin class for time-based analysis methods
    """
    
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
