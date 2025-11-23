import pandas as pd


class TimeAnalysisMixin:
    """
    Mixin class for time-based analysis methods
    """
    
    def analyse_time(self, determine_side=None, min_valence_fraction=0.0, save_to_file=True):
        """
        Calculate learned index with automatic CS+/CS- detection and side-switching handling
        
        Parameters:
        -----------
        determine_side : float, optional
            Threshold for determining which side flies are on (0.0 to 100.0). Default: 10
        min_valence_fraction : float, optional
            Minimum fraction of total time a fly must spend in initial valence period to be included in analysis. Default: 0.0
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

        # 7. Calculate CS+ preference for each phase (adaptive to actual side)
        if valence_cs_side == 'left':
            initial_val = (filtered_valence_df.iloc[1]) / (filtered_valence_df.iloc[0] + filtered_valence_df.iloc[1])  # left/total
        else:
            initial_val = (filtered_valence_df.iloc[0]) / (filtered_valence_df.iloc[0] + filtered_valence_df.iloc[1])  # right/total
        
        # Filter out flies with CS+ preference < min_valence_fraction in initial valence
        if min_valence_fraction > 0:
            initial_val_filter = initial_val >= min_valence_fraction
            filtered_valence_df = filtered_valence_df.loc[:, initial_val_filter]
            filtered_test_df = filtered_test_df.loc[:, initial_val_filter]
            initial_val = initial_val[initial_val_filter]
            
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
                'min_valence_fraction': min_valence_fraction,
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

    def analyse_time_matlab(self, determine_side=None, min_valence_fraction=0.0, save_to_file=True):
        """
        Calculate learned index using V1-compatible trial epoch analysis.
        This replicates V1's exact logic:
        - Uses trial epochs (first vs last trial) instead of phases
        - Uses trained odor identification from shock patterns
        - Uses V1's TimeRatioCalculator logic (always ternary, excludes center)
        - Calculates preferences for trained odor epochs
        
        Parameters:
        -----------
        determine_side : float, optional
            Not used in V1-style (always uses choicepoint_halfwidth=0.2). Kept for compatibility.
        min_valence_fraction : float, optional
            Minimum fraction of total time a fly must spend on trained odor side in first trial to be included.
            Default: 0.0 (no filtering). Values 0.0 to 1.0.
        save_to_file : bool, optional
            Whether to save results and metadata to files. Default: True
        """
        import numpy as np
        
        # 1. Identify trained odor using V1 logic (from shock patterns)
        try:
            trained_odor = self.identify_trained_odor_v1()
        except Exception as e:
            print(f"Warning: Could not identify trained odor from shocks: {e}")
            print("Falling back to CS+ detection method")
            cs_plus_odor, _ = self.identify_cs_plus_odor()
            # Convert CS+ odor name to trained odor format
            # This is a fallback - not ideal but maintains compatibility
            if cs_plus_odor == 'mch':
                trained_odor = '010'  # Only MCH
            elif cs_plus_odor == 'oct':
                trained_odor = '001'  # Only OCT
            else:
                trained_odor = '010'  # Default to MCH
        
        print(f"Trained odor identified: {trained_odor}")
        
        # 2. Detect trial epochs (first and last trial)
        try:
            first_trial_data, last_trial_data, first_trial_indices, last_trial_indices = self.detect_trial_epochs()
        except Exception as e:
            print(f"Error detecting trial epochs: {e}")
            raise
        
        # 3. Get odor states for each trial epoch to determine trained odor side
        first_trial_odor_data = self.raw_data.iloc[first_trial_indices[0]:first_trial_indices[1]+1]
        last_trial_odor_data = self.raw_data.iloc[last_trial_indices[0]:last_trial_indices[1]+1]
        
        # Determine which side trained odor is on in each trial
        def get_trained_odor_side(odor_data, trained_odor):
            """Determine which side trained odor is on in this epoch"""
            if odor_data.empty:
                return 'left'  # Default
            
            # Get odor status at first frame of epoch
            odor_columns = ['mch_left_status', 'mch_right_status',
                          'oct_left_status', 'oct_right_status']
            
            first_row = odor_data.iloc[0]
            
            # Build left and right odor states
            mch_left = first_row.get('mch_left_status', 0)
            mch_right = first_row.get('mch_right_status', 0)
            oct_left = first_row.get('oct_left_status', 0)
            oct_right = first_row.get('oct_right_status', 0)
            
            # Infer AIR
            air_left = 1 if (mch_left == 0 and oct_left == 0) else 0
            air_right = 1 if (mch_right == 0 and oct_right == 0) else 0
            
            left_state = ''.join([str(int(air_left)), str(int(mch_left)), str(int(oct_left))])
            right_state = ''.join([str(int(air_right)), str(int(mch_right)), str(int(oct_right))])
            
            if left_state == trained_odor:
                return 'left'
            elif right_state == trained_odor:
                return 'right'
            else:
                # Try to match by checking if trained odor components match
                # Trained odor format: [AIR, MCH, OCT]
                # Check MCH and OCT components
                if trained_odor[1] == '1':  # MCH component
                    if mch_left == 1:
                        return 'left'
                    elif mch_right == 1:
                        return 'right'
                if trained_odor[2] == '1':  # OCT component
                    if oct_left == 1:
                        return 'left'
                    elif oct_right == 1:
                        return 'right'
                
                # Default fallback
                return 'left'
        
        first_trial_side = get_trained_odor_side(first_trial_odor_data, trained_odor)
        last_trial_side = get_trained_odor_side(last_trial_odor_data, trained_odor)
        
        print(f"Trained odor side in first trial: {first_trial_side.upper()}")
        print(f"Trained odor side in last trial: {last_trial_side.upper()}")
        
        # 4. Apply threshold and midline_borders filtering (use only flies that passed filter_by_num_choices)
        # Get valid fly columns from processed_data (which has already been filtered)
        if hasattr(self, 'processed_data') and self.processed_data is not None:
            # processed_data is a tuple of (valence_df, test_df) when filter='both'
            # Get the valid fly column names from processed_data
            if isinstance(self.processed_data, tuple):
                # Both phases filtered: use common columns
                valid_columns = self.processed_data[0].columns
            else:
                # Single phase filtered
                valid_columns = self.processed_data.columns
            
            # Extract chamber location column names (remove any non-loc columns like Timestamp)
            valid_loc_columns = [col for col in valid_columns if col.startswith('chamber_') and col.endswith('_loc')]
            
            # Check if valid_loc_columns exist in trial data
            available_loc_columns = [col for col in first_trial_data.columns if col.startswith('chamber_') and col.endswith('_loc')]
            if len(valid_loc_columns) == 0:
                print(f"Warning: No valid location columns found in processed_data. Using all available columns.")
                valid_loc_columns = available_loc_columns
            else:
                # Only use columns that exist in both processed_data and trial data
                valid_loc_columns = [col for col in valid_loc_columns if col in available_loc_columns]
                if len(valid_loc_columns) == 0:
                    print(f"Warning: No matching columns between processed_data and trial data. Using all available columns.")
                    valid_loc_columns = available_loc_columns
            
            # Filter trial data to only include valid flies
            first_trial_locations = first_trial_data[valid_loc_columns]
            last_trial_locations = last_trial_data[valid_loc_columns]
        else:
            # No filtering applied, use all flies
            first_trial_locations = first_trial_data.filter(regex=r'^chamber_\d+_loc$')
            last_trial_locations = last_trial_data.filter(regex=r'^chamber_\d+_loc$')
        
        if first_trial_locations.empty or last_trial_locations.empty:
            raise ValueError(f"No location data found in trial epochs. First trial: {len(first_trial_locations.columns)} columns, Last trial: {len(last_trial_locations.columns)} columns")
        
        print(f"Analyzing {len(first_trial_locations.columns)} flies in trial epochs")
        
        # Calculate time ratios using V1-style method (always ternary, excludes center)
        # V1 uses choicepoint_halfwidth = 0.2 (normalized coordinates)
        # Note: determine_side parameter is not used in V1-style (always uses choicepoint_halfwidth=0.2)
        first_trial_df = self.time_spent_v1_style(first_trial_locations, choicepoint_halfwidth=0.2)
        last_trial_df = self.time_spent_v1_style(last_trial_locations, choicepoint_halfwidth=0.2)
        
        # 5. Apply filtering: flies must have valid data in both trials
        first_denominator = first_trial_df.iloc[1] + first_trial_df.iloc[0]  # left + right
        last_denominator = last_trial_df.iloc[0] + last_trial_df.iloc[1]  # right + left
        combined_mask = (first_denominator != 0) & (last_denominator != 0)
        
        filtered_first_df = first_trial_df.loc[:, combined_mask]
        filtered_last_df = last_trial_df.loc[:, combined_mask]
        
        # 6. Calculate preferences for trained odor epochs (V1 logic)
        # V1: if trained odor on left, preference = left_percentage
        #     if trained odor on right, preference = 100 - left_percentage = right_percentage
        
        # Calculate left percentages for each trial
        # Handle division by zero: if total is 0, the fly was always in center zone, so preference is undefined
        first_denominator = filtered_first_df.iloc[0] + filtered_first_df.iloc[1]
        last_denominator = filtered_last_df.iloc[0] + filtered_last_df.iloc[1]
        
        # Check for flies with no decision frames (always in center) - these should be filtered out
        valid_flies = (first_denominator > 0) & (last_denominator > 0)
        if not valid_flies.all():
            num_filtered = valid_flies.sum()
            print(f"Warning: {len(valid_flies) - num_filtered} flies have no decision frames (always in center zone) - filtering out")
            filtered_first_df = filtered_first_df.loc[:, valid_flies]
            filtered_last_df = filtered_last_df.loc[:, valid_flies]
            first_denominator = filtered_first_df.iloc[0] + filtered_first_df.iloc[1]
            last_denominator = filtered_last_df.iloc[0] + filtered_last_df.iloc[1]
        
        # Calculate left percentage for each trial
        # DataFrame structure: iloc[0] = right_side row, iloc[1] = left_side row
        first_left_pc = (filtered_first_df.iloc[1] / first_denominator) * 100
        last_left_pc = (filtered_last_df.iloc[1] / last_denominator) * 100
        
        # Debug: Check for extreme values and understand why
        if len(first_left_pc) > 0:
            extreme_first = (first_left_pc == 0) | (first_left_pc == 100)
            if extreme_first.sum() > 0:
                print(f"Debug: {extreme_first.sum()} flies have extreme first trial left_pc values (0 or 100)")
                print(f"  First trial left_pc stats: min={first_left_pc.min():.2f}, max={first_left_pc.max():.2f}, mean={first_left_pc.mean():.2f}")
                print(f"  First trial denominator stats: min={first_denominator.min():.0f}, max={first_denominator.max():.0f}, mean={first_denominator.mean():.0f}")
                # Check if extreme values are due to very few decision frames
                extreme_flies = first_denominator[extreme_first]
                if len(extreme_flies) > 0:
                    print(f"  Extreme value flies have denominator range: min={extreme_flies.min():.0f}, max={extreme_flies.max():.0f}, mean={extreme_flies.mean():.0f}")
                    small_denom = extreme_flies[extreme_flies < 100]
                    if len(small_denom) > 0:
                        print(f"  WARNING: {len(small_denom)} flies with extreme values have very few decision frames (<100), suggesting most frames are in center zone")
        
        # Apply V1 trained odor preference logic
        # V1 calculates preference for trained odor side:
        # - If trained odor on left: preference = left_percentage
        # - If trained odor on right: preference = right_percentage = 100 - left_percentage
        if first_trial_side == 'left':
            time_before = first_left_pc  # Preference for trained odor side (left)
        else:
            time_before = 100 - first_left_pc  # Preference for trained odor side (right) = 100 - left_percentage
        
        # 7. Apply min_valence_fraction filtering (similar to analyse_time)
        # Filter out flies with trained odor preference < min_valence_fraction in first trial
        if min_valence_fraction > 0:
            time_before_filter = time_before >= (min_valence_fraction * 100)  # Convert fraction to percentage
            if not time_before_filter.all():
                num_filtered = time_before_filter.sum()
                print(f"Filtering out {len(time_before_filter) - num_filtered} flies with first trial preference < {min_valence_fraction * 100:.1f}%")
                filtered_first_df = filtered_first_df.loc[:, time_before_filter]
                filtered_last_df = filtered_last_df.loc[:, time_before_filter]
                time_before = time_before[time_before_filter]
                first_left_pc = first_left_pc[time_before_filter]
                last_left_pc = last_left_pc[time_before_filter]
        
        if last_trial_side == 'left':
            time_after = last_left_pc
        else:
            time_after = 100 - last_left_pc
        
        # 8. Calculate learned index (change in preference)
        learned_index = time_after - time_before
        
        # 9. Create results dataframe
        results_data = []
        for i, col in enumerate(filtered_first_df.columns):
            fly_id = col.replace('chamber_', '').replace('_loc', '')
            results_data.append({
                'fly_id': fly_id,
                'cs_plus_odor': trained_odor,  # Store as cs_plus_odor for compatibility
                'trained_odor': trained_odor,  # Store as trained_odor for V1 compatibility
                'initial_valence_cs_side': first_trial_side,  # Keep for compatibility
                'test_cs_side': last_trial_side,  # Keep for compatibility
                'sides_switched': (first_trial_side != last_trial_side),
                'initial_valence_cs_preference': time_before.iloc[i],
                'test_cs_preference': time_after.iloc[i],
                'learned_index': learned_index.iloc[i],
                'valid_fly': True
            })
        
        results_df = pd.DataFrame(results_data)
        
        # 10. Print summary
        print(f"\n=== V1-Compatible Trial Epoch Analysis ===")
        print(f"Trained Odor: {trained_odor}")
        print(f"Number of valid flies: {len(results_df)}")
        print(f"Mean learned index: {learned_index.mean():.2f}")
        print(f"Std learned index: {learned_index.std():.2f}")
        print(f"Note: Uses V1-compatible logic (trial epochs, trained odor, ternary classification)")
        
        # 11. Save to CSV (only if save_to_file is True)
        self.save_analysis_results(results_df, save_to_file)
        
        # 12. Save analysis metadata
        if save_to_file:
            metadata_params = {
                'choicepoint_halfwidth': 0.2,
                'determine_side': determine_side,  # Note: not used in V1-style, kept for compatibility
                'min_valence_fraction': min_valence_fraction,
                'threshold': getattr(self, 'filter_threshold', None),  # From filter_by_num_choices
                'midline_borders': getattr(self, 'midline_borders', None),  # From filter_by_num_choices
                'trained_odor': trained_odor,
                'first_trial_side': first_trial_side,
                'last_trial_side': last_trial_side,
                'sides_switched': (first_trial_side != last_trial_side),
                'num_valid_flies': len(results_df),
                'mean_learned_index': learned_index.mean(),
                'std_learned_index': learned_index.std(),
                'method_note': 'V1-compatible trial epoch analysis (replicates V1 TimeRatioCalculator and TrainedOdorIdentifier logic)'
            }
            self.save_analysis_metadata('time-matlab', save_to_file, **metadata_params)
        
        return results_df
