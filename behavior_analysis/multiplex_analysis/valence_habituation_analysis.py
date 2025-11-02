import pandas as pd
import re


class ValenceHabituationAnalysisMixin:
    """
    Mixin class for valence habituation analysis methods
    """
    
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
