import pandas as pd


class LearningValenceAnalysisMixin:
    """
    Mixin class for learning valence analysis methods
    """
    
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
