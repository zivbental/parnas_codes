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
    def time_spent(df, determine_side=10):
        """
        Calculate sample counts for each side of the chamber.
        Returns a dataframe showing for each fly the number of samples on each side.
        """
        def process_counts(counts):
            df_transposed = counts.reset_index().T
            df_transposed.columns = df_transposed.iloc[0]
            return df_transposed.drop(df_transposed.index[0])

        mask_greater = df > determine_side
        mask_less = df < -determine_side

        # Count samples (no sampling_rate multiplication - returns raw counts/proportions)
        count_greater = mask_greater.sum()
        count_less = mask_less.sum()

        count_greater_processed = process_counts(count_greater)
        count_less_processed = process_counts(count_less)

        df_combined = pd.concat([count_greater_processed, count_less_processed])
        df_combined.index = ['right_side', 'left_side']
        return df_combined

    @staticmethod
    def time_spent_v1_style(df, choicepoint_halfwidth=0.2):
        """
        Calculate time ratios using V1's exact TimeRatioCalculator logic.
        Always uses ternary classification with center exclusion.
        
        This replicates V1's TimeRatioCalculator.calculate() exactly:
        - Always uses ternary classification with choicepoint_halfwidth (default 0.2)
        - Excludes center frames from calculation: total = sum(indices where abs(bvec) == 1)
        - Returns left and right index sums for compatibility
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame with position data (columns are flies, rows are frames)
        choicepoint_halfwidth : float
            Half-width of central choicepoint zone in normalized coordinates (-1 to 1).
            Default 0.2 means positions within [-0.2, 0.2] are considered center.
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with rows ['right_side', 'left_side'] and columns for each fly.
            Values are the sum of 1-based frame indices where fly was on that side.
        """
        def process_counts(counts):
            df_transposed = counts.reset_index().T
            df_transposed.columns = df_transposed.iloc[0]
            return df_transposed.drop(df_transposed.index[0])

        import numpy as np
        
        # V1 logic: always use ternary classification with choicepoint_halfwidth
        cp = choicepoint_halfwidth
        
        sum_greater = {}  # right side
        sum_less = {}     # left side
        
        for col in df.columns:
            col_data = df[col].values  # Get column data as numpy array
            
            # Normalize data to -1 to 1 range for V1 compatibility
            # V1 uses normalized coordinates where -1 = left, 0 = center, +1 = right
            # V2 data could be in various scales:
            # - Already normalized: -1 to 1 (no change needed)
            # - 0-100 scale: 0 = left, 50 = center, 100 = right
            # - -50 to +50 scale: -50 = left, 0 = center, +50 = right
            # - Other scales possible
            
            # Normalize data to -1 to 1 range for V1 compatibility
            # V1 uses normalized coordinates where -1 = left, 0 = center, +1 = right
            # V2 data is in -100 to 100 scale, so normalize by dividing by 100
            col_data_abs_max = np.max(np.abs(col_data)) if len(col_data) > 0 else 1
            
            # Determine scale based on maximum absolute value
            if col_data_abs_max <= 1.5:
                # Already normalized (-1 to 1), use as-is
                pass
            elif col_data_abs_max > 50:
                # V2 scale: -100 to 100, normalize by dividing by 100
                col_data = col_data / 100
            elif col_data_abs_max > 1.5 and col_data_abs_max <= 55:
                # Likely in -50 to +50 scale (centered around 0)
                # Normalize: divide by 50
                col_data = col_data / 50
            else:
                # Already normalized or very small scale, use as-is
                pass
            
            # V1 ternary classification: -1 (left), 0 (center), 1 (right)
            # Left: position < -cp
            # Center: -cp <= position <= cp
            # Right: position > cp
            bvec = np.zeros_like(col_data, dtype=int)
            bvec[col_data > cp] = 1    # Right
            bvec[col_data < -cp] = -1  # Left
            # Center remains 0
            
            # V1 logic: sum 1-based indices where fly is on left
            left_indices = np.where(bvec == -1)[0]
            if len(left_indices) > 0:
                left_indices_1based = left_indices + 1  # Convert to 1-based
                leftindex = np.sum(left_indices_1based)
            else:
                leftindex = 0
            
            # V1 logic: sum 1-based indices where fly is in decision zone (left or right, excluding center)
            # total = sum(find(abs(bvec)==1))
            total_indices = np.where(np.abs(bvec) == 1)[0]
            if len(total_indices) > 0:
                total_indices_1based = total_indices + 1  # Convert to 1-based
                total = np.sum(total_indices_1based)
            else:
                total = 0
            
            # Separate right indices for compatibility with existing code
            right_indices = np.where(bvec == 1)[0]
            if len(right_indices) > 0:
                right_indices_1based = right_indices + 1  # Convert to 1-based
                sum_greater[col] = np.sum(right_indices_1based)
            else:
                sum_greater[col] = 0
            
            sum_less[col] = leftindex
        
        sum_greater_series = pd.Series(sum_greater)
        sum_less_series = pd.Series(sum_less)
        
        sum_greater_processed = process_counts(sum_greater_series)
        sum_less_processed = process_counts(sum_less_series)

        df_combined = pd.concat([sum_greater_processed, sum_less_processed])
        df_combined.index = ['right_side', 'left_side']
        return df_combined

    @staticmethod
    def time_spent_matlab(df, determine_side=10):
        """
        Calculate time ratios using MATLAB index-summing method.
        This replicates the logic from timeratio_alistair.m where frame indices
        are summed instead of counted, creating a time-weighted metric.
        
        Uses ternarylocationfunc logic when determine_side corresponds to V1's cp=0.2
        (which is 20 in V2 scale). This excludes midline frames from the calculation.
        
        Returns a dataframe showing for each fly the index-sum ratio.
        """
        def process_counts(counts):
            df_transposed = counts.reset_index().T
            df_transposed.columns = df_transposed.iloc[0]
            return df_transposed.drop(df_transposed.index[0])

        import numpy as np
        
        # Check if we should use ternary location (exclude midline)
        # MATLAB's timeratio_alistair uses ternarylocationfunc with cp=0.2 (V1 scale)
        # In V2 scale, this is 20
        use_ternary = (determine_side == 20)
        
        # MATLAB approach: sum the indices where condition is true
        # timeratio_alistair: leftindex=sum(find(bvec==-1)); total=sum(find(abs(bvec)==1));
        sum_greater = {}
        sum_less = {}
        
        for col in df.columns:
            col_data = df[col].values  # Get column data as numpy array
            
            if use_ternary:
                # Use ternary location: -1 (left), 0 (midline), 1 (right)
                # Only sum indices for left (-1) and right (1), excluding midline (0)
                bvec = np.where(col_data > determine_side, 1, np.where(col_data < -determine_side, -1, 0))
                
                # MATLAB: leftindex=sum(find(bvec==-1))
                left_indices = np.where(bvec == -1)[0] + 1  # +1 for 1-indexing
                leftindex = np.sum(left_indices) if len(left_indices) > 0 else 0
                
                # MATLAB: total=sum(find(abs(bvec)==1))  (excludes midline)
                total_indices = np.where(np.abs(bvec) == 1)[0] + 1
                total = np.sum(total_indices) if len(total_indices) > 0 else 0
                
                # Separate left and right sums for compatibility
                right_indices = np.where(bvec == 1)[0] + 1
                sum_greater[col] = np.sum(right_indices) if len(right_indices) > 0 else 0
                sum_less[col] = leftindex
            else:
                # Use binary location: >= 0 for right, < 0 for left
                if determine_side == 0:
                    mask_greater = col_data >= 0
                    mask_less = col_data < 0
                else:
                    mask_greater = col_data > determine_side
                    mask_less = col_data < -determine_side
                
                bvec = np.where(mask_greater, 1, np.where(mask_less, -1, 0))
                
                # Binary location logic
                indices_greater = np.where(bvec == 1)[0] + 1  # +1 for 1-indexing
                indices_less = np.where(bvec == -1)[0] + 1
                
                sum_greater[col] = np.sum(indices_greater) if len(indices_greater) > 0 else 0
                sum_less[col] = np.sum(indices_less) if len(indices_less) > 0 else 0
        
        sum_greater_series = pd.Series(sum_greater)
        sum_less_series = pd.Series(sum_less)
        
        sum_greater_processed = process_counts(sum_greater_series)
        sum_less_processed = process_counts(sum_less_series)

        df_combined = pd.concat([sum_greater_processed, sum_less_processed])
        df_combined.index = ['right_side', 'left_side']
        return df_combined

    def detect_trial_epochs(self):
        """
        Detect trial epochs in the data (V1-style).
        
        This method identifies trial periods (epochs where flies can make choices)
        by finding phases that correspond to trial epochs. In V1, these are epochs
        with tag==1 (trial periods with asymmetric odors, no shock).
        
        For V2, we identify trial-like phases (typically "Test" phases or "Initial Valence")
        and return the first and last trial epochs.
        
        Returns:
        --------
        tuple
            (first_trial_data, last_trial_data, first_trial_indices, last_trial_indices) where:
            - first_trial_data: DataFrame with position data for first trial epoch
            - last_trial_data: DataFrame with position data for last trial epoch  
            - first_trial_indices: (start_idx, end_idx) frame indices for first trial (0-based)
            - last_trial_indices: (start_idx, end_idx) frame indices for last trial (0-based)
        """
        if self.raw_data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        # Find "Initial Valence" and "Test" phases
        # V1 uses first trial vs last trial, so we use Initial Valence as first trial and Test as last trial
        valence_phase = self.raw_data[self.raw_data['experiment_step'] == 'Initial Valence']
        test_phases = self.raw_data[self.raw_data['experiment_step'] == 'Test']
        
        # Determine first and last trial epochs
        # First trial: Initial Valence (if exists), otherwise first Test phase
        # Last trial: Last Test phase (if exists), otherwise Initial Valence
        
        if len(valence_phase) > 0 and len(test_phases) > 0:
            # Both phases exist: use Initial Valence as first, Test as last
            first_trial_data = valence_phase.filter(regex=r'^(Timestamp|chamber_\d+_loc)$')
            first_trial_data = first_trial_data.reset_index(drop=True)
            first_start = valence_phase.index[0]
            first_end = valence_phase.index[-1]
            
            # Get last Test phase (all Test phases are typically one continuous block, but get the end)
            last_test_start_idx = test_phases.index[0]
            last_test_end_idx = test_phases.index[-1]
            # Find where Test phase actually ends
            for i in range(last_test_end_idx + 1, len(self.raw_data)):
                if self.raw_data.iloc[i]['experiment_step'] != 'Test':
                    break
                last_test_end_idx = i
            
            last_trial_data = self.raw_data.iloc[last_test_start_idx:last_test_end_idx+1].filter(regex=r'^(Timestamp|chamber_\d+_loc)$')
            last_trial_data = last_trial_data.reset_index(drop=True)
            last_start = last_test_start_idx
            last_end = last_test_end_idx
            
        elif len(test_phases) > 0:
            # Only Test phases exist: check if there are multiple distinct Test blocks
            # Group consecutive Test phases by checking gaps in indices
            test_indices = test_phases.index.tolist()
            test_groups = []
            if len(test_indices) > 0:
                current_group_start = test_indices[0]
                current_group_end = test_indices[0]
                
                for i in range(1, len(test_indices)):
                    # Check if there's a gap (more than 1 index difference indicates non-consecutive)
                    if test_indices[i] > test_indices[i-1] + 1:
                        # Gap detected, save current group and start new one
                        test_groups.append((current_group_start, current_group_end))
                        current_group_start = test_indices[i]
                    current_group_end = test_indices[i]
                
                # Add the last group
                test_groups.append((current_group_start, current_group_end))
            
            if len(test_groups) >= 2:
                # Multiple Test blocks: use first as first trial, last as last trial
                first_start, first_end = test_groups[0]
                last_start, last_end = test_groups[-1]
            else:
                # Single Test block: split it in half for first vs last trial
                last_start = test_phases.index[0]
                last_end = test_phases.index[-1]
                # Find where Test phase actually ends (might extend beyond indexed rows)
                for i in range(last_end + 1, len(self.raw_data)):
                    if self.raw_data.iloc[i]['experiment_step'] != 'Test':
                        break
                    last_end = i
                
                # Split Test phase in half for first vs last
                test_midpoint = (last_start + last_end) // 2
                first_start = last_start
                first_end = test_midpoint
                last_start = test_midpoint + 1
            
            first_trial_data = self.raw_data.iloc[first_start:first_end+1].filter(regex=r'^(Timestamp|chamber_\d+_loc)$')
            last_trial_data = self.raw_data.iloc[last_start:last_end+1].filter(regex=r'^(Timestamp|chamber_\d+_loc)$')
            first_trial_data = first_trial_data.reset_index(drop=True)
            last_trial_data = last_trial_data.reset_index(drop=True)
            
        elif len(valence_phase) > 0:
            # Only Initial Valence exists: split it in half
            first_start = valence_phase.index[0]
            first_end = valence_phase.index[-1]
            valence_midpoint = (first_start + first_end) // 2
            first_end = valence_midpoint
            last_start = valence_midpoint + 1
            last_end = valence_phase.index[-1]
            
            first_trial_data = self.raw_data.iloc[first_start:first_end+1].filter(regex=r'^(Timestamp|chamber_\d+_loc)$')
            last_trial_data = self.raw_data.iloc[last_start:last_end+1].filter(regex=r'^(Timestamp|chamber_\d+_loc)$')
            first_trial_data = first_trial_data.reset_index(drop=True)
            last_trial_data = last_trial_data.reset_index(drop=True)
        else:
            raise ValueError("No trial epochs found: no 'Test' or 'Initial Valence' phases")
        
        return first_trial_data, last_trial_data, (first_start, first_end), (last_start, last_end)
    
    def identify_trained_odor_v1(self):
        """
        Identify trained odor using V1's logic (from shock patterns).
        
        This replicates V1's TrainedOdorIdentifier._find_trained_odor() method:
        - Finds first shock location
        - Determines trained odor from odor state at first shock
        - Returns trained odor as 3-character string [AIR, MCH, OCT]
        
        Returns:
        --------
        str
            3-character string encoding trained odor: [AIR, MCH, OCT]
            Example: '110' = AIR and MCH on, OCT off
            '010' = only MCH on
        """
        if self.raw_data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        import numpy as np
        
        # Find shock columns
        shock_cols = [col for col in self.raw_data.columns if 'shock' in col.lower() and 'chamber' in col.lower()]
        
        if len(shock_cols) == 0:
            raise ValueError("No shock columns found in data. Cannot identify trained odor.")
        
        # Get shock data - use first fly's shock data (all flies get same training)
        shock_col = shock_cols[0]  # Use first shock column
        shock_vector = self.raw_data[shock_col].values
        
        # Find first frame where shock was delivered
        shock_locs = np.where(shock_vector == 1)[0]
        
        if len(shock_locs) == 0:
            raise ValueError("No shock locations found. Cannot identify trained odor.")
        
        first_shock_idx = shock_locs[0]
        first_shock_row = self.raw_data.iloc[first_shock_idx]
        
        # Get odor status at first shock
        odor_columns = ['mch_left_status', 'mch_right_status',
                       'oct_left_status', 'oct_right_status',
                       'moil_left_status', 'moil_right_status',
                       'mch_left_status', 'oct_left_status']  # Add air if available
        
        # Check which columns exist
        available_cols = [col for col in odor_columns if col in self.raw_data.columns]
        
        if len(available_cols) < 4:
            raise ValueError(f"Insufficient odor status columns found. Need at least mch and oct on both sides.")
        
        # Build odor state vector (6-element: left AIR, MCH, OCT, right AIR, MCH, OCT)
        # For V2, we need to infer AIR from absence of other odors, or use a default
        # V1 uses: [LEFTAIR, LEFTMCH, LEFTOCT, RIGHTAIR, RIGHTMCH, RIGHTOCT]
        
        # Try to get odor states
        mch_left = first_shock_row.get('mch_left_status', 0)
        mch_right = first_shock_row.get('mch_right_status', 0)
        oct_left = first_shock_row.get('oct_left_status', 0)
        oct_right = first_shock_row.get('oct_right_status', 0)
        moil_left = first_shock_row.get('moil_left_status', 0)
        moil_right = first_shock_row.get('moil_right_status', 0)
        
        # Infer AIR: if no other odor is on, AIR is on (1), otherwise AIR is off (0)
        # But in V2, we might not have explicit AIR status
        # For now, assume AIR is on if no other odor is on that side
        air_left = 1 if (mch_left == 0 and oct_left == 0 and moil_left == 0) else 0
        air_right = 1 if (mch_right == 0 and oct_right == 0 and moil_right == 0) else 0
        
        # Build 6-character odor state string: [LEFTAIR, LEFTMCH, LEFTOCT, RIGHTAIR, RIGHTMCH, RIGHTOCT]
        # Note: MOIL is not in V1's standard format, so we'll treat it as MCH or OCT for compatibility
        # For simplicity, if MOIL is present, we'll encode it based on context
        # But for trained odor identification, we only need left or right side
        
        # Get position at first shock to determine which side was trained
        loc_cols = [col for col in self.raw_data.columns if col.startswith('chamber_') and col.endswith('_loc')]
        if len(loc_cols) == 0:
            raise ValueError("No location columns found.")
        
        position_snap = first_shock_row[loc_cols[0]]
        
        # Determine trained odor based on fly position at first shock
        # If fly was on left (position <= 0), use left-side odors
        # If fly was on right (position > 0), use right-side odors
        if position_snap <= 0:
            # Fly was on left side: left odors are the trained ones
            # Convert to 3-character string: [AIR, MCH, OCT]
            trained_odor = ''.join([str(int(air_left)), str(int(mch_left)), str(int(oct_left))])
        else:
            # Fly was on right side: right odors are the trained ones
            trained_odor = ''.join([str(int(air_right)), str(int(mch_right)), str(int(oct_right))])
        
        return trained_odor
    
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
        elif analysis_method == 'time-matlab':
            metadata["analysis_info"]["description"] = "MATLAB-style time analysis - uses index-summing from timeratio_alistair.m"
            metadata["data_info"]["filtering_details"] = {
                "midline_borders": getattr(self, 'midline_borders', 'Not set'),
                "filter_threshold": getattr(self, 'filter_threshold', 'Not set'),
                "filter_phase": kwargs.get('filter_phase', 'Not specified'),
                "calculation_method": "Index-summing (MATLAB timeratio_alistair.m replication)"
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
