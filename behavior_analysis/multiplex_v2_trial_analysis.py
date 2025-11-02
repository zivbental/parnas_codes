"""
Simple analysis script for individual multiplex trials.
Uses the shared core module for all analysis functionality.
Supports both time-based and snapshot-based analysis methods.
"""

from multiplex_core import MultiplexTrial

# Example usage
if __name__ == "__main__":
    # Load a single Trial to the object
    file_path = "fly_loc.csv"
    trial_1 = MultiplexTrial()
    trial_1.load_data(file_path)

    trial_1.filter_by_num_choices(midline_borders=60, threshold=4, filter='both')

    # Choose analysis method
    analysis_method = 'time'  # Change to 'snapshot' for population-level analysis, 'learning_valence' for learning valence, 'valence_habituation' for habituation
    
    # Parameters for side determination and filtering
    determine_side = 10  # Threshold for determining which side flies are on
    min_valence_seconds = 30  # Minimum seconds in initial valence period for valid fly
    time_window = [0, 5]  # For snapshot analysis: time range [start_seconds, end_seconds] from phase start
    
    if analysis_method == 'time':
        print("Running TIME-BASED analysis with automatic CS+/CS- detection and side-switching handling...")
        print("This analyzes individual fly time spent on each side.")
        results = trial_1.analyse_time(determine_side, min_valence_seconds)
        print("\nIndividual fly results:")
        
    elif analysis_method == 'snapshot':
        print("Running SNAPSHOT analysis with automatic CS+/CS- detection and side-switching handling...")
        print("This analyzes population preference at the end of each phase.")
        results = trial_1.analyse_snapshot(determine_side, time_window)
        print("\nTrial-level results:")
        
    elif analysis_method == 'learning_valence':
        print("Running LEARNING VALENCE analysis - measures changes in valence with a learning session...")
        print("This measures how fly preference changes after learning by comparing behavior before and after shock.")
        results = trial_1.analyse_learning_valence(determine_side)
        print("\nIndividual fly learning valence results:")
    
    elif analysis_method == 'valence_habituation':
        print("Running VALENCE HABITUATION analysis - measures valence across multiple repeated exposures...")
        print("This measures valence across multiple Valence X steps (where X is 1, 2, 3, etc.).")
        results = trial_1.analyse_valence_habituation(determine_side)
        print("\nIndividual fly valence habituation results:")
    
    print(results)
    
    # Example of switching analysis methods on the same trial
    print("\n" + "="*60)
    print("COMPARISON: Running all analysis methods on the same trial")
    print("="*60)
    
    # Time-based analysis
    print("\n1. TIME-BASED ANALYSIS:")
    time_results = trial_1.analyse_time(determine_side, min_valence_seconds)
    print(f"   Mean learned index: {time_results['learned_index'].mean():.2f}")
    
    # Snapshot analysis
    print("\n2. SNAPSHOT ANALYSIS:")
    snapshot_results = trial_1.analyse_snapshot(determine_side, time_window)
    print(f"   Learned index: {snapshot_results['learned_index'].iloc[0]:.2f}")
    
    # Learning valence analysis
    print("\n3. LEARNING VALENCE ANALYSIS:")
    learning_valence_results = trial_1.analyse_learning_valence(determine_side)
    print(f"   Mean valence before: {learning_valence_results['valence_before'].mean():.1f}")
    print(f"   Mean valence after: {learning_valence_results['valence_after'].mean():.1f}")
    print(f"   Mean learned index: {learning_valence_results['learned_index'].mean():.1f}")
    
    # Valence habituation analysis
    print("\n4. VALENCE HABITUATION ANALYSIS:")
    try:
        habituation_results = trial_1.analyse_valence_habituation(determine_side)
        valence_cols = [col for col in habituation_results.columns if col.startswith('valence_') and col.replace('valence_', '').isdigit()]
        for col in sorted(valence_cols, key=lambda x: int(x.replace('valence_', ''))):
            step_num = col.replace('valence_', '')
            print(f"   Mean valence step {step_num}: {habituation_results[col].mean():.1f}")
    except ValueError as e:
        print(f"   Error: {str(e)} - trial may not have habituation protocol")
    
    print("\nNote: Time-based gives individual fly results, Snapshot gives trial-level results, Learning Valence measures changes with learning, Habituation measures multiple exposures")