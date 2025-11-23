import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import datetime
from scipy import stats
from scipy.stats import shapiro, levene, ttest_ind, mannwhitneyu
from multiplex_analysis import MultiplexTrial


def load_experiment_config(config_path="experiment_config.json"):
    """
    Load experiment configuration from JSON file
    """
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    else:
        print(f"Warning: Config file {config_path} not found. Using default configuration.")
        return {
            "groups": {"control": [], "experimental": []},
            "comparisons": [],
            "analysis_settings": {
                "midline_borders": 0.6,
                "threshold": 4,
                "multiple_testing_correction": "bonferroni",
                "significance_level": 0.05
            }
        }


def collect_trial_data(folder_path, threshold=4, midline_borders=60, filter_phase='both', analysis_method='time', determine_side=10, min_valence_fraction=0.0, time_window=[0, 5]):
    """
    Collect data from all trials in experiment folder with comprehensive tracking
    
    Parameters:
    -----------
    folder_path : str
        Path to the experiment folder
    threshold : int
        Minimum choices required for valid fly
    midline_borders : float
        Midline border threshold (0.0 to 100.0)
    filter_phase : str
        Which phases to filter ('both', 'initial', 'test', or 'none')
    analysis_method : str
        Analysis method to use ('time' for time-based individual fly analysis, 
        'snapshot' for population-level snapshot analysis, 
        'time-matlab' for MATLAB-style index-summing time analysis,
        'learning_valence' for learning valence analysis,
        'valence_habituation' for valence habituation analysis)
    determine_side : float
        Threshold for determining which side flies are on (0.0 to 100.0)
    min_valence_fraction : float
        Minimum fraction of time a fly must spend in initial valence period to be included in analysis (0.0 to 1.0)
    time_window : list of two floats
        For snapshot analysis: time range [start_seconds, end_seconds] from phase start to average positions
    
    Returns:
        DataFrame with columns: genotype, trial_date, trial_number, fly_id,
        learned_index, cs_plus_odor, initial_valence_cs_side, test_cs_side,
        sides_switched, num_valid_flies, num_total_flies
        For learning_valence analysis: also includes valence_before, valence_after, odor
        For valence_habituation analysis: includes dynamic columns valence_1, valence_2, etc., plus odor
    """
    all_results = []
    
    print(f"Scanning experiment folder: {folder_path}")
    
    for date_folder in os.listdir(folder_path):
        date_path = os.path.join(folder_path, date_folder)
        if not os.path.isdir(date_path):
            continue
            
        print(f"  Processing date folder: {date_folder}")
        
        for trial_folder in os.listdir(date_path):
            trial_path = os.path.join(date_path, trial_folder)
            if not os.path.isdir(trial_path):
                continue
                
            # Check for required files
            metadata_path = os.path.join(trial_path, 'experiment_metadata.json')
            data_path = os.path.join(trial_path, 'fly_loc.csv')
            
            if not (os.path.exists(metadata_path) and os.path.exists(data_path)):
                print(f"    Skipping {trial_folder}: Missing required files")
                continue
            
            try:
                # Load metadata for genotype
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                fly_genotype = metadata.get('flyGenotype', 'unknown')
                
                print(f"    Processing trial: {trial_folder} (genotype: {fly_genotype})")
                
                # Create trial object and run analysis
                trial = MultiplexTrial()
                trial.load_data(data_path)
                trial.filter_by_num_choices(midline_borders, threshold, filter_phase)
                
                # Get results with CS+ information using specified analysis method
                if analysis_method == 'time':
                    results_df = trial.analyse_time(determine_side, min_valence_fraction, save_to_file=False)
                elif analysis_method == 'time-matlab':
                    results_df = trial.analyse_time_matlab(determine_side, min_valence_fraction, save_to_file=False)
                elif analysis_method == 'snapshot':
                    results_df = trial.analyse_snapshot(determine_side, time_window, save_to_file=False)
                elif analysis_method == 'learning_valence':
                    results_df = trial.analyse_learning_valence(determine_side, save_to_file=False)
                elif analysis_method == 'valence_habituation':
                    results_df = trial.analyse_valence_habituation(determine_side, save_to_file=False)
                else:
                    raise ValueError(f"Unknown analysis method: {analysis_method}. Use 'time', 'snapshot', 'learning_valence', or 'valence_habituation'.")
                
                # Add metadata to results (handles both fly-level and trial-level results)
                for _, row in results_df.iterrows():
                    if analysis_method == 'time':
                        # Time-based analysis returns fly-level results
                        all_results.append({
                            'genotype': fly_genotype,
                            'trial_date': date_folder,
                            'trial_number': trial_folder,
                            'fly_id': row['fly_id'],
                            'learned_index': row['learned_index'],
                            'cs_plus_odor': row['cs_plus_odor'],
                            'initial_valence_cs_side': row['initial_valence_cs_side'],
                            'test_cs_side': row['test_cs_side'],
                            'sides_switched': row['sides_switched'],
                            'initial_valence_cs_preference': row['initial_valence_cs_preference'],
                            'test_cs_preference': row['test_cs_preference'],
                            'valid_fly': row.get('valid_fly', True),
                            'analysis_method': 'time'
                        })
                    elif analysis_method == 'time-matlab':
                        # MATLAB-style time-based analysis returns fly-level results
                        all_results.append({
                            'genotype': fly_genotype,
                            'trial_date': date_folder,
                            'trial_number': trial_folder,
                            'fly_id': row['fly_id'],
                            'learned_index': row['learned_index'],
                            'cs_plus_odor': row['cs_plus_odor'],
                            'initial_valence_cs_side': row['initial_valence_cs_side'],
                            'test_cs_side': row['test_cs_side'],
                            'sides_switched': row['sides_switched'],
                            'initial_valence_cs_preference': row['initial_valence_cs_preference'],
                            'test_cs_preference': row['test_cs_preference'],
                            'valid_fly': row.get('valid_fly', True),
                            'analysis_method': 'time-matlab'
                        })
                    elif analysis_method == 'snapshot':
                        # Snapshot analysis returns trial-level results
                        all_results.append({
                            'genotype': fly_genotype,
                            'trial_date': date_folder,
                            'trial_number': trial_folder,
                            'fly_id': f"trial_{trial_folder}",  # Use trial as fly_id for snapshot
                            'learned_index': row['learned_index'],
                            'cs_plus_odor': row['cs_plus_odor'],
                            'initial_valence_cs_side': row['initial_valence_cs_side'],
                            'test_cs_side': row['test_cs_side'],
                            'sides_switched': row['sides_switched'],
                            'initial_valence_cs_preference': row['initial_valence_cs_preference'],
                            'test_cs_preference': row['test_cs_preference'],
                            'num_flies': row.get('num_flies', 'N/A'),
                            'analysis_method': 'snapshot'
                        })
                    elif analysis_method == 'learning_valence':
                        # Learning valence analysis returns fly-level results
                        all_results.append({
                            'genotype': fly_genotype,
                            'trial_date': date_folder,
                            'trial_number': trial_folder,
                            'fly_id': row['fly_id'],
                            'learned_index': row['learned_index'],  # Learning valence learned index
                            'cs_plus_odor': None,  # Not applicable for learning valence analysis
                            'initial_valence_cs_side': None,  # Not applicable for learning valence analysis
                            'test_cs_side': None,  # Not applicable for learning valence analysis
                            'sides_switched': None,  # Not applicable for learning valence analysis
                            'initial_valence_cs_preference': None,  # Not applicable for learning valence analysis
                            'test_cs_preference': None,  # Not applicable for learning valence analysis
                            'valid_fly': row.get('valid_fly', True),
                            'analysis_method': 'learning_valence',
                            'valence_before': row['valence_before'],
                            'valence_after': row['valence_after'],
                            'odor': row['odor'],
                            'time_right_before_right': row['time_right_before_right'],
                            'time_right_before_left': row['time_right_before_left'],
                            'time_right_after_right': row['time_right_after_right'],
                            'time_right_after_left': row['time_right_after_left']
                        })
                    elif analysis_method == 'valence_habituation':
                        # Valence habituation analysis returns fly-level results with dynamic columns
                        result_dict = {
                            'genotype': fly_genotype,
                            'trial_date': date_folder,
                            'trial_number': trial_folder,
                            'fly_id': row['fly_id'],
                            'learned_index': None,  # Not applicable for habituation analysis
                            'cs_plus_odor': None,  # Not applicable for habituation analysis
                            'initial_valence_cs_side': None,  # Not applicable for habituation analysis
                            'test_cs_side': None,  # Not applicable for habituation analysis
                            'sides_switched': None,  # Not applicable for habituation analysis
                            'initial_valence_cs_preference': None,  # Not applicable for habituation analysis
                            'test_cs_preference': None,  # Not applicable for habituation analysis
                            'valid_fly': row.get('valid_fly', True),
                            'analysis_method': 'valence_habituation',
                            'odor': row['odor']
                        }
                        # Add all dynamic valence and time columns
                        for col in row.index:
                            if col.startswith('valence_') or col.startswith('time_right_') or col.startswith('time_left_'):
                                result_dict[col] = row[col]
                        all_results.append(result_dict)
                
                print(f"      Added {len(results_df)} flies from {trial_folder}")
                
            except ValueError as e:
                # For habituation analysis, ValueError means missing phase - discard trial
                if analysis_method == 'valence_habituation':
                    print(f"    Error processing {trial_folder}: {str(e)} - discarding trial")
                else:
                    print(f"    Error processing {trial_folder}: {str(e)}")
                continue
            except Exception as e:
                print(f"    Error processing {trial_folder}: {str(e)}")
                continue
    
    print(f"Total flies collected: {len(all_results)}")
    return pd.DataFrame(all_results)


def test_assumptions(data1, data2, group1_name, group2_name):
    """
    Test statistical assumptions for parametric tests
    
    Returns:
        dict with test results and recommendations
    """
    results = {
        'group1_name': group1_name,
        'group2_name': group2_name,
        'n1': len(data1),
        'n2': len(data2),
        'normality_passed': False,
        'homogeneity_passed': False,
        'recommended_test': 'mannwhitneyu'
    }
    
    # Test normality (Shapiro-Wilk)
    if len(data1) >= 3 and len(data2) >= 3:
        try:
            _, p1 = shapiro(data1)
            _, p2 = shapiro(data2)
            results['normality_p1'] = p1
            results['normality_p2'] = p2
            results['normality_passed'] = p1 > 0.05 and p2 > 0.05
        except:
            results['normality_passed'] = False
    
    # Test homogeneity of variance (Levene's test)
    if len(data1) >= 2 and len(data2) >= 2:
        try:
            _, p_homogeneity = levene(data1, data2)
            results['homogeneity_p'] = p_homogeneity
            results['homogeneity_passed'] = p_homogeneity > 0.05
        except:
            results['homogeneity_passed'] = False
    
    # Determine recommended test
    if results['normality_passed'] and results['homogeneity_passed']:
        results['recommended_test'] = 'ttest_ind'
    elif results['normality_passed'] and not results['homogeneity_passed']:
        results['recommended_test'] = 'ttest_ind_unequal'
    else:
        results['recommended_test'] = 'mannwhitneyu'
    
    return results


def perform_statistical_test(data1, data2, test_type):
    """
    Perform the appropriate statistical test
    
    Returns:
        dict with test results
    """
    if test_type == 'ttest_ind':
        statistic, p_value = ttest_ind(data1, data2)
        test_name = "Independent t-test"
    elif test_type == 'ttest_ind_unequal':
        statistic, p_value = ttest_ind(data1, data2, equal_var=False)
        test_name = "Welch's t-test"
    elif test_type == 'mannwhitneyu':
        statistic, p_value = mannwhitneyu(data1, data2, alternative='two-sided')
        test_name = "Mann-Whitney U test"
    else:
        raise ValueError(f"Unknown test type: {test_type}")
    
    # Calculate effect size
    if test_type in ['ttest_ind', 'ttest_ind_unequal']:
        # Cohen's d
        pooled_std = np.sqrt(((len(data1) - 1) * np.var(data1, ddof=1) + 
                             (len(data2) - 1) * np.var(data2, ddof=1)) / 
                            (len(data1) + len(data2) - 2))
        effect_size = (np.mean(data1) - np.mean(data2)) / pooled_std
        effect_size_name = "Cohen's d"
    else:
        # Rank-biserial correlation
        n1, n2 = len(data1), len(data2)
        effect_size = 1 - (2 * statistic) / (n1 * n2)
        effect_size_name = "Rank-biserial correlation"
    
    # Calculate confidence interval for mean difference
    mean_diff = np.mean(data1) - np.mean(data2)
    se_diff = np.sqrt(np.var(data1, ddof=1)/len(data1) + np.var(data2, ddof=1)/len(data2))
    ci_lower = mean_diff - 1.96 * se_diff
    ci_upper = mean_diff + 1.96 * se_diff
    
    return {
        'test_name': test_name,
        'statistic': statistic,
        'p_value': p_value,
        'effect_size': effect_size,
        'effect_size_name': effect_size_name,
        'mean_difference': mean_diff,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper
    }


def perform_statistical_analysis(data, config):
    """
    Perform comprehensive statistical analysis with assumption testing
    """
    print("\n" + "="*60)
    print("STATISTICAL ANALYSIS")
    print("="*60)
    
    results = []
    comparisons = config.get('comparisons', [])
    
    if not comparisons:
        print("No comparisons defined in configuration.")
        return pd.DataFrame()
    
    # Determine analysis type based on data
    analysis_method = data['analysis_method'].iloc[0] if 'analysis_method' in data.columns else 'time'
    
    if analysis_method == 'learning_valence':
        # For learning valence analysis, compare valence_before, valence_after, and learned_index
        metrics = ['valence_before', 'valence_after', 'learned_index']
    elif analysis_method == 'valence_habituation':
        # For habituation analysis, find all valence_X columns dynamically
        valence_cols = [col for col in data.columns if col.startswith('valence_') and col.replace('valence_', '').isdigit()]
        metrics = sorted(valence_cols, key=lambda x: int(x.replace('valence_', '')))
        if not metrics:
            print("Warning: No valence columns found for habituation analysis")
            return pd.DataFrame()
    else:
        # For learning analysis, use learned_index
        metrics = ['learned_index']
    
    for metric in metrics:
        if metric not in data.columns:
            print(f"Warning: Metric {metric} not found in data, skipping...")
            continue
            
        print(f"\nAnalyzing metric: {metric}")
        print("-" * 50)
        
        for i, comparison in enumerate(comparisons):
            exp_genotype = comparison['experimental']
            control_genotypes = comparison['controls']
            
            print(f"\nComparison {i+1}: {exp_genotype} vs {', '.join(control_genotypes)}")
            print("-" * 50)
            
            # Get experimental data
            exp_data = data[data['genotype'] == exp_genotype][metric].dropna()
            
            if len(exp_data) == 0:
                print(f"  Warning: No data found for experimental group {exp_genotype}")
                continue
            
            for control_genotype in control_genotypes:
                # Get control data
                ctrl_data = data[data['genotype'] == control_genotype][metric].dropna()
                
                if len(ctrl_data) == 0:
                    print(f"  Warning: No data found for control group {control_genotype}")
                    continue
                
                print(f"\n  {exp_genotype} (n={len(exp_data)}) vs {control_genotype} (n={len(ctrl_data)})")
                
                # Test assumptions
                assumptions = test_assumptions(exp_data, ctrl_data, exp_genotype, control_genotype)
                
                print(f"    Normality test: p1={assumptions.get('normality_p1', 'N/A'):.3f}, "
                      f"p2={assumptions.get('normality_p2', 'N/A'):.3f} "
                      f"({'PASSED' if assumptions['normality_passed'] else 'FAILED'})")
                print(f"    Homogeneity test: p={assumptions.get('homogeneity_p', 'N/A'):.3f} "
                      f"({'PASSED' if assumptions['homogeneity_passed'] else 'FAILED'})")
                print(f"    Recommended test: {assumptions['recommended_test']}")
                
                # Perform statistical test
                test_results = perform_statistical_test(exp_data, ctrl_data, assumptions['recommended_test'])
                
                print(f"    {test_results['test_name']}: statistic={test_results['statistic']:.3f}, "
                      f"p={test_results['p_value']:.3f}")
                print(f"    Effect size ({test_results['effect_size_name']}): {test_results['effect_size']:.3f}")
                print(f"    Mean difference: {test_results['mean_difference']:.3f} "
                      f"(95% CI: {test_results['ci_lower']:.3f}, {test_results['ci_upper']:.3f})")
                
                # Store results
                results.append({
                    'metric': metric,
                    'comparison': f"{exp_genotype} vs {control_genotype}",
                    'experimental_group': exp_genotype,
                    'control_group': control_genotype,
                    'n_experimental': len(exp_data),
                    'n_control': len(ctrl_data),
                    'test_name': test_results['test_name'],
                    'statistic': test_results['statistic'],
                    'p_value': test_results['p_value'],
                    'effect_size': test_results['effect_size'],
                    'effect_size_name': test_results['effect_size_name'],
                    'mean_difference': test_results['mean_difference'],
                    'ci_lower': test_results['ci_lower'],
                    'ci_upper': test_results['ci_upper'],
                    'normality_passed': assumptions['normality_passed'],
                    'homogeneity_passed': assumptions['homogeneity_passed']
                })
    
    # Apply multiple testing correction
    if results:
        results_df = pd.DataFrame(results)
        correction_method = config.get('analysis_settings', {}).get('multiple_testing_correction', 'bonferroni')
        
        if correction_method == 'bonferroni':
            results_df['corrected_p_value'] = results_df['p_value'] * len(results_df)
            results_df['corrected_p_value'] = results_df['corrected_p_value'].clip(upper=1.0)
        else:
            # Benjamini-Hochberg FDR
            from scipy.stats import false_discovery_control
            results_df['corrected_p_value'] = false_discovery_control(results_df['p_value'])
        
        print(f"\nMultiple testing correction ({correction_method}):")
        for _, row in results_df.iterrows():
            significance = "***" if row['corrected_p_value'] < 0.001 else \
                          "**" if row['corrected_p_value'] < 0.01 else \
                          "*" if row['corrected_p_value'] < 0.05 else "ns"
            print(f"  {row['comparison']}: p={row['p_value']:.3f} -> "
                  f"p_corrected={row['corrected_p_value']:.3f} {significance}")
        
        return results_df
    
    return pd.DataFrame()


def create_plots(data, stats_results, output_folder):
    """
    Create comprehensive plots with dynamic limits and CS+ information
    """
    print(f"\nCreating plots in {output_folder}")
    
    # Create output directory
    os.makedirs(output_folder, exist_ok=True)
    
    # Determine analysis type
    analysis_method = data['analysis_method'].iloc[0] if 'analysis_method' in data.columns else 'time'
    
    if analysis_method == 'learning_valence':
        create_learning_valence_plots(data, stats_results, output_folder)
    elif analysis_method == 'valence_habituation':
        create_valence_habituation_plots(data, stats_results, output_folder)
    else:
        create_learning_plots(data, stats_results, output_folder)


def create_learning_plots(data, stats_results, output_folder):
    """
    Create plots for learning analysis (time/snapshot methods)
    """
    # Calculate dynamic y-axis limits
    all_values = data['learned_index'].dropna()
    
    if len(all_values) == 0:
        print("Warning: No learned_index data found for plotting")
        return
    
    y_min = all_values.min() - 0.1 * (all_values.max() - all_values.min())
    y_max = all_values.max() + 0.1 * (all_values.max() - all_values.min())
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Multiplex Learning Analysis Results', fontsize=16, fontweight='bold')
    
    # 1. Bar plot with SEM error bars
    ax1 = axes[0, 0]
    sample_sizes = data.groupby('genotype').size()
    sns.barplot(data=data, x='genotype', y='learned_index', hue='genotype', 
                palette="deep", errorbar="se", estimator="mean", capsize=0.1, ax=ax1)
    
    # Add N values to x-axis labels
    ax1.set_xticklabels([f'{label.get_text()}\n(n={sample_sizes[label.get_text()]})' for label in ax1.get_xticklabels()])
    ax1.set_title('Mean Learned Index by Genotype')
    ax1.set_ylabel('Learned Index (%)')
    ax1.set_ylim(y_min, y_max)
    ax1.tick_params(axis='x', rotation=45)
    ax1.legend().remove()
    
    # 2. Box plot
    ax2 = axes[0, 1]
    sns.boxplot(data=data, x='genotype', y='learned_index', hue='genotype', palette="deep", ax=ax2)
    ax2.set_xticklabels([f'{label.get_text()}\n(n={sample_sizes[label.get_text()]})' for label in ax2.get_xticklabels()])
    ax2.set_title('Distribution of Learned Index by Genotype')
    ax2.set_ylabel('Learned Index (%)')
    ax2.set_ylim(y_min, y_max)
    ax2.tick_params(axis='x', rotation=45)
    ax2.legend().remove()
    
    # 3. Swarm plot
    ax3 = axes[1, 0]
    sns.swarmplot(data=data, x='genotype', y='learned_index', hue='genotype', palette="deep", ax=ax3)
    ax3.set_xticklabels([f'{label.get_text()}\n(n={sample_sizes[label.get_text()]})' for label in ax3.get_xticklabels()])
    ax3.set_title('Individual Fly Learned Index by Genotype')
    ax3.set_ylabel('Learned Index (%)')
    ax3.set_ylim(y_min, y_max)
    ax3.tick_params(axis='x', rotation=45)
    ax3.legend().remove()
    
    # 4. CS+ configuration plot
    ax4 = axes[1, 1]
    cs_config = data.groupby(['genotype', 'cs_plus_odor', 'sides_switched']).size().reset_index(name='count')
    cs_pivot = cs_config.pivot_table(index='genotype', columns=['cs_plus_odor', 'sides_switched'], 
                                    values='count', fill_value=0)
    cs_pivot.plot(kind='bar', stacked=True, ax=ax4, colormap='Set3')
    ax4.set_title('CS+ Configuration by Genotype')
    ax4.set_ylabel('Number of Flies')
    ax4.tick_params(axis='x', rotation=45)
    ax4.legend(title='CS+ Odor & Side Switch', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    # Save combined plot
    combined_path = os.path.join(output_folder, 'combined_analysis_plots.png')
    plt.savefig(combined_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create individual plots
    plot_types = [
        ('bar_plot', axes[0, 0]),
        ('box_plot', axes[0, 1]),
        ('swarm_plot', axes[1, 0]),
        ('cs_config_plot', axes[1, 1])
    ]
    
    for plot_name, ax in plot_types:
        fig, ax = plt.subplots(figsize=(8, 6))
        
        if plot_name == 'bar_plot':
            sns.barplot(data=data, x='genotype', y='learned_index', hue='genotype', 
                       palette="deep", errorbar="se", estimator="mean", capsize=0.1, ax=ax)
            ax.set_title('Mean Learned Index by Genotype')
        elif plot_name == 'box_plot':
            sns.boxplot(data=data, x='genotype', y='learned_index', hue='genotype', palette="deep", ax=ax)
            ax.set_title('Distribution of Learned Index by Genotype')
        elif plot_name == 'swarm_plot':
            sns.swarmplot(data=data, x='genotype', y='learned_index', hue='genotype', palette="deep", ax=ax)
            ax.set_title('Individual Fly Learned Index by Genotype')
        elif plot_name == 'cs_config_plot':
            cs_config = data.groupby(['genotype', 'cs_plus_odor', 'sides_switched']).size().reset_index(name='count')
            cs_pivot = cs_config.pivot_table(index='genotype', columns=['cs_plus_odor', 'sides_switched'], 
                                           values='count', fill_value=0)
            cs_pivot.plot(kind='bar', stacked=True, ax=ax, colormap='Set3')
            ax.set_title('CS+ Configuration by Genotype')
            ax.legend(title='CS+ Odor & Side Switch', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Add N values to x-axis labels (except for cs_config_plot)
        if plot_name != 'cs_config_plot':
            ax.set_xticklabels([f'{label.get_text()}\n(n={sample_sizes[label.get_text()]})' for label in ax.get_xticklabels()])
        
        ax.set_ylabel('Learned Index (%)')
        ax.set_ylim(y_min, y_max)
        ax.tick_params(axis='x', rotation=45)
        
        if plot_name != 'cs_config_plot':
            ax.legend().remove()
        
        plt.tight_layout()
        
        # Save individual plot
        individual_path = os.path.join(output_folder, f'{plot_name}.png')
        plt.savefig(individual_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Plots saved to {output_folder}")


def create_learning_valence_plots(data, stats_results, output_folder):
    """
    Create plots for learning valence analysis
    """
    # Calculate dynamic y-axis limits for all metrics
    valence_before_values = data['valence_before'].dropna()
    valence_after_values = data['valence_after'].dropna()
    learned_index_values = data['learned_index'].dropna()
    
    if len(valence_before_values) == 0 and len(valence_after_values) == 0 and len(learned_index_values) == 0:
        print("Warning: No valence data found for plotting")
        return
    
    all_valence_values = pd.concat([valence_before_values, valence_after_values, learned_index_values])
    y_min = all_valence_values.min() - 0.1 * (all_valence_values.max() - all_valence_values.min())
    y_max = all_valence_values.max() + 0.1 * (all_valence_values.max() - all_valence_values.min())
    
    # Create figure with subplots for learning valence analysis (1x3 layout)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Learning Valence Analysis Results', fontsize=16, fontweight='bold')
    
    # 1. Valence Before - Combined bar and scatter plot
    ax1 = axes[0]
    sample_sizes = data.groupby('genotype').size()
    
    # Create bar plot with SEM error bars
    sns.barplot(data=data, x='genotype', y='valence_before', hue='genotype', 
                palette="deep", errorbar="se", estimator="mean", capsize=0.1, ax=ax1, alpha=0.7)
    
    # Add scatter plot on top
    sns.stripplot(data=data, x='genotype', y='valence_before', hue='genotype', 
                  palette="deep", size=4, alpha=0.8, ax=ax1, dodge=False)
    
    # Add N values to x-axis labels
    ax1.set_xticklabels([f'{label.get_text()}\n(n={sample_sizes[label.get_text()]})' for label in ax1.get_xticklabels()])
    ax1.set_title('Valence Before by Genotype')
    ax1.set_ylabel('Valence Before (-100 to +100)')
    ax1.set_ylim(y_min, y_max)
    ax1.tick_params(axis='x', rotation=45)
    ax1.legend().remove()
    
    # 2. Valence After - Combined bar and scatter plot
    ax2 = axes[1]
    
    # Create bar plot with SEM error bars
    sns.barplot(data=data, x='genotype', y='valence_after', hue='genotype', 
                palette="deep", errorbar="se", estimator="mean", capsize=0.1, ax=ax2, alpha=0.7)
    
    # Add scatter plot on top
    sns.stripplot(data=data, x='genotype', y='valence_after', hue='genotype', 
                  palette="deep", size=4, alpha=0.8, ax=ax2, dodge=False)
    
    # Add N values to x-axis labels
    ax2.set_xticklabels([f'{label.get_text()}\n(n={sample_sizes[label.get_text()]})' for label in ax2.get_xticklabels()])
    ax2.set_title('Valence After by Genotype')
    ax2.set_ylabel('Valence After (-100 to +100)')
    ax2.set_ylim(y_min, y_max)
    ax2.tick_params(axis='x', rotation=45)
    ax2.legend().remove()
    
    # 3. Learned Index - Combined bar and scatter plot
    ax3 = axes[2]
    
    # Create bar plot with SEM error bars
    sns.barplot(data=data, x='genotype', y='learned_index', hue='genotype', 
                palette="deep", errorbar="se", estimator="mean", capsize=0.1, ax=ax3, alpha=0.7)
    
    # Add scatter plot on top
    sns.stripplot(data=data, x='genotype', y='learned_index', hue='genotype', 
                  palette="deep", size=4, alpha=0.8, ax=ax3, dodge=False)
    
    # Add N values to x-axis labels
    ax3.set_xticklabels([f'{label.get_text()}\n(n={sample_sizes[label.get_text()]})' for label in ax3.get_xticklabels()])
    ax3.set_title('Learned Index by Genotype')
    ax3.set_ylabel('Learned Index (Valence After - Before)')
    ax3.set_ylim(y_min, y_max)
    ax3.tick_params(axis='x', rotation=45)
    ax3.legend().remove()
    
    plt.tight_layout()
    
    # Save combined plot
    combined_path = os.path.join(output_folder, 'combined_learning_valence_plots.png')
    plt.savefig(combined_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create individual plots for learning valence (combined bar + scatter)
    valence_metrics = ['valence_before', 'valence_after', 'learned_index']
    
    for metric in valence_metrics:
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Create bar plot with SEM error bars
        sns.barplot(data=data, x='genotype', y=metric, hue='genotype', 
                   palette="deep", errorbar="se", estimator="mean", capsize=0.1, ax=ax, alpha=0.7)
        
        # Add scatter plot on top
        sns.stripplot(data=data, x='genotype', y=metric, hue='genotype', 
                      palette="deep", size=4, alpha=0.8, ax=ax, dodge=False)
        
        # Add N values to x-axis labels
        ax.set_xticklabels([f'{label.get_text()}\n(n={sample_sizes[label.get_text()]})' for label in ax.get_xticklabels()])
        ax.set_title(f'{metric.replace("_", " ").title()} by Genotype')
        
        if metric == 'learned_index':
            ax.set_ylabel(f"{metric.replace('_', ' ').title()} (Valence After - Before)")
        else:
            ax.set_ylabel(f"{metric.replace('_', ' ').title()} (-100 to +100)")
        ax.set_ylim(y_min, y_max)
        ax.tick_params(axis='x', rotation=45)
        ax.legend().remove()
        
        plt.tight_layout()
        
        # Save individual plot
        individual_path = os.path.join(output_folder, f'{metric}_combined_plot.png')
        plt.savefig(individual_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Learning valence plots saved to {output_folder}")


def create_valence_habituation_plots(data, stats_results, output_folder):
    """
    Create plots for valence habituation analysis.
    X-axis: Valence step numbers (1, 2, 3, etc.)
    Y-axis: Valence value
    Bars: Grouped by genotype (side by side) for each valence step
    Single plot showing all valence steps across all protocol types.
    """
    # Find all valence columns
    valence_cols = [col for col in data.columns if col.startswith('valence_') and col.replace('valence_', '').isdigit()]
    valence_cols = sorted(valence_cols, key=lambda x: int(x.replace('valence_', '')))
    
    if not valence_cols:
        print("Warning: No valence columns found for habituation plotting")
        return
    
    # Prepare data for plotting - reshape to long format
    plot_data_list = []
    for col in valence_cols:
        step_num = col.replace('valence_', '')
        step_data = data[['genotype', col]].copy()
        step_data = step_data.dropna(subset=[col])
        if len(step_data) > 0:  # Only add if there's data
            step_data['valence_step'] = f'Step {step_num}'
            step_data = step_data.rename(columns={col: 'valence'})
            plot_data_list.append(step_data)
    
    if not plot_data_list:
        print("Warning: No valence data found for plotting")
        return
    
    plot_data = pd.concat(plot_data_list, ignore_index=True)
    
    # Calculate dynamic y-axis limits
    all_valence_values = plot_data['valence'].dropna()
    
    if len(all_valence_values) == 0:
        print("Warning: No valence values found")
        return
    
    y_min = all_valence_values.min() - 0.1 * (all_valence_values.max() - all_valence_values.min())
    y_max = all_valence_values.max() + 0.1 * (all_valence_values.max() - all_valence_values.min())
    
    # Get unique step labels and sort them numerically
    unique_steps = sorted(plot_data['valence_step'].unique(), 
                          key=lambda x: int(x.replace('Step ', '')))
    n_steps = len(unique_steps)
    
    # Create figure - single plot with X bars (valence steps) and genotypes side by side
    fig, ax = plt.subplots(figsize=(max(6, 2*n_steps), 6))
    
    # Create grouped bar plot
    # X-axis: valence step (Step 1, Step 2, etc.)
    # Hue: genotype (bars side by side for each step)
    sns.barplot(data=plot_data, x='valence_step', y='valence', hue='genotype', 
               palette="deep", errorbar="se", estimator="mean", capsize=0.1, ax=ax, alpha=0.7,
               order=unique_steps)
    
    # Add scatter plot on top
    sns.stripplot(data=plot_data, x='valence_step', y='valence', hue='genotype', 
                 palette="deep", size=4, alpha=0.8, ax=ax, dodge=True, order=unique_steps)
    
    ax.set_title('Valence Habituation Analysis', fontsize=16, fontweight='bold')
    ax.set_xlabel('Valence Step', fontsize=12)
    ax.set_ylabel('Valence (-100 to +100)', fontsize=12)
    ax.set_ylim(y_min, y_max)
    ax.tick_params(axis='x', rotation=0)
    
    # Add N values to legend - calculate sample sizes per genotype
    sample_sizes = data.groupby('genotype').size()
    handles, labels = ax.get_legend_handles_labels()
    # Get unique genotypes from data (not just from legend) to ensure accurate counts
    unique_genotypes = sorted(data['genotype'].unique())
    new_labels = []
    for label in labels:
        if label in sample_sizes.index:
            new_labels.append(f'{label} (n={sample_sizes[label]})')
        else:
            new_labels.append(label)
    ax.legend(handles, new_labels, title='Genotype', loc='best')
    
    plt.tight_layout()
    
    # Save combined plot
    combined_path = os.path.join(output_folder, 'combined_valence_habituation_plots.png')
    plt.savefig(combined_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Valence habituation plots saved to {output_folder}")


def save_results_to_csv(folder_path, data, stats_results):
    """
    Save analysis results to CSV files in timestamped output folder
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder = os.path.join(folder_path, f'output_{timestamp}')
    os.makedirs(output_folder, exist_ok=True)
    
    # Save raw data
    data_path = os.path.join(output_folder, 'experiment_data_cleaned.csv')
    data.to_csv(data_path, index=False)
    print(f"Raw data saved to: {data_path}")
    
    # Save statistical results
    if not stats_results.empty:
        stats_path = os.path.join(output_folder, 'statistical_results.csv')
        stats_results.to_csv(stats_path, index=False)
        print(f"Statistical results saved to: {stats_path}")
    
    return output_folder


def save_project_metadata(output_folder, config, analysis_method, determine_side, min_valence_fraction, time_window, data, stats_results):
    """
    Save project analysis metadata including all parameters, version info, and analysis details.
    Creates a comprehensive JSON file for reproducibility and version control.
    """
    # Create metadata dictionary
    metadata = {
        "analysis_info": {
            "method": "project_analysis",
            "analysis_type": analysis_method,
            "timestamp": datetime.datetime.now().isoformat(),
            "version": "2.0.0",
            "description": f"Project-level {analysis_method} analysis across multiple trials"
        },
        "parameters": {
            "determine_side": determine_side,
            "min_valence_fraction": min_valence_fraction,
            "time_window": time_window,
            "analysis_method": analysis_method,
            "experiment_config": config
        },
        "data_info": {
            "total_trials_analyzed": len(data['trial_number'].unique()) if 'trial_number' in data.columns else len(data),
            "total_flies_analyzed": len(data),
            "genotypes_analyzed": list(data['genotype'].unique()) if 'genotype' in data.columns else [],
            "date_folders_analyzed": list(data['trial_date'].unique()) if 'trial_date' in data.columns else [],
            "statistical_tests_performed": len(stats_results) if not stats_results.empty else 0
        },
        "results_summary": {
            "mean_learned_index": float(data['learned_index'].mean()) if 'learned_index' in data.columns else None,
            "std_learned_index": float(data['learned_index'].std()) if 'learned_index' in data.columns else None,
            "min_learned_index": float(data['learned_index'].min()) if 'learned_index' in data.columns else None,
            "max_learned_index": float(data['learned_index'].max()) if 'learned_index' in data.columns else None,
            "significant_comparisons": len(stats_results[stats_results['significant'] == True]) if not stats_results.empty and 'significant' in stats_results.columns else 0
                }
     }
     
     # Add method-specific information
    if analysis_method == 'time':
         metadata["analysis_info"]["description"] = "Project-level time-based individual fly analysis - measures time spent on each side for each fly"
         metadata["data_info"]["fly_level_analysis"] = True
    elif analysis_method == 'time-matlab':
         metadata["analysis_info"]["description"] = "Project-level MATLAB-style time analysis - uses index-summing from timeratio_alistair.m"
         metadata["data_info"]["fly_level_analysis"] = True
    elif analysis_method == 'snapshot':
         metadata["analysis_info"]["description"] = "Project-level snapshot population analysis - measures fly positions at end of phases"
         metadata["data_info"]["population_level_analysis"] = True
    elif analysis_method == 'learning_valence':
         metadata["analysis_info"]["description"] = "Project-level learning valence analysis - measures changes in valence with learning session"
         metadata["data_info"]["fly_level_analysis"] = True
    elif analysis_method == 'valence_habituation':
         metadata["analysis_info"]["description"] = "Project-level valence habituation analysis - measures valence across multiple repeated exposures"
         metadata["data_info"]["fly_level_analysis"] = True
    
    # Save metadata to JSON file
    metadata_filename = f"project_analysis_metadata_{analysis_method}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    metadata_path = os.path.join(output_folder, metadata_filename)
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Project analysis metadata saved to: {metadata_path}")
    return metadata_path


def analyze_experiment_folder(folder_path, config_path="experiment_config.json", 
                            threshold=4, midline_borders=60, filter_phase='both', analysis_method='time', determine_side=10, min_valence_fraction=0.0, time_window=[0, 5]):
    """
    Main function to analyze an entire experiment folder
    
    Parameters:
    -----------
    folder_path : str
        Path to the experiment folder
    config_path : str
        Path to the configuration file
    threshold : int
        Minimum choices required for valid fly
    midline_borders : float
        Midline border threshold (0.0 to 100.0)
    filter_phase : str
        Which phases to filter ('both', 'initial', 'test', or 'none')
    analysis_method : str
        Analysis method to use ('time' for time-based individual fly analysis, 
        'snapshot' for population-level snapshot analysis)
    determine_side : float
        Threshold for determining which side flies are on (0.0 to 100.0)
    min_valence_fraction : float
        Minimum fraction of time a fly must spend in initial valence period to be included in analysis (0.0 to 1.0)
    """
    print("="*80)
    print("MULTIPLEX BATCH ANALYSIS")
    print("="*80)
    
    # Load configuration
    config = load_experiment_config(config_path)
    
    # Override config settings with provided parameters
    config['analysis_settings']['threshold'] = threshold
    config['analysis_settings']['midline_borders'] = midline_borders
    config['analysis_settings']['filter_phase'] = filter_phase
    
    # Collect data from all trials
    print(f"\nCollecting data from: {folder_path}")
    print(f"Filtering parameters: threshold={threshold}, midline_borders={midline_borders}, filter_phase='{filter_phase}'")
    print(f"Analysis parameters: method={analysis_method}, determine_side={determine_side}")
    data = collect_trial_data(folder_path, threshold=threshold, midline_borders=midline_borders, 
                            filter_phase=filter_phase, analysis_method=analysis_method, determine_side=determine_side, min_valence_fraction=min_valence_fraction, time_window=time_window)
    
    if data.empty:
        print("No data collected. Exiting.")
        return
    
    # Perform statistical analysis
    stats_results = perform_statistical_analysis(data, config)
    
    # Save results to CSV and get output folder path
    output_folder = save_results_to_csv(folder_path, data, stats_results)
    
    # Save project analysis metadata
    save_project_metadata(output_folder, config, analysis_method, determine_side, min_valence_fraction, time_window, data, stats_results)
    
    # Create plots in the timestamped output folder
    plots_folder = os.path.join(output_folder, 'plots')
    create_plots(data, stats_results, plots_folder)
    
    print(f"\nAnalysis complete! Results saved to {output_folder}")


# Example usage
if __name__ == "__main__":
    # =============================================================================
    # ANALYSIS PARAMETERS - Modify these as needed
    # =============================================================================
    ANALYSIS_PARAMS = {
        'folder_path': r"D:\multiplex\project_students\Noa\exp_noa\vid_exp",
        'config_path': "experiment_config.json",
        'threshold': 4,                    # Minimum choices required for valid fly
        'midline_borders': 60,           # Midline border threshold for filtering (0.0 to 100.0)
        'filter_phase': 'both',           # Which phases to filter: 'both', 'initial', 'test', or 'none'
        'analysis_method': 'time',    # Analysis method: 'time' for time-based, 'snapshot' for population-level, 'learning_valence', 'valence_habituation', 'time-matlab' for MATLAB-style time analysis
        'determine_side': 0,             # Threshold for determining which side flies are on (0.0 to 100.0)
        'min_valence_fraction': 0,     # Minimum fraction of time in initial valence period for valid fly (0.0 to 1.0)
        'time_window': [80, 110]             # For snapshot analysis: time range [start_seconds, end_seconds] from phase start
    }
    
    # =============================================================================
    # RUN ANALYSIS
    # =============================================================================
    analyze_experiment_folder(
        folder_path=ANALYSIS_PARAMS['folder_path'],
        config_path=ANALYSIS_PARAMS['config_path'],
        threshold=ANALYSIS_PARAMS['threshold'],
        midline_borders=ANALYSIS_PARAMS['midline_borders'],
        filter_phase=ANALYSIS_PARAMS['filter_phase'],
        analysis_method=ANALYSIS_PARAMS['analysis_method'],
        determine_side=ANALYSIS_PARAMS['determine_side'],
        min_valence_fraction=ANALYSIS_PARAMS['min_valence_fraction'],
        time_window=ANALYSIS_PARAMS['time_window']
    )
