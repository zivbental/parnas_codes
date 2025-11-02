<p align="center">
  <img src="https://i.imgur.com/yhcZ4Tl.png" alt="Parnas Lab Code Utilities logo showing a Drosophila coding" width="600"/>
</p>

# Parnas Lab Code Utilities

This repository contains all code developed in the **Moshe Parnas Lab** for various research projects and data analysis tasks. The repository is organized into specialized modules for different types of experiments and analyses.

## 📁 Repository Structure

### 🧠 **Behavior Analysis** (`behavior_analysis/`)
Comprehensive Python package for analyzing Drosophila learning behavior in multiplex experiments.

**Key Features:**
- Automatic CS+/CS- detection for operant and classical conditioning
- Side-switching support for complex experimental protocols
- Statistical analysis with assumption checking
- Batch processing for entire experiments
- Publication-ready visualizations

**Main Files:**
- `multiplex_core.py` - Core analysis functionality
- `multiplex_v2_analysis.py` - Main batch analysis script
- `multiplex_v2_simple_analysis.py` - Simplified analysis version
- `multiplex_v2_trial_analysis.py` - Individual trial analysis

**Requirements:** numpy, pandas, matplotlib, seaborn, scipy

---

### 🔬 **Patch Clamp Analysis** (`patch_clamp_analysis/`)

#### Perfusion System (`perfusion_system/`)
Analysis tools for dose-response experiments and patch clamp data processing.

**Features:**
- Dose-response analysis templates
- Data cleaning and processing utilities
- Excel-based data organization

**Files:**
- `dose_response_template.ipynb` - Jupyter notebook for dose-response analysis
- `dop2r dose-response.xlsx` - Sample dose-response data
- `patch_data.xlsx` - Patch clamp experimental data

#### PicoSpritzer (`picosprizer/`)
Automated analysis pipeline for PicoSpritzer pressure ejection experiments.

**Features:**
- Automated peak detection in electrophysiological recordings
- Batch processing of .abf files
- Statistical analysis and visualization
- Excel output with organized results

**Files:**
- `findPeaks.py` - Main analysis script for peak detection and data processing

**Requirements:** numpy, matplotlib, pandas, pyabf, scipy

---

### 🪰 **T-Maze Fly Counting** (`t_maze_countflies/`)
Computer vision tools for automated fly counting in T-maze behavioral experiments.

**Features:**
- Image processing for fly detection
- Automated counting algorithms
- Visualization of counting results

**Files:**
- `main.py` - Main counting script
- `functions.py` - Image processing functions
- `source_img.png` - Sample image for processing

**Requirements:** PIL (Pillow), matplotlib, numpy

---

### 📤 **Uploads** (`uploads/`)
Temporary storage for custom logic and experimental data files.

**Files:**
- `custom_logic.py` - Custom analysis scripts
- `data.txt` - Experimental data files

---

## 🚀 Getting Started

### Prerequisites
- Python 3.7+
- Virtual environment (recommended)

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/parnas-lab-code-utilities.git
cd parnas-lab-code-utilities
```

2. **Set up virtual environment:**
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

3. **Install dependencies for specific modules:**
```bash
# For behavior analysis
cd behavior_analysis
pip install -r requirements.txt

# For patch clamp analysis
cd ../patch_clamp_analysis
pip install numpy matplotlib pandas pyabf scipy

# For T-maze analysis
cd ../t_maze_countflies
pip install pillow matplotlib numpy
```

## 📊 Usage Examples

### Behavior Analysis
```python
from behavior_analysis.multiplex_core import MultiplexTrial

# Load and analyze a trial
trial = MultiplexTrial()
trial.load_data("path/to/fly_loc.csv")
trial.filter_by_num_choices(midline_borders=0.6, threshold=4)
results = trial.analyse_time()
```

### Patch Clamp Analysis
```python
# Run PicoSpritzer analysis
python patch_clamp_analysis/picosprizer/findPeaks.py
```

### T-Maze Counting
```python
# Run fly counting analysis
python t_maze_countflies/main.py
```

## 🔧 Configuration

Each module has its own configuration requirements:

- **Behavior Analysis:** Configure via `experiment_config.json`
- **Patch Clamp:** Uses file naming conventions for automatic processing
- **T-Maze:** Image-based processing with customizable parameters

## 📈 Data Formats

### Behavior Analysis
- **Input:** `fly_loc.csv` with timestamp, location, and stimulus data
- **Output:** Statistical analysis results and publication-ready plots

### Patch Clamp
- **Input:** `.abf` files with electrophysiological recordings
- **Output:** Excel files with peak analysis and statistical results

### T-Maze
- **Input:** Image files (PNG, JPG) of T-maze experiments
- **Output:** Fly counts and visualization overlays

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-analysis`)
3. Commit your changes (`git commit -m 'Add new analysis feature'`)
4. Push to the branch (`git push origin feature/new-analysis`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](behavior_analysis/LICENSE) file for details.

## 📞 Contact

**Moshe Parnas Lab**
- For questions about specific analyses, please refer to the individual module READMEs
- For general repository questions, contact: [lab.email@domain.com]

## 🙏 Acknowledgments

- Built for Drosophila and electrophysiological research in the Parnas Lab
- Designed to support reproducible research practices
- Inspired by classical behavioral and electrophysiological paradigms

---

## 📚 Additional Resources

- [Behavior Analysis Documentation](behavior_analysis/README.md)
- [Patch Clamp Analysis Examples](patch_clamp_analysis/)
- [T-Maze Counting Guide](t_maze_countflies/)

*This repository is actively maintained and updated with new analysis tools and improvements.*
