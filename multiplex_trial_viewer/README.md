# Multiplex Trial Viewer

A PyQt5 desktop application for viewing and analyzing multiplex trial videos with fly detection and location tracking.

## Features

- **Video Playback**: Load and play pre-recorded trial videos with full playback controls
- **Fly Detection**: Real-time fly detection using background subtraction algorithms
- **Location Tracking**: Track fly positions within 1D chambers (-100 to +100 scale)
- **Trace Visualization**: Display synchronized trace plots showing fly movement over time
- **Export Capabilities**: Export data as CSV, images, and trace plots
- **Chamber Configuration**: Load and edit chamber boundary configurations

## Installation

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Application**:
   ```bash
   python trial_viewer_app.py
   ```

## Usage

### Loading a Video
1. Click "File" → "Open Video" or press Ctrl+O
2. Select a video file (MP4, AVI, MOV, MKV supported)
3. Wait for processing to complete (progress bar will show status)

### Video Controls
- **Play/Pause**: Toggle video playback
- **Frame Navigation**: Use << and >> buttons for frame-by-frame navigation
- **Seek Slider**: Drag to jump to any point in the video
- **Frame Info**: Current frame number and total frames displayed

### Trace Analysis
- **Chamber Selection**: Use checkboxes to select which chambers to display
- **Time Synchronization**: Red vertical line shows current video timestamp
- **Trace Colors**: Each chamber has a unique color for easy identification

### Export Options
- **CSV Export**: Export all detected fly locations with timestamps
- **Frame Export**: Save current frame as annotated image
- **Trace Plot Export**: Export trace visualization as PNG/PDF/SVG

## File Structure

```
multiplex_trial_viewer/
├── trial_viewer_app.py          # Main application window
├── video_processor.py           # Video loading & processing
├── fly_detection_core.py        # Fly detection algorithms
├── trace_plotter.py             # Trace visualization
├── chamber_config.py            # Chamber configuration management
├── chambers_configuration.json  # Chamber boundary coordinates
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Components

### Fly Detection Core (`fly_detection_core.py`)
- Extracted from `experiment_dashboard.py`
- Background subtraction with running average
- Contour detection and filtering
- Chamber assignment and location calculation

### Video Processor (`video_processor.py`)
- Video loading and metadata extraction
- Batch processing with progress tracking
- Frame caching for smooth playback
- CSV export functionality

### Trace Plotter (`trace_plotter.py`)
- Matplotlib canvas for PyQt integration
- Multi-chamber trace overlay
- Time synchronization with video
- Interactive chamber selection

### Chamber Configuration (`chamber_config.py`)
- JSON-based configuration management
- Chamber boundary validation
- Manual coordinate editing support

## Technical Details

- **Detection Algorithm**: Background subtraction with morphological operations
- **Location Scale**: -100 to +100 within each chamber (1D positioning)
- **Video Formats**: MP4, AVI, MOV, MKV support
- **Processing**: Pre-processes entire video for smooth playback
- **Threading**: Background processing to prevent UI blocking

## Requirements

- Python 3.7+
- PyQt5
- OpenCV
- NumPy
- Pandas
- Matplotlib

## Testing

Run the test script to verify all components work correctly:

```bash
python test_app.py
```

## Troubleshooting

1. **Video Loading Issues**: Ensure video file is not corrupted and format is supported
2. **Processing Errors**: Check that chamber configuration is valid
3. **Display Issues**: Verify PyQt5 installation and display drivers
4. **Memory Issues**: Large videos may require significant RAM for processing

## Development

The application follows a modular architecture:
- Backend components handle data processing
- Frontend components manage user interface
- Clear separation of concerns for maintainability

Each component can be tested independently using the provided test functions.
