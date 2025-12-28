# flappy-bird-ai

A computer vision-based AI system for playing Flappy Bird using template matching.

## Features

- **Template Matching**: Uses OpenCV to detect game elements (pipes) in screenshots
- **Multi-Template Support**: Can detect multiple types of game elements simultaneously
- **Visual Labeling**: Labels detected pipes with different colors for easy visualization
- **Confidence Scoring**: Provides confidence scores for each detection

## Usage

1. Take a screenshot of the Flappy Bird game
2. Place pipe template images (`pipe-up.png`, `pipe-down.png`) in the project directory
3. Run the script:

```bash
python main.py
```

## Output

The script will:
- Detect all pipes in the screenshot using template matching
- Label pipes with different colors:
  - **Green rectangles**: Upward-facing pipes
  - **Red rectangles**: Downward-facing pipes
- Save a labeled image as `labeled_pipes.png`
- Display detection results with confidence scores

## Requirements

- Python 3.x
- OpenCV (`pip install opencv-python`)
- NumPy (usually comes with OpenCV)

## Files

- `main.py`: Main script with template matching functions
- `pipe-up.png`: Template for upward-facing pipes
- `pipe-down.png`: Template for downward-facing pipes
- `Screenshot_2025.12.28_00.54.50.642.png`: Example game screenshot
- `labeled_pipes.png`: Output with labeled detections