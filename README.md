# Offroad Segmentation and Path Planning

This project implements an autonomous offroad navigation system that combines semantic segmentation with terrain-aware path planning. It uses a SegFormer model to classify terrain types from images and an A* based planner to find optimal paths, avoiding obstacles and prioritizing safer terrain.

## Output Video

[Click Me to View Demo Video](https://drive.google.com/file/d/1xB-DVqyLfOaZR0ry7qNas0xq4Gl2lCSV/view?usp=sharing)

## Features

### 1. Semantic Segmentation
- **Model**: Uses `nvidia/mit-b0` (SegFormer) for efficient and accurate semantic segmentation.
- **Classes**:
  - `Obstacle` (ID: 0): Non-traversable objects.
  - `Landscape` (ID: 1): Safe, flat terrain.
  - `Grass` (ID: 2): Traversable but higher cost than landscape.
  - `Rough` (ID: 3): Difficult terrain, high traversal cost.
- **Training**: Custom training loop with MPS (Apple Silicon) and CPU support, utilizing Hugging Face `transformers` and `evaluate` libraries.
- **Metrics**: Evaluates using Mean IoU (Intersection over Union).

### 2. Terrain-Aware Path Planning
- **Algorithm**: A* Search with dynamic edge costs based on terrain semantic class.
- **Cost Map**:
  - Dynamic cost allocation for `Landscape`, `Grass`, and `Rough` terrain.
  - Repulsion field to keep a safe distance from obstacles.
  - Inflation radius to account for robot size.
- **Recovery Strategies**:
  - **Standard**: Optimal path with full safety margins.
  - **Reduced Safety**: Reduces safety margins if no path is found.
  - **Tunneled Costs**: Allows traversing "soft" obstacles with high penalties if necessary.
  - **Desperation Mode**: Generates a straight line path if all else fails (last resort).

## Directory Structure

```
Hackathon/
├── Offroad_Segmentation_Scripts/
│   ├── train_segmentation.py    # Main script for training the SegFormer model
│   ├── path_planner.py          # A* Path Planner execution and logic
│   ├── outputs/                 # Directory for outputted paths and logs
│   ├── segformer_offroad_output/# Checkpoints and saved models
│   └── ...
├── PathPlanner/                 # Contains saved best models
└── ...
```
```

## Dataset

The dataset used for this project is available on the Falcon Duality website.

[Download Dataset](https://falcon.duality.ai/secure/documentation/hackathon-segmentation-desert?utm_source=hackathon&utm_medium=instructions&utm_campaign=technomania)

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd Hackathon
    ```

2.  **Install Dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    pip install torch torchvision transformers evaluate datasets opencv-python numpy matplotlib scipy pillow
    ```

## Usage

### Training the Segmentation Model

To train the model on your dataset:

1.  Ensure your dataset is organized in `../Offroad_Segmentation_Training_Dataset` with `train` and `val` subfolders containing `Color_Images` and `Segmentation` masks.
2.  Run the training script:
    ```bash
    python Hackathon/Offroad_Segmentation_Scripts/train_segmentation.py
    ```
    *The script supports MPS acceleration for macOS users automatically.*

### Running the Path Planner

To run the path planning algorithm on segmented outputs:

```bash
python Hackathon/Offroad_Segmentation_Scripts/path_planner.py
```

## Configuration

-   **Training Config**: Modify `train_segmentation.py` to adjust `BATCH_SIZE`, `LEARNING_RATE`, `EPOCHS`, or dataset paths.
-   **Planner Config**: Modify `PlannerConfig` class in `path_planner.py` to tune:
    -   `COST_*` weights for different terrains.
    -   `ROBOT_RADIUS_PX` for obstacle avoidance.
    -   `SAFETY_MARGIN_FACTOR` for safety buffers.

## Contribution

Feel free to fork this repository and submit pull requests. For major changes, please open an issue first to discuss what you would like to change.
