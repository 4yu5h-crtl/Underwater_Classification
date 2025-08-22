# 🐋 Underwater Acoustic Object Detection & Classification

An AI-powered pipeline for detecting and classifying underwater acoustic objects from raw `.wav` recordings using classical machine learning techniques.

## 🎯 Project Overview

### What This Project Does
This system analyzes underwater acoustic recordings to automatically detect and classify different types of underwater sounds:

- **🚢 Ships & Vessels**: Engine noise, propeller sounds, hull vibrations
- **🐋 Marine Animals**: Whale songs, dolphin clicks, fish sounds
- **🚁 Submarines**: Sonar pings, underwater propulsion systems
- **🌊 Ambient Noise**: Ocean waves, currents, natural underwater sounds

### How It Works
The system processes raw `.wav` files through a complete pipeline:
1. **Preprocessing** → Clean and filter audio (noise reduction, band-pass filtering)
2. **Feature Extraction** → Extract numerical features (MFCCs, spectral properties)
3. **Classification** → Train ML models to predict sound categories
4. **Evaluation** → Assess model performance with metrics
5. **Inference** → Classify new audio files in real-time

### Why Numerical Features?
Instead of complex spectrogram images, we extract **numerical features** that are:
- **Faster** to compute and process
- **More interpretable** (you can see which features matter most)
- **Easier to deploy** (no deep learning dependencies)
- **Better for small datasets** (classical ML works well with limited data)

## 📁 Project Structure

```
underwater-classification/
├── data/
│   ├── raw/                   # Raw .wav files organized by class
│   │   ├── ship/              # Ship and vessel recordings
│   │   ├── animal/            # Marine mammal and fish sounds
│   │   ├── submarine/         # Submarine and underwater vehicle sounds
│   │   └── noise/             # Ambient underwater noise
│   ├── processed/             # Cleaned and filtered audio files
│   └── features/              # Extracted numerical features
│       └── features.csv       # Main features file with labels
├── src/                       # Python source code
│   ├── preprocessing.py       # Audio cleaning and filtering
│   ├── feature_extraction.py  # Extract MFCCs and spectral features
│   ├── anomaly_detection.py   # Detect interesting audio segments
│   ├── train_classifier.py    # Train ML models
│   ├── evaluate.py            # Evaluate model performance
│   └── inference.py           # Run classification on new files
├── models/                    # Trained machine learning models
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## 🎵 Dataset Preparation

### 1. Organize Your Audio Files
Place your `.wav` files in the appropriate subfolders:

```bash
data/raw/
├── ship/
│   ├── ship_001.wav
│   ├── ship_002.wav
│   └── ...
├── animal/
│   ├── whale_001.wav
│   ├── dolphin_001.wav
│   └── ...
├── submarine/
│   ├── sub_001.wav
│   ├── sub_002.wav
│   └── ...
└── noise/
    ├── ambient_001.wav
    ├── waves_001.wav
    └── ...
```

### 2. Audio File Requirements
For best results, standardize your audio files:

- **Format**: `.wav` files (16-bit PCM recommended)
- **Channels**: Mono (single channel)
- **Sample Rate**: 16 kHz or 22.05 kHz
- **Duration**: 3-5 seconds per clip (can be adjusted)
- **Quality**: Clear recordings with minimal background noise

### 3. File Naming Convention
Use descriptive names that include the class:
- `ship_engine_001.wav`
- `whale_song_001.wav`
- `submarine_sonar_001.wav`
- `ocean_waves_001.wav`

### 4. Handling Different Audio Lengths
- **Long recordings**: Split into 3-5 second chunks
- **Short clips**: Pad with silence to reach minimum duration
- **Variable lengths**: The system can handle different durations

## 🔧 Feature Extraction

### What Features Are Extracted
The system automatically extracts these numerical features from each audio file:

1. **MFCCs (Mel-Frequency Cepstral Coefficients)**
   - 13 coefficients that capture the "fingerprint" of the sound
   - Statistical measures: mean, std, min, max

2. **Spectral Features**
   - **Centroid**: Center of mass of the spectrum
   - **Bandwidth**: Spread of frequencies around the centroid
   - **Rolloff**: Frequency below which 85% of energy is contained

3. **Zero-Crossing Rate**
   - Measures how often the audio signal crosses zero
   - Indicates the "noisiness" of the signal

### Running Feature Extraction
```bash
# Extract features from all audio files
python src/feature_extraction.py

# Custom input/output directories
python src/feature_extraction.py --input data/raw/ --output data/features/my_features.csv

# Save feature scaler for later use
python src/feature_extraction.py --save-scaler
```

**Output**: `data/features/features.csv` with columns like:
```csv
filename,label,mfccs_mean,mfccs_std,spectral_centroid_mean,spectral_bandwidth_mean,...
ship_001.wav,ship,0.123,0.456,0.789,0.012,...
whale_001.wav,animal,0.234,0.567,0.890,0.123,...
```

## 🤖 Model Training

### Training Process
The system trains a **Random Forest Classifier** that:
- Learns patterns from your extracted features
- Handles multiple classes automatically
- Provides feature importance rankings
- Saves the trained model for later use

### Running Training
```bash
# Train with default settings
python src/train_classifier.py

# Custom features file
python src/train_classifier.py --features data/features/my_features.csv

# Custom train/test split
python src/train_classifier.py --test-size 0.3

# Generate and save plots
python src/train_classifier.py --save-plots
```

**Output**: Trained models saved in `models/` folder:
- `random_forest_YYYYMMDD_HHMMSS.pkl` - The trained classifier
- `scaler_YYYYMMDD_HHMMSS.pkl` - Feature normalization scaler
- `label_encoder_YYYYMMDD_HHMMSS.pkl` - Class label encoder
- `evaluation_results_YYYYMMDD_HHMMSS.json` - Training metrics

## 📊 Model Evaluation

### What Gets Evaluated
The evaluation provides comprehensive performance metrics:

- **Accuracy**: Overall correct predictions
- **Precision**: How many predictions were actually correct
- **Recall**: How many actual instances were found
- **F1-Score**: Balanced measure of precision and recall
- **Confusion Matrix**: Detailed breakdown of predictions vs. actual

### Running Evaluation
```bash
# Evaluate the trained model
python src/evaluate.py

# Custom model directory
python src/evaluate.py --model-dir models/my_experiment

# Save evaluation plots
python src/evaluate.py --save-plots
```

**Output**: 
- Console summary of performance metrics
- Confusion matrix visualization
- ROC curves (if probabilities available)
- Detailed evaluation report saved as text file

## 🚀 Inference (Real-Time Classification)

### How Inference Works
The inference pipeline takes a new `.wav` file and:
1. **Preprocesses** the audio (cleaning, filtering)
2. **Detects anomalies** (interesting sound segments)
3. **Extracts features** (same features used in training)
4. **Classifies** using your trained model
5. **Outputs results** with timestamps and predictions

### Running Inference
```bash
# Classify a single audio file
python src/inference.py data/raw/new_recording.wav

# Custom model directory
python src/inference.py --model-dir models/my_model data/raw/new_recording.wav

# Save results to file
python src/inference.py --output results.json data/raw/new_recording.wav
```

**Output**: JSON with detection results:
```json
{
  "file": "new_recording.wav",
  "detections": [
    {
      "start": 15.2,
      "end": 18.7,
      "label": "ship",
      "confidence": 0.89
    },
    {
      "start": 45.1,
      "end": 48.3,
      "label": "animal",
      "confidence": 0.76
    }
  ]
}
```

## 🛠 Installation & Setup

### 1. Clone the Repository
```bash
git clone <your-repository-url>
cd underwater-classification
```

### 2. Install Dependencies
```bash
# Install Python packages
pip install -r requirements.txt

# Or if using conda
conda create -n underwater python=3.10
conda activate underwater
pip install -r requirements.txt
```

### 3. Verify Installation
```bash
# Check if all modules can be imported
python -c "import librosa, sklearn, pandas, numpy; print('All packages installed successfully!')"
```

## 📋 Complete Workflow Example

### Step-by-Step Pipeline

```bash
# 1. Prepare your data
# Place .wav files in data/raw/ subfolders (ship/, animal/, submarine/, noise/)

# 2. Extract features from all audio files
python src/feature_extraction.py

# 3. Train the classifier
python src/train_classifier.py

# 4. Evaluate model performance
python src/evaluate.py

# 5. Run inference on new audio
python src/inference.py data/raw/test_recording.wav
```

### Expected Output Files
```
data/features/features.csv        # Extracted features
models/                           # Trained models
├── random_forest_*.pkl           # Classifier
├── scaler_*.pkl                  # Feature scaler
├── label_encoder_*.pkl           # Label encoder
└── evaluation_results_*.json     # Training metrics
```

## 📚 Dataset Sources


### Data Organization Tips
- **Consistent naming**: Use clear, descriptive filenames
- **Quality control**: Remove very noisy or unclear recordings
- **Balanced classes**: Try to have similar numbers of samples per class
- **Validation set**: Keep some files separate for final testing

## 🔍 Troubleshooting

### Common Issues

**"No module named 'librosa'"**
```bash
pip install librosa
```

**"Audio file not found"**
- Check file paths are correct
- Ensure files are in the right subfolders
- Verify file extensions are `.wav`

**"Model training failed"**
- Check that `features.csv` exists and has data
- Ensure there are enough samples per class
- Verify all features are numerical

**"Inference pipeline error"**
- Make sure trained models exist in `models/` folder
- Check that input audio file is valid `.wav`
- Verify model and feature extraction use same parameters

### Getting Help
- Check the console output for detailed error messages
- Verify all dependencies are installed correctly
- Ensure your audio files meet the format requirements
- Check that the project structure matches the expected layout

## 🚀 Advanced Usage

### Custom Feature Extraction
```python
from src.feature_extraction import FeatureExtractor

extractor = FeatureExtractor(
    sample_rate=16000,      # Custom sample rate
    n_mfcc=20,             # More MFCC coefficients
    hop_length=256         # Smaller hop for more detail
)
```

### Custom Model Training
```python
from src.train_classifier import UnderwaterClassifier

classifier = UnderwaterClassifier(
    random_state=42,        # Reproducible results
    test_size=0.3          # 30% for testing
)
```

### Batch Processing
```bash
# Process multiple directories
for dir in ship animal submarine noise; do
    python src/feature_extraction.py --input "data/raw/$dir" --output "data/features/${dir}_features.csv"
done
```

## 📈 Performance Tips

### For Better Results
- **Clean audio**: Minimize background noise in recordings
- **Balanced dataset**: Similar number of samples per class
- **Feature engineering**: Experiment with different MFCC coefficients
- **Hyperparameter tuning**: Adjust Random Forest parameters
- **Cross-validation**: Use multiple train/test splits

### Scaling Up
- **Large datasets**: Process files in batches
- **Parallel processing**: Use multiple CPU cores
- **Memory management**: Process one file at a time for very large datasets


## 📝 License

This project is open source. Please check individual dependencies for their respective licenses.