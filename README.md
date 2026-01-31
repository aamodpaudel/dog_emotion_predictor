# Dog Emotion Recognition Engine

Link to the dog emotions datasets for training: https://www.kaggle.com/datasets/danielshanbalico/dog-emotion
Note: We haven't provided the trained model here, you need to retrain the model to run the streamlit file. 

An AI-powered application designed to analyze and classify the emotional state of dogs from facial images. This project utilizes a fine-tuned ResNet-18 architecture to detect emotions with high accuracy.

## Features
- **Emotion Classification**: Detects 4 distinct emotional states: Happy, Sad, Angry, and Relaxed.
- **Deep Learning Model**: Built on top of ResNet-18 using PyTorch.
- **Interactive UI**: User-friendly web interface powered by Streamlit.
- **Visual Analytics**: Probability breakdown charts using Plotly.

## Tech Stack
- **Languages**: Python
- **Frameworks**: PyTorch, Streamlit
- **Libraries**: Pandas, Plotly, Pillow, OpenCV, Scikit-learn, Timm

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/aamodpaudel/dog_emotion_predictor.git
   cd dog_emotion_predictor
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # Mac/Linux
   source .venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Web App
To launch the interactive interface:
```bash
streamlit run app.py
```

### Training the Model
To retrain the model on your own dataset:
1. Ensure your dataset is organized in `datasets/` with subfolders matching the class names (happy, sad, angry, relaxed).
2. Run the training script:
   ```bash
   python train.py
   ```
   This will train the model in two stages:
   - **Stage 1**: Training the head with a frozen backbone (10 epochs).
   - **Stage 2**: Fine-tuning the entire network with a lower learning rate.

## Model Details
- **Architecture**: ResNet-18
- **Input Size**: 224x224 pixels
- **Classes**: Happy, Sad, Angry, Relaxed
- **Device**: Automatially selects CUDA if available, otherwise uses CPU.

## License
This project is for academic and research purposes.
