# Neural Machine Translation (NMT) - English to French

A modern, production-ready neural machine translation system that translates English text to French using deep learning with attention mechanisms.

## Features

- **Modern Architecture**: Seq2Seq model with LSTM encoder-decoder and attention mechanism
- **Web Interface**: Beautiful, responsive web UI for interactive translation
- **Mock Database**: JSON-based database system for storing translation pairs
- **Model Management**: Save/load trained models with metadata
- **Evaluation Metrics**: Built-in model evaluation and BLEU score calculation
- **Configuration Management**: Centralized configuration with logging
- **API Endpoints**: RESTful API for translation and data management
- **Production Ready**: Proper error handling, logging, and deployment setup

## Usage

### Prerequisites

- Python 3.8+
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd 0102_Basic_neural_machine_translation
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**

   **Option A: Train and test the model**
   ```bash
   python 0102.py
   ```

   **Option B: Run the web interface**
   ```bash
   python 0102.py web
   ```

4. **Access the web interface**
   - Open your browser and go to `http://localhost:5000`
   - Start translating English text to French!

## Project Structure

```
0102_Basic_neural_machine_translation/
├── 0102.py                 # Main application file
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── templates/
│   └── index.html         # Web interface template
├── models/                # Saved models (created automatically)
├── data/                  # Translation database (created automatically)
└── logs/                  # Application logs (created automatically)
```

## Model Architecture

The neural machine translation model uses:

- **Encoder**: LSTM with embedding layer
- **Decoder**: LSTM with attention mechanism
- **Attention**: Bahdanau-style attention for better context
- **Embedding**: 128-dimensional word embeddings
- **Dropout**: 0.2 dropout rate for regularization

### Model Parameters

- **Embedding Dimension**: 128
- **LSTM Units**: 256
- **Dropout Rate**: 0.2
- **Batch Size**: 32
- **Epochs**: 100 (with early stopping)
- **Max Sequence Length**: 20

## 🔧 Configuration

The application uses a centralized configuration system. Key parameters can be modified in the `Config` class:

```python
class Config:
    def __init__(self):
        self.EMBEDDING_DIM = 128
        self.LSTM_UNITS = 256
        self.DROPOUT_RATE = 0.2
        self.BATCH_SIZE = 32
        self.EPOCHS = 100
        self.VALIDATION_SPLIT = 0.2
        self.MAX_SEQUENCE_LENGTH = 20
```

## Web Interface

The web interface provides:

- **Translation**: Real-time English to French translation
- **Add Translations**: Add new translation pairs to the database
- **Statistics**: View database and model statistics
- **Responsive Design**: Works on desktop and mobile devices

### API Endpoints

- `POST /translate` - Translate text
- `POST /add_translation` - Add new translation pair
- `GET /stats` - Get database and model statistics

## Database System

The mock database system stores translation pairs in JSON format:

```json
{
  "pairs": [
    ["hello", "bonjour"],
    ["thank you", "merci"]
  ],
  "metadata": {
    "created": "2024-01-01T00:00:00",
    "total_pairs": 30,
    "source_lang": "en",
    "target_lang": "fr"
  }
}
```

## Model Evaluation

The system includes comprehensive evaluation metrics:

- **Accuracy**: Word-level accuracy on test data
- **BLEU Score**: Standard machine translation evaluation metric
- **Validation**: Built-in validation during training

## Training Process

1. **Data Preparation**: Tokenization and sequence padding
2. **Model Building**: Create encoder-decoder architecture
3. **Training**: Train with early stopping and model checkpointing
4. **Inference Setup**: Build separate models for inference
5. **Evaluation**: Test on validation data

## Deployment

### Local Development

```bash
# Run training
python 0102.py

# Run web server
python 0102.py web
```

### Production Deployment

```bash
# Using Gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 0102:app

# Using Waitress (Windows)
waitress-serve --host=0.0.0.0 --port=5000 0102:app
```

## Usage Examples

### Python API

```python
from 0102 import NeuralMachineTranslation, Config

# Initialize model
config = Config()
nmt = NeuralMachineTranslation(config)

# Load trained model
nmt.load_model()
nmt.build_inference_models()

# Translate text
translation = nmt.translate("hello world")
print(translation)  # Output: "bonjour monde"
```

### Web API

```bash
# Translate text
curl -X POST http://localhost:5000/translate \
  -H "Content-Type: application/json" \
  -d '{"text": "hello world"}'

# Add translation
curl -X POST http://localhost:5000/add_translation \
  -H "Content-Type: application/json" \
  -d '{"source": "goodbye", "target": "au revoir"}'
```

## 🔧 Customization

### Adding New Languages

To add support for new language pairs:

1. Update the dataset in `pairs` variable
2. Modify tokenization settings
3. Adjust model parameters if needed
4. Update the web interface labels

### Improving Model Performance

- Increase training data
- Adjust hyperparameters
- Implement beam search for decoding
- Add more sophisticated attention mechanisms
- Use pre-trained embeddings

## Troubleshooting

### Common Issues

1. **Model not loading**: Ensure model files exist in `models/` directory
2. **Memory errors**: Reduce batch size or sequence length
3. **Poor translations**: Increase training data or epochs
4. **Web interface not loading**: Check Flask installation and port availability

### Logs

Check the `logs/nmt.log` file for detailed error messages and training progress.

## Dependencies

- **TensorFlow**: Deep learning framework
- **Flask**: Web framework
- **NumPy**: Numerical computing
- **NLTK**: Natural language processing
- **Pandas**: Data manipulation
- **Matplotlib**: Visualization

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is open source and available under the MIT License.

## Acknowledgments

- TensorFlow team for the excellent deep learning framework
- The machine translation research community
- Contributors and users of this project

## Support

For questions, issues, or contributions:

- Create an issue on GitHub
- Check the logs for error details
- Review the configuration settings
- Ensure all dependencies are installed


# Neural-Machine-Translation-NMT-
