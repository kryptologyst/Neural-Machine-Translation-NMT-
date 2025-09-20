# Project 102. Enhanced Neural Machine Translation
# Description:
# Neural Machine Translation (NMT) is the task of automatically translating text from one language to another using neural networks. 
# This enhanced version implements a modern sequence-to-sequence model with attention mechanism and improved architecture.

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, Attention, Concatenate, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import json
import os
import logging
from datetime import datetime
import pickle

# Configuration class
class Config:
    def __init__(self):
        self.EMBEDDING_DIM = 128
        self.LSTM_UNITS = 256
        self.DROPOUT_RATE = 0.2
        self.BATCH_SIZE = 32
        self.EPOCHS = 100
        self.VALIDATION_SPLIT = 0.2
        self.MAX_SEQUENCE_LENGTH = 20
        self.MODEL_DIR = "models"
        self.DATA_DIR = "data"
        self.LOG_DIR = "logs"
        
        # Create directories if they don't exist
        for dir_path in [self.MODEL_DIR, self.DATA_DIR, self.LOG_DIR]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.LOG_DIR, 'nmt.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

# Initialize configuration
config = Config()

# Enhanced bilingual sentence pairs with more examples
# Enhanced bilingual sentence pairs with more examples
pairs = [
    ("hello", "bonjour"),
    ("how are you", "comment ça va"),
    ("good morning", "bonjour"),
    ("thank you", "merci"),
    ("i am fine", "je vais bien"),
    ("see you later", "à plus tard"),
    ("what is your name", "comment tu t'appelles"),
    ("i love you", "je t'aime"),
    ("goodbye", "au revoir"),
    ("yes", "oui"),
    ("no", "non"),
    ("please", "s'il vous plaît"),
    ("excuse me", "excusez-moi"),
    ("i don't understand", "je ne comprends pas"),
    ("can you help me", "pouvez-vous m'aider"),
    ("where is the bathroom", "où sont les toilettes"),
    ("how much does it cost", "combien ça coûte"),
    ("i would like to order", "je voudrais commander"),
    ("the weather is nice", "il fait beau"),
    ("i am learning french", "j'apprends le français"),
    ("this is delicious", "c'est délicieux"),
    ("i am sorry", "je suis désolé"),
    ("have a nice day", "bonne journée"),
    ("see you tomorrow", "à demain"),
    ("i miss you", "tu me manques"),
    ("happy birthday", "bon anniversaire"),
    ("congratulations", "félicitations"),
    ("good luck", "bonne chance"),
    ("i am tired", "je suis fatigué"),
    ("let's go", "allons-y")
]

# Mock Database System
class TranslationDatabase:
    def __init__(self, db_path="data/translations.json"):
        self.db_path = db_path
        self.translations = self.load_translations()
    
    def load_translations(self):
        """Load translations from JSON file or create default dataset"""
        if os.path.exists(self.db_path):
            with open(self.db_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            # Create default dataset
            default_data = {
                "pairs": pairs,
                "metadata": {
                    "created": datetime.now().isoformat(),
                    "total_pairs": len(pairs),
                    "source_lang": "en",
                    "target_lang": "fr"
                }
            }
            self.save_translations(default_data)
            return default_data
    
    def save_translations(self, data=None):
        """Save translations to JSON file"""
        if data is None:
            data = self.translations
        with open(self.db_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def add_translation(self, source, target):
        """Add a new translation pair"""
        self.translations["pairs"].append((source, target))
        self.translations["metadata"]["total_pairs"] = len(self.translations["pairs"])
        self.translations["metadata"]["last_updated"] = datetime.now().isoformat()
        self.save_translations()
        config.logger.info(f"Added translation: {source} -> {target}")
    
    def get_pairs(self):
        """Get all translation pairs"""
        return self.translations["pairs"]
    
    def get_stats(self):
        """Get database statistics"""
        return self.translations["metadata"]

# Initialize database
db = TranslationDatabase()
pairs = db.get_pairs()
 
# Enhanced Neural Machine Translation Model Class
class NeuralMachineTranslation:
    def __init__(self, config):
        self.config = config
        self.eng_tokenizer = None
        self.fr_tokenizer = None
        self.model = None
        self.encoder_model = None
        self.decoder_model = None
        self.max_eng_len = 0
        self.max_fr_len = 0
        self.eng_vocab_size = 0
        self.fr_vocab_size = 0
        
    def prepare_data(self, pairs):
        """Prepare and tokenize the training data"""
        config.logger.info("Preparing training data...")
        
        # Split into source and target
        eng_sentences, fr_sentences = zip(*pairs)
        fr_sentences = ['<sos> ' + text + ' <eos>' for text in fr_sentences]
        
        # Tokenize English
        self.eng_tokenizer = Tokenizer(filters='', lower=True)
        self.eng_tokenizer.fit_on_texts(eng_sentences)
        eng_seq = self.eng_tokenizer.texts_to_sequences(eng_sentences)
        self.eng_vocab_size = len(self.eng_tokenizer.word_index) + 1
        self.max_eng_len = min(max(len(seq) for seq in eng_seq), config.MAX_SEQUENCE_LENGTH)
        
        # Tokenize French
        self.fr_tokenizer = Tokenizer(filters='', lower=True)
        self.fr_tokenizer.fit_on_texts(fr_sentences)
        fr_seq = self.fr_tokenizer.texts_to_sequences(fr_sentences)
        self.fr_vocab_size = len(self.fr_tokenizer.word_index) + 1
        self.max_fr_len = min(max(len(seq) for seq in fr_seq), config.MAX_SEQUENCE_LENGTH)
        
        # Prepare sequences
        X = pad_sequences(eng_seq, maxlen=self.max_eng_len, padding='post')
        
        # Prepare decoder input and target sequences
        decoder_input_seq = [seq[:-1] for seq in fr_seq]
        decoder_target_seq = [seq[1:] for seq in fr_seq]
        
        decoder_input = pad_sequences(decoder_input_seq, maxlen=self.max_fr_len - 1, padding='post')
        decoder_target = pad_sequences(decoder_target_seq, maxlen=self.max_fr_len - 1, padding='post')
        decoder_target = tf.keras.utils.to_categorical(decoder_target, num_classes=self.fr_vocab_size)
        
        config.logger.info(f"Data prepared - EN vocab: {self.eng_vocab_size}, FR vocab: {self.fr_vocab_size}")
        config.logger.info(f"Max lengths - EN: {self.max_eng_len}, FR: {self.max_fr_len}")
        
        return X, decoder_input, decoder_target
    
    def build_model(self):
        """Build the enhanced NMT model with attention"""
        config.logger.info("Building NMT model with attention mechanism...")
        
        # Encoder
        encoder_inputs = Input(shape=(self.max_eng_len,), name='encoder_inputs')
        enc_emb = Embedding(self.eng_vocab_size, config.EMBEDDING_DIM, 
                           mask_zero=True, name='encoder_embedding')(encoder_inputs)
        enc_lstm = LSTM(config.LSTM_UNITS, return_sequences=True, return_state=True, 
                       dropout=config.DROPOUT_RATE, name='encoder_lstm')
        encoder_outputs, state_h, state_c = enc_lstm(enc_emb)
        encoder_states = [state_h, state_c]
        
        # Decoder
        decoder_inputs = Input(shape=(self.max_fr_len - 1,), name='decoder_inputs')
        dec_emb = Embedding(self.fr_vocab_size, config.EMBEDDING_DIM, 
                           mask_zero=True, name='decoder_embedding')(decoder_inputs)
        dec_lstm = LSTM(config.LSTM_UNITS, return_sequences=True, return_state=True, 
                       dropout=config.DROPOUT_RATE, name='decoder_lstm')
        decoder_outputs, _, _ = dec_lstm(dec_emb, initial_state=encoder_states)
        
        # Attention mechanism
        attention = Attention(name='attention')
        context_vector = attention([decoder_outputs, encoder_outputs])
        
        # Concatenate context vector with decoder output
        decoder_concat = Concatenate(axis=-1, name='decoder_concat')
        decoder_concat_input = decoder_concat([decoder_outputs, context_vector])
        
        # Dense layers
        decoder_dense1 = Dense(config.LSTM_UNITS, activation='relu', name='decoder_dense1')
        decoder_dense2 = Dense(self.fr_vocab_size, activation='softmax', name='decoder_dense2')
        
        decoder_outputs = decoder_dense1(decoder_concat_input)
        decoder_outputs = Dropout(config.DROPOUT_RATE)(decoder_outputs)
        decoder_outputs = decoder_dense2(decoder_outputs)
        
        # Create model
        self.model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        
        # Compile model
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        config.logger.info("Model built successfully!")
        return self.model
    
    def train(self, X, decoder_input, decoder_target):
        """Train the model with callbacks"""
        config.logger.info("Starting model training...")
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ModelCheckpoint(
                filepath=os.path.join(config.MODEL_DIR, 'best_model.h5'),
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train model
        history = self.model.fit(
            [X, decoder_input], decoder_target,
            batch_size=config.BATCH_SIZE,
            epochs=config.EPOCHS,
            validation_split=config.VALIDATION_SPLIT,
            callbacks=callbacks,
            verbose=1
        )
        
        config.logger.info("Training completed!")
        return history
    
    def build_inference_models(self):
        """Build models for inference"""
        config.logger.info("Building inference models...")
        
        # Encoder inference model
        self.encoder_model = Model(self.model.get_layer('encoder_inputs').input, 
                                 [self.model.get_layer('encoder_lstm').output, 
                                  self.model.get_layer('encoder_lstm').states])
        
        # Decoder inference model
        decoder_state_input_h = Input(shape=(config.LSTM_UNITS,))
        decoder_state_input_c = Input(shape=(config.LSTM_UNITS,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        
        decoder_inputs = Input(shape=(1,))
        dec_emb = self.model.get_layer('decoder_embedding')(decoder_inputs)
        dec_lstm = self.model.get_layer('decoder_lstm')
        decoder_outputs, state_h, state_c = dec_lstm(dec_emb, initial_state=decoder_states_inputs)
        decoder_states = [state_h, state_c]
        
        # Attention for inference
        encoder_outputs = Input(shape=(self.max_eng_len, config.LSTM_UNITS))
        attention = self.model.get_layer('attention')
        context_vector = attention([decoder_outputs, encoder_outputs])
        
        # Concatenate and dense layers
        decoder_concat = self.model.get_layer('decoder_concat')
        decoder_concat_input = decoder_concat([decoder_outputs, context_vector])
        
        decoder_dense1 = self.model.get_layer('decoder_dense1')
        decoder_dense2 = self.model.get_layer('decoder_dense2')
        
        decoder_outputs = decoder_dense1(decoder_concat_input)
        decoder_outputs = decoder_dense2(decoder_outputs)
        
        self.decoder_model = Model(
            [decoder_inputs, encoder_outputs] + decoder_states_inputs,
            [decoder_outputs] + decoder_states
        )
        
        config.logger.info("Inference models built successfully!")
    
    def save_model(self, filepath=None):
        """Save the trained model and tokenizers"""
        if filepath is None:
            filepath = os.path.join(config.MODEL_DIR, 'nmt_model.h5')
        
        self.model.save(filepath)
        
        # Save tokenizers
        with open(os.path.join(config.MODEL_DIR, 'eng_tokenizer.pkl'), 'wb') as f:
            pickle.dump(self.eng_tokenizer, f)
        
        with open(os.path.join(config.MODEL_DIR, 'fr_tokenizer.pkl'), 'wb') as f:
            pickle.dump(self.fr_tokenizer, f)
        
        # Save model metadata
        metadata = {
            'max_eng_len': self.max_eng_len,
            'max_fr_len': self.max_fr_len,
            'eng_vocab_size': self.eng_vocab_size,
            'fr_vocab_size': self.fr_vocab_size,
            'config': {
                'EMBEDDING_DIM': config.EMBEDDING_DIM,
                'LSTM_UNITS': config.LSTM_UNITS,
                'DROPOUT_RATE': config.DROPOUT_RATE
            }
        }
        
        with open(os.path.join(config.MODEL_DIR, 'model_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        config.logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath=None):
        """Load a trained model and tokenizers"""
        if filepath is None:
            filepath = os.path.join(config.MODEL_DIR, 'nmt_model.h5')
        
        if not os.path.exists(filepath):
            config.logger.error(f"Model file not found: {filepath}")
            return False
        
        self.model = load_model(filepath)
        
        # Load tokenizers
        with open(os.path.join(config.MODEL_DIR, 'eng_tokenizer.pkl'), 'rb') as f:
            self.eng_tokenizer = pickle.load(f)
        
        with open(os.path.join(config.MODEL_DIR, 'fr_tokenizer.pkl'), 'rb') as f:
            self.fr_tokenizer = pickle.load(f)
        
        # Load metadata
        with open(os.path.join(config.MODEL_DIR, 'model_metadata.json'), 'r') as f:
            metadata = json.load(f)
        
        self.max_eng_len = metadata['max_eng_len']
        self.max_fr_len = metadata['max_fr_len']
        self.eng_vocab_size = metadata['eng_vocab_size']
        self.fr_vocab_size = metadata['fr_vocab_size']
        
        config.logger.info(f"Model loaded from {filepath}")
        return True
    
    def translate(self, sentence, max_length=None):
        """Translate a sentence from English to French"""
        if self.encoder_model is None or self.decoder_model is None:
            config.logger.error("Inference models not built. Call build_inference_models() first.")
            return None
        
        if max_length is None:
            max_length = self.max_fr_len
        
        # Preprocess input
        seq = self.eng_tokenizer.texts_to_sequences([sentence.lower()])
        seq = pad_sequences(seq, maxlen=self.max_eng_len, padding='post')
        
        # Encode input
        encoder_outputs, states_value = self.encoder_model.predict(seq, verbose=0)
        
        # Initialize decoder
        target_seq = np.array([[self.fr_tokenizer.word_index['<sos>']]])
        decoded = []
        
        # Decode
        for _ in range(max_length):
            output_tokens, h, c = self.decoder_model.predict(
                [target_seq, encoder_outputs] + states_value, verbose=0
            )
            next_idx = np.argmax(output_tokens[0, -1, :])
            next_word = self.fr_tokenizer.index_word.get(next_idx, '')
            
            if next_word == '<eos>' or next_word == '':
                break
            
            decoded.append(next_word)
            target_seq = np.array([[next_idx]])
            states_value = [h, c]
        
        return ' '.join(decoded)
    
    def evaluate_model(self, test_pairs):
        """Evaluate the model on test data"""
        config.logger.info("Evaluating model...")
        
        correct = 0
        total = len(test_pairs)
        results = []
        
        for eng, fr in test_pairs:
            predicted = self.translate(eng)
            actual = fr.lower()
            
            # Simple word-level accuracy
            pred_words = set(predicted.split())
            actual_words = set(actual.split())
            
            if pred_words == actual_words:
                correct += 1
            
            results.append({
                'input': eng,
                'actual': actual,
                'predicted': predicted,
                'correct': pred_words == actual_words
            })
        
        accuracy = correct / total
        config.logger.info(f"Model accuracy: {accuracy:.2%}")
        
        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'results': results
        }

# Model Evaluation Metrics
def calculate_bleu_score(references, predictions):
    """Calculate BLEU score for evaluation"""
    try:
        from nltk.translate.bleu_score import sentence_bleu
        scores = []
        for ref, pred in zip(references, predictions):
            ref_tokens = ref.split()
            pred_tokens = pred.split()
            score = sentence_bleu([ref_tokens], pred_tokens)
            scores.append(score)
        return np.mean(scores)
    except ImportError:
        config.logger.warning("NLTK not available for BLEU score calculation")
        return None

# Main execution function
def main():
    """Main function to train and test the NMT model"""
    config.logger.info("Starting Neural Machine Translation training...")
    
    # Initialize NMT model
    nmt = NeuralMachineTranslation(config)
    
    # Check if model already exists
    if nmt.load_model():
        config.logger.info("Loaded existing model")
    else:
        config.logger.info("Training new model...")
        
        # Prepare data
        X, decoder_input, decoder_target = nmt.prepare_data(pairs)
        
        # Build and train model
        nmt.build_model()
        history = nmt.train(X, decoder_input, decoder_target)
        
        # Build inference models
        nmt.build_inference_models()
        
        # Save model
        nmt.save_model()
    
    # Test translations
    test_sentences = [
        "hello",
        "how are you",
        "i love you",
        "thank you",
        "good morning"
    ]
    
    config.logger.info("Testing translations:")
    for sentence in test_sentences:
        translation = nmt.translate(sentence)
        print(f"EN: {sentence}")
        print(f"FR: {translation}")
        print()
    
    # Evaluate model
    evaluation = nmt.evaluate_model(pairs[:5])  # Test on first 5 pairs
    config.logger.info(f"Model evaluation: {evaluation['accuracy']:.2%} accuracy")
    
    return nmt

# Flask Web Interface
from flask import Flask, render_template, request, jsonify
import threading

app = Flask(__name__)
nmt_model = None

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/translate', methods=['POST'])
def translate_api():
    """API endpoint for translation"""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        if nmt_model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        translation = nmt_model.translate(text)
        
        return jsonify({
            'original': text,
            'translation': translation,
            'success': True
        })
    
    except Exception as e:
        config.logger.error(f"Translation error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/add_translation', methods=['POST'])
def add_translation():
    """API endpoint to add new translation pairs"""
    try:
        data = request.get_json()
        source = data.get('source', '').strip()
        target = data.get('target', '').strip()
        
        if not source or not target:
            return jsonify({'error': 'Both source and target required'}), 400
        
        db.add_translation(source, target)
        
        return jsonify({
            'message': 'Translation added successfully',
            'success': True
        })
    
    except Exception as e:
        config.logger.error(f"Add translation error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/stats')
def get_stats():
    """Get database and model statistics"""
    try:
        db_stats = db.get_stats()
        return jsonify({
            'database': db_stats,
            'model_loaded': nmt_model is not None,
            'success': True
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def run_web_app():
    """Run the Flask web application"""
    global nmt_model
    
    # Load the model
    nmt_model = NeuralMachineTranslation(config)
    if not nmt_model.load_model():
        config.logger.error("Failed to load model for web app")
        return
    
    nmt_model.build_inference_models()
    config.logger.info("Model loaded for web application")
    
    # Run Flask app
    app.run(host='0.0.0.0', port=5000, debug=False)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'web':
        # Run web application
        run_web_app()
    else:
        # Run training and testing
        main()

# 🌍 What This Enhanced Project Demonstrates:
# ✅ Modern TensorFlow/Keras implementation with attention mechanism
# ✅ Comprehensive configuration management and logging
# ✅ Mock database system for storing translation pairs
# ✅ Model evaluation metrics and validation
# ✅ Web interface for interactive translation
# ✅ Proper model saving/loading with metadata
# ✅ Enhanced error handling and logging
# ✅ Ready for production deployment