#!/usr/bin/env python3
"""
Test script for Neural Machine Translation project
This script tests the basic functionality of the NMT system
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import importlib.util
spec = importlib.util.spec_from_file_location("nmt_module", "0102.py")
nmt_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nmt_module)

NeuralMachineTranslation = nmt_module.NeuralMachineTranslation
Config = nmt_module.Config
TranslationDatabase = nmt_module.TranslationDatabase

def test_database():
    """Test the mock database functionality"""
    print("🧪 Testing Database System...")
    
    db = TranslationDatabase()
    pairs = db.get_pairs()
    stats = db.get_stats()
    
    print(f"✅ Database loaded with {len(pairs)} translation pairs")
    print(f"✅ Database stats: {stats}")
    
    # Test adding a new translation
    db.add_translation("test phrase", "phrase de test")
    updated_pairs = db.get_pairs()
    
    print(f"✅ Added translation, now have {len(updated_pairs)} pairs")
    return True

def test_model_initialization():
    """Test model initialization"""
    print("\n🧪 Testing Model Initialization...")
    
    config = Config()
    nmt = NeuralMachineTranslation(config)
    
    print("✅ Model class initialized successfully")
    print(f"✅ Configuration loaded: {config.EMBEDDING_DIM}D embeddings, {config.LSTM_UNITS} LSTM units")
    
    return True

def test_data_preparation():
    """Test data preparation"""
    print("\n🧪 Testing Data Preparation...")
    
    config = Config()
    nmt = NeuralMachineTranslation(config)
    
    # Get sample data
    db = TranslationDatabase()
    pairs = db.get_pairs()[:5]  # Use first 5 pairs for testing
    
    try:
        X, decoder_input, decoder_target = nmt.prepare_data(pairs)
        print(f"✅ Data prepared successfully")
        print(f"✅ Input shape: {X.shape}")
        print(f"✅ Decoder input shape: {decoder_input.shape}")
        print(f"✅ Decoder target shape: {decoder_target.shape}")
        print(f"✅ English vocab size: {nmt.eng_vocab_size}")
        print(f"✅ French vocab size: {nmt.fr_vocab_size}")
        return True
    except Exception as e:
        print(f"❌ Data preparation failed: {e}")
        return False

def test_model_building():
    """Test model building"""
    print("\n🧪 Testing Model Building...")
    
    config = Config()
    nmt = NeuralMachineTranslation(config)
    
    # Prepare data first
    db = TranslationDatabase()
    pairs = db.get_pairs()[:5]
    X, decoder_input, decoder_target = nmt.prepare_data(pairs)
    
    try:
        model = nmt.build_model()
        print(f"✅ Model built successfully")
        print(f"✅ Model summary:")
        model.summary()
        return True
    except Exception as e:
        print(f"❌ Model building failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Starting Neural Machine Translation Tests\n")
    
    tests = [
        test_database,
        test_model_initialization,
        test_data_preparation,
        test_model_building
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The system is ready to use.")
        print("\n📝 Next steps:")
        print("1. Run 'python 0102.py' to train the model")
        print("2. Run 'python 0102.py web' to start the web interface")
        print("3. Open http://localhost:5000 in your browser")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
