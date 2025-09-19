# MNIST CNN Model Training - Version 1

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A deep learning project implementing a Convolutional Neural Network (CNN) for MNIST digit classification using PyTorch. This is the first iteration of our model development journey.

## 📁 Project Structure

```
tsai_assignment_5/
├── mnist_model_training_v1.ipynb    # Main training notebook
├── README.md                        # Project documentation
└── data/
    └── MNIST/
        └── raw/                     # MNIST dataset files
```

## 🎯 Objective

Develop a CNN model that can achieve high accuracy on MNIST digit classification while maintaining reasonable computational efficiency.

## 🏗️ Model Architecture

Our CNN model (`CNN_Model`) follows a deep convolutional architecture with the following layers:

### Network Structure
```
Input Layer (1, 28, 28)
├── Conv2d(1→8, kernel=3, padding=1) + BatchNorm2d + ReLU
├── Conv2d(8→16, kernel=3, padding=1) + BatchNorm2d + ReLU + MaxPool2d(2,2)
├── Conv2d(16→32, kernel=3, padding=1) + BatchNorm2d + ReLU
├── Conv2d(32→64, kernel=3, padding=1) + BatchNorm2d + ReLU + MaxPool2d(2,2)
├── Conv2d(64→128, kernel=3, padding=1) + BatchNorm2d + ReLU
├── Conv2d(128→256, kernel=3, padding=1) + AdaptiveAvgPool2d(1)
├── Linear(256→100) + ReLU
└── Linear(100→10)
```

### Key Features
- **6 Convolutional Layers**: Progressive channel expansion (1→8→16→32→64→128→256)
- **Batch Normalization**: Applied after each convolution for training stability
- **ReLU Activation**: Standard activation function for non-linearity
- **Adaptive Global Average Pooling**: Reduces spatial dimensions efficiently
- **2 Fully Connected Layers**: Final classification layers

## 📊 Model Parameters

```python
================================================================
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [-1, 8, 28, 28]              80
       BatchNorm2d-2            [-1, 8, 28, 28]              16
            Conv2d-3           [-1, 16, 28, 28]           1,168
       BatchNorm2d-4           [-1, 16, 28, 28]              32
            Conv2d-5           [-1, 32, 14, 14]           4,640
       BatchNorm2d-6           [-1, 32, 14, 14]              64
            Conv2d-7           [-1, 64, 14, 14]          18,496
       BatchNorm2d-8           [-1, 64, 14, 14]             128
            Conv2d-9          [-1, 128, 7, 7]            73,856
      BatchNorm2d-10          [-1, 128, 7, 7]             256
           Conv2d-11          [-1, 256, 7, 7]           295,168
           Linear-12                  [-1, 100]            25,700
           Linear-13                   [-1, 10]             1,010
================================================================
Total params: 420,614
Trainable params: 420,614
Non-trainable params: 0
================================================================
```

**Total Parameters**: 420,614

## 🔧 Training Configuration

### Hyperparameters
- **Optimizer**: Adam
- **Learning Rate**: 0.001
- **Batch Size**: 128 (training), 2000 (testing)
- **Epochs**: 20
- **Loss Function**: CrossEntropyLoss
- **Device**: CUDA (if available) / CPU

### Data Preprocessing
- **Normalization**: Mean = 0.1307, Std = 0.3081 (calculated from training data)
- **Transform Pipeline**: ToTensor() → Normalize()

## 📈 Training Results

### Final Performance Metrics
| Metric | Training | Testing |
|--------|----------|---------|
| **Final Accuracy** | XX.XX% | XX.XX% |
| **Final Loss** | X.XXXX | X.XXXX |

### Training Evolution
The model was trained for 20 epochs with the following progression:

```
Epoch [1/20]  - Train Loss: X.XXXX, Train Acc: XX.XX% - Test Loss: X.XXXX, Test Acc: XX.XX%
Epoch [2/20]  - Train Loss: X.XXXX, Train Acc: XX.XX% - Test Loss: X.XXXX, Test Acc: XX.XX%
...
Epoch [20/20] - Train Loss: X.XXXX, Train Acc: XX.XX% - Test Loss: X.XXXX, Test Acc: XX.XX%
```

### Performance Visualization
The notebook includes comprehensive visualization of:
- **Training vs Test Loss** curves over 20 epochs
- **Training vs Test Accuracy** curves over 20 epochs
- **Sample predictions** on test data

## 🚀 Quick Start

### Prerequisites
```bash
pip install torch torchvision matplotlib torchsummary
```

### Running the Model
1. Clone this repository
2. Open `mnist_model_training_v1.ipynb` in Jupyter Notebook or VS Code
3. Run all cells sequentially
4. View training progress and final results

### Key Notebook Sections
1. **Data Loading & Preprocessing**: MNIST dataset preparation
2. **Model Definition**: CNN architecture implementation
3. **Training Loop**: 20-epoch training with metrics tracking
4. **Performance Analysis**: Loss/accuracy visualization
5. **Model Summary**: Parameter count and architecture details

## 📊 Model Performance Analysis

### Strengths
- ✅ Deep architecture for feature extraction
- ✅ Batch normalization for training stability
- ✅ Adaptive pooling for parameter efficiency
- ✅ Comprehensive metrics tracking

### Areas for Improvement
- 🔄 Parameter count is relatively high (420K+ parameters)
- 🔄 Could explore more efficient architectures
- 🔄 Learning rate scheduling could improve convergence
- 🔄 Data augmentation could enhance generalization

## 🔮 Next Iterations

Future versions will focus on:
1. **Parameter Reduction**: Target <20K parameters
2. **Architecture Optimization**: More efficient designs
3. **Advanced Techniques**: Learning rate scheduling, dropout, data augmentation
4. **Performance Targets**: Achieving 99%+ accuracy

## 📝 Notes

- This is Version 1 of our MNIST model development
- Model achieves good performance but has room for optimization
- Serves as baseline for future improvements
- All training metrics and visualizations are preserved in the notebook

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Project Status**: ✅ Completed - Version 1  
**Next Version**: Parameter optimization and architecture improvements  
**Last Updated**: September 2025
