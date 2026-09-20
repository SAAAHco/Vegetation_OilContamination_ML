"""
Custom CNN Architecture for Oil Contamination Detection and Vegetation Health Prediction

This module implements the encoder-decoder CNN with skip connections as described in the manuscript.
The architecture is specifically designed for multispectral satellite imagery analysis in arid
oil-contaminated environments.

Architecture:
    Encoder: 5 convolutional blocks (32→64→128→256→512 filters)
    Decoder: Transposed convolutions with skip connections
    Output: 1×1 convolution for pixel-wise classification/regression

References:
    - Manuscript Section 2.3 (Methods - CNN Architecture)
    - Supplementary Material Section S1.4 (Deep Learning Model Architecture)
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, callbacks
from tensorflow.keras.optimizers import Adam
from typing import Tuple, Optional, List


def conv_block(
    x: tf.Tensor,
    filters: int,
    kernel_size: Tuple[int, int] = (3, 3),
    use_batch_norm: bool = True,
    l2_reg: float = 1e-4,
    dropout_rate: float = 0.0,
    name_prefix: str = ""
) -> tf.Tensor:
    """
    Convolutional block with two 3×3 convolutions, batch normalization, and ReLU.
    
    As described in Supplementary Material Section S1.4:
    "Each containing two 3×3 convolutional layers followed by batch normalization 
    and ReLU activation"
    
    Args:
        x: Input tensor
        filters: Number of filters
        kernel_size: Kernel size (default 3×3)
        use_batch_norm: Whether to use batch normalization
        l2_reg: L2 regularization weight (λ=1×10⁻⁴ as per manuscript)
        dropout_rate: Dropout rate (0.3 for fully connected layers)
        name_prefix: Prefix for layer names
        
    Returns:
        Output tensor after convolution block
    """
    # First convolution
    x = layers.Conv2D(
        filters,
        kernel_size,
        padding='same',
        kernel_regularizer=regularizers.l2(l2_reg),
        name=f"{name_prefix}_conv1"
    )(x)
    if use_batch_norm:
        x = layers.BatchNormalization(name=f"{name_prefix}_bn1")(x)
    x = layers.Activation('relu', name=f"{name_prefix}_relu1")(x)
    
    # Second convolution
    x = layers.Conv2D(
        filters,
        kernel_size,
        padding='same',
        kernel_regularizer=regularizers.l2(l2_reg),
        name=f"{name_prefix}_conv2"
    )(x)
    if use_batch_norm:
        x = layers.BatchNormalization(name=f"{name_prefix}_bn2")(x)
    x = layers.Activation('relu', name=f"{name_prefix}_relu2")(x)
    
    if dropout_rate > 0:
        x = layers.Dropout(dropout_rate, name=f"{name_prefix}_dropout")(x)
    
    return x


def build_cnn_encoder_decoder(
    input_shape: Tuple[int, int, int] = (256, 256, 6),
    num_classes: int = 1,
    task: str = 'classification',
    l2_reg: float = 1e-4,
    dropout_rate: float = 0.3,
    use_batch_norm: bool = True,
    filters: List[int] = [32, 64, 128, 256, 512]
) -> models.Model:
    """
    Build the custom CNN encoder-decoder architecture with skip connections.
    
    Architecture as described in manuscript Section 2.3:
    "The encoder pathway consisted of five convolutional blocks (32→64→128→256→512 filters) 
    with 3×3 kernels, batch normalization, and ReLU activation, followed by 2×2 max-pooling; 
    the decoder pathway used 2×2 transposed convolutions with corresponding skip connections 
    to preserve spatial detail"
    
    Args:
        input_shape: Input tensor shape (height, width, channels)
                    Default (256, 256, 6) for 6-band multispectral imagery
        num_classes: Number of output classes/channels
                    - 1 for binary classification (contaminated/non-contaminated)
                    - 2 for SAVI/ARVI regression
        task: 'classification' or 'regression'
        l2_reg: L2 regularization weight (λ=1×10⁻⁴)
        dropout_rate: Dropout rate (0.3 for fully connected layers)
        use_batch_norm: Whether to use batch normalization
        filters: List of filter counts for each encoder block
        
    Returns:
        Keras Model instance
    """
    inputs = layers.Input(shape=input_shape, name='input')
    
    # Store skip connections
    skip_connections = []
    
    # ==================== ENCODER ====================
    # "Feature extraction pathway consists of five convolutional blocks"
    
    x = inputs
    for i, f in enumerate(filters):
        x = conv_block(
            x, f,
            use_batch_norm=use_batch_norm,
            l2_reg=l2_reg,
            name_prefix=f"encoder_block{i+1}"
        )
        
        # Store for skip connection (except last block)
        if i < len(filters) - 1:
            skip_connections.append(x)
            # 2×2 max-pooling after each block
            x = layers.MaxPooling2D(
                pool_size=(2, 2),
                name=f"encoder_pool{i+1}"
            )(x)
    
    # Bottleneck (last encoder block serves as bottleneck)
    
    # ==================== DECODER ====================
    # "Decoder pathway used 2×2 transposed convolutions with corresponding skip connections"
    
    decoder_filters = filters[-2::-1]  # Reverse, excluding last (bottleneck)
    
    for i, f in enumerate(decoder_filters):
        # 2×2 transposed convolution (upsampling)
        x = layers.Conv2DTranspose(
            f,
            kernel_size=(2, 2),
            strides=(2, 2),
            padding='same',
            kernel_regularizer=regularizers.l2(l2_reg),
            name=f"decoder_upconv{i+1}"
        )(x)
        
        # Skip connection - concatenate with corresponding encoder feature map
        skip_idx = len(skip_connections) - 1 - i
        x = layers.Concatenate(name=f"decoder_concat{i+1}")([x, skip_connections[skip_idx]])
        
        # Convolutional block
        x = conv_block(
            x, f,
            use_batch_norm=use_batch_norm,
            l2_reg=l2_reg,
            name_prefix=f"decoder_block{i+1}"
        )
    
    # ==================== OUTPUT ====================
    # "Output layer employed 1×1 convolution for pixel-wise classification"
    
    if task == 'classification':
        # Binary classification: sigmoid activation
        # "Sigmoid activation with binary cross-entropy loss"
        outputs = layers.Conv2D(
            num_classes,
            kernel_size=(1, 1),
            activation='sigmoid',
            name='output_classification'
        )(x)
    else:
        # Regression: linear activation
        # "For predicting continuous ARVI and SAVI values, the output layer consisted 
        # of two neurons with a linear activation function"
        outputs = layers.Conv2D(
            num_classes,
            kernel_size=(1, 1),
            activation='linear',
            name='output_regression'
        )(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name='CNN_EncoderDecoder')
    
    return model


def build_classification_model(
    input_shape: Tuple[int, int, int] = (256, 256, 6),
    learning_rate: float = 0.001
) -> models.Model:
    """
    Build and compile the CNN for binary classification task.
    
    Classifies pixels as contaminated or non-contaminated.
    
    Args:
        input_shape: Input tensor shape
        learning_rate: Initial learning rate (0.001 as per manuscript)
        
    Returns:
        Compiled Keras Model
    """
    model = build_cnn_encoder_decoder(
        input_shape=input_shape,
        num_classes=1,
        task='classification'
    )
    
    # Adam optimizer with β₁=0.9, β₂=0.999
    optimizer = Adam(
        learning_rate=learning_rate,
        beta_1=0.9,
        beta_2=0.999,
        clipnorm=1.0  # Gradient clipping (norm=1.0)
    )
    
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), 
                 tf.keras.metrics.Recall(), tf.keras.metrics.AUC()]
    )
    
    return model


def build_regression_model(
    input_shape: Tuple[int, int, int] = (256, 256, 6),
    num_outputs: int = 2,
    learning_rate: float = 0.001
) -> models.Model:
    """
    Build and compile the CNN for regression task (SAVI/ARVI prediction).
    
    Predicts continuous vegetation index values for each pixel.
    
    Args:
        input_shape: Input tensor shape
        num_outputs: Number of indices to predict (2 for SAVI and ARVI)
        learning_rate: Initial learning rate
        
    Returns:
        Compiled Keras Model
    """
    model = build_cnn_encoder_decoder(
        input_shape=input_shape,
        num_classes=num_outputs,
        task='regression'
    )
    
    optimizer = Adam(
        learning_rate=learning_rate,
        beta_1=0.9,
        beta_2=0.999,
        clipnorm=1.0
    )
    
    # Mean squared error for regression
    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae', tf.keras.metrics.RootMeanSquaredError()]
    )
    
    return model


def get_training_callbacks(
    model_path: str = 'best_model.h5',
    patience: int = 15,
    min_delta: float = 0.001
) -> List[callbacks.Callback]:
    """
    Get training callbacks as described in manuscript.
    
    "Training proceeded for 100 epochs with early stopping based on validation 
    performance (patience=15 epochs)"
    
    Args:
        model_path: Path to save best model
        patience: Early stopping patience (15 epochs)
        min_delta: Minimum improvement for early stopping
        
    Returns:
        List of Keras callbacks
    """
    callback_list = [
        # Early stopping
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            min_delta=min_delta,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Model checkpoint
        callbacks.ModelCheckpoint(
            filepath=model_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        
        # Learning rate scheduler - cosine annealing
        callbacks.LearningRateScheduler(
            cosine_annealing_schedule,
            verbose=0
        ),
        
        # TensorBoard logging
        callbacks.TensorBoard(
            log_dir='./logs',
            histogram_freq=1
        ),
        
        # Reduce LR on plateau as backup
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    return callback_list


def cosine_annealing_schedule(epoch: int, lr: float) -> float:
    """
    Cosine annealing learning rate schedule.
    
    "Adam optimizer with initial learning rate of 0.001 and a cosine decay schedule"
    
    Args:
        epoch: Current epoch
        lr: Current learning rate
        
    Returns:
        Updated learning rate
    """
    initial_lr = 0.001
    min_lr = 1e-6
    total_epochs = 100
    
    # Cosine annealing formula
    new_lr = min_lr + 0.5 * (initial_lr - min_lr) * (1 + np.cos(np.pi * epoch / total_epochs))
    
    return new_lr


class F1Score(tf.keras.metrics.Metric):
    """Custom F1 Score metric for model evaluation."""
    
    def __init__(self, name='f1_score', **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.precision = tf.keras.metrics.Precision()
        self.recall = tf.keras.metrics.Recall()
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)
    
    def result(self):
        p = self.precision.result()
        r = self.recall.result()
        return 2 * ((p * r) / (p + r + tf.keras.backend.epsilon()))
    
    def reset_state(self):
        self.precision.reset_state()
        self.recall.reset_state()


def train_model_with_cross_validation(
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
    epochs: int = 100,
    batch_size: int = 32,
    task: str = 'classification'
) -> dict:
    """
    Train model with k-fold cross-validation.
    
    "Model generalization was rigorously validated through 5-fold stratified 
    cross-validation, which demonstrated consistent performance across all folds 
    (mean accuracy=89.3%, SD=1.8%)"
    
    Args:
        X: Input data
        y: Labels/targets
        n_folds: Number of folds (5)
        epochs: Maximum epochs (100)
        batch_size: Batch size (32)
        task: 'classification' or 'regression'
        
    Returns:
        Dictionary with cross-validation results
    """
    from sklearn.model_selection import StratifiedKFold, KFold
    
    if task == 'classification':
        kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        # Flatten y for stratification
        y_flat = y.reshape(-1) if y.ndim > 1 else y
        splits = kfold.split(X, (y_flat > 0.5).astype(int))
    else:
        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        splits = kfold.split(X)
    
    results = {
        'fold_accuracies': [],
        'fold_f1_scores': [],
        'fold_losses': [],
        'histories': []
    }
    
    for fold, (train_idx, val_idx) in enumerate(splits):
        print(f"\n{'='*50}")
        print(f"Training Fold {fold + 1}/{n_folds}")
        print(f"{'='*50}")
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Build fresh model for each fold
        if task == 'classification':
            model = build_classification_model(input_shape=X.shape[1:])
        else:
            model = build_regression_model(input_shape=X.shape[1:])
        
        # Train
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=get_training_callbacks(
                model_path=f'model_fold{fold+1}.h5',
                patience=15
            ),
            verbose=1
        )
        
        # Evaluate
        eval_results = model.evaluate(X_val, y_val, verbose=0)
        
        results['fold_accuracies'].append(eval_results[1])  # accuracy
        results['fold_losses'].append(eval_results[0])  # loss
        results['histories'].append(history.history)
        
        # Clean up
        tf.keras.backend.clear_session()
    
    # Summary statistics
    results['mean_accuracy'] = np.mean(results['fold_accuracies'])
    results['std_accuracy'] = np.std(results['fold_accuracies'])
    results['mean_loss'] = np.mean(results['fold_losses'])
    
    print(f"\n{'='*50}")
    print(f"Cross-Validation Results:")
    print(f"Mean Accuracy: {results['mean_accuracy']*100:.1f}% ± {results['std_accuracy']*100:.1f}%")
    print(f"{'='*50}")
    
    return results


if __name__ == "__main__":
    # Example usage
    print("Building CNN Encoder-Decoder Model...")
    
    # Build classification model
    model = build_classification_model(input_shape=(256, 256, 6))
    model.summary()
    
    # Print model parameters
    trainable_params = np.sum([np.prod(v.shape) for v in model.trainable_weights])
    print(f"\nTotal trainable parameters: {trainable_params:,}")
