#

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow_federated as tff
import collections

# Check TensorFlow and TFF versions
print(f"TensorFlow version: {tf.__version__}")
print(f"TensorFlow Federated version: {tff.__version__}")

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Load and preprocess the dataset
def load_and_preprocess_data(file_path):
    try:
        # Load dataset
        df = pd.read_csv(file_path)
        
        # Separate features and labels
        X = df.drop('Label', axis=1)
        y = df['Label']
        
        # Encode labels
        le = LabelEncoder()
        y = le.fit_transform(y)
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        return X_train, X_test, y_train, y_test, le, scaler, X.shape[1], len(np.unique(y))
    except FileNotFoundError:
        print(f"Error: File {file_path} not found. Please ensure the dataset is uploaded.")
        raise
    except Exception as e:
        print(f"Error in data preprocessing: {str(e)}")
        raise

# ANN Model
def build_ann_model(input_dim, num_classes):
    model = models.Sequential([
        layers.Dense(128, activation='relu', input_dim=input_dim),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
    return model

# CNN Model (1D for tabular data)
def build_cnn_model(input_dim, num_classes):
    model = models.Sequential([
        layers.Input(shape=(input_dim, 1)),
        layers.Conv1D(64, kernel_size=3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Conv1D(32, kernel_size=3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
    return model

# Federated Continual Learning Model
def build_fcl_keras_model(input_dim, num_classes):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

def build_fcl_model(input_dim, num_classes):
    def model_fn():
        model = build_fcl_keras_model(input_dim, num_classes)
        return tff.learning.models.from_keras_model(
            model,
            input_spec=collections.OrderedDict(
                x=tf.TensorSpec(shape=[None, input_dim], dtype=tf.float32),
                y=tf.TensorSpec(shape=[None], dtype=tf.int32)
            ),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
        )
    return model_fn

# Create federated dataset
def create_federated_dataset(X_train, y_train, num_clients=3):
    client_data = []
    client_size = len(X_train) // num_clients
    for i in range(num_clients):
        start_idx = i * client_size
        end_idx = (i + 1) * client_size if i < num_clients - 1 else len(X_train)
        client_x = X_train[start_idx:end_idx]
        client_y = y_train[start_idx:end_idx]
        client_data.append(
            tf.data.Dataset.from_tensor_slices(
                (client_x, client_y)
            ).batch(32)
        )
    return client_data

# Train and evaluate models
def train_and_evaluate(file_path):
    # Load and preprocess data
    X_train, X_test, y_train, y_test, le, scaler, input_dim, num_classes = load_and_preprocess_data(file_path)
    
    # Reshape for CNN
    X_train_cnn = X_train.reshape(-1, input_dim, 1)
    X_test_cnn = X_test.reshape(-1, input_dim, 1)
    
    # Define EarlyStopping
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    
    # Train ANN
    ann_model = build_ann_model(input_dim, num_classes)
    ann_history = ann_model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )
    ann_test_loss, ann_test_acc = ann_model.evaluate(X_test, y_test, verbose=0)
    
    # Train CNN
    cnn_model = build_cnn_model(input_dim, num_classes)
    cnn_history = cnn_model.fit(
        X_train_cnn, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )
    cnn_test_loss, cnn_test_acc = cnn_model.evaluate(X_test_cnn, y_test, verbose=0)
    
    # Train FCL
    client_datasets = create_federated_dataset(X_train, y_train)
    iterative_process = tff.learning.algorithms.build_fed_avg(
        model_fn=build_fcl_model(input_dim, num_classes),
        client_optimizer_fn=lambda: tf.keras.optimizers.Adam(0.001),
        server_optimizer_fn=lambda: tf.keras.optimizers.Adam(0.01)
    )
    
    state = iterative_process.initialize()
    fcl_accuracies = []
    fcl_val_accuracies = []
    
    # Create validation dataset for FCL
    val_size = int(0.2 * len(X_train))
    X_val, y_val = X_train[-val_size:], y_train[-val_size:]
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(32)
    
    # FCL Keras model for evaluation
    fcl_keras_model = build_fcl_keras_model(input_dim, num_classes)
    fcl_keras_model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    for round_num in range(50):
        state, metrics = iterative_process.next(state, client_datasets)
        train_acc = metrics['client_work']['train']['sparse_categorical_accuracy']
        fcl_accuracies.append(train_acc)
        
        # Evaluate on validation set
        fcl_weights = tff.learning.models.ModelWeights.from_tff_result(state.model)
        fcl_keras_model.set_weights(fcl_weights.trainable)
        val_loss, val_acc = fcl_keras_model.evaluate(val_dataset, verbose=0)
        fcl_val_accuracies.append(val_acc)
        
        print(f'Round {round_num+1}, Train Accuracy: {train_acc:.4f}, Val Accuracy: {val_acc:.4f}')
        
        # Early stopping for FCL
        if round_num > 5 and fcl_val_accuracies[-1] <= fcl_val_accuracies[-2]:
            patience_counter = patience_counter + 1 if 'patience_counter' in locals() else 1
            if patience_counter >= 5:
                print("Early stopping triggered for FCL")
                break
        else:
            patience_counter = 0
    
    # Evaluate FCL on test set
    fcl_weights = tff.learning.models.ModelWeights.from_tff_result(state.model)
    fcl_keras_model.set_weights(fcl_weights.trainable)
    fcl_test_loss, fcl_test_acc = fcl_keras_model.evaluate(X_test, y_test, verbose=0)
    
    # Plot accuracies
    plt.figure(figsize=(12, 6))
    
    # Plot training accuracies
    plt.plot(ann_history.history['accuracy'], label='ANN Train', linestyle='--')
    plt.plot(cnn_history.history['accuracy'], label='CNN Train', linestyle='--')
    plt.plot(fcl_accuracies, label='FCL Train', linestyle='--')
    
    # Plot validation accuracies
    plt.plot(ann_history.history['val_accuracy'], label='ANN Validation')
    plt.plot(cnn_history.history['val_accuracy'], label='CNN Validation')
    plt.plot(fcl_val_accuracies, label='FCL Validation')
    
    # Plot test accuracies
    plt.axhline(y=ann_test_acc, color='r', linestyle=':', label=f'ANN Test ({ann_test_acc:.4f})')
    plt.axhline(y=cnn_test_acc, color='g', linestyle=':', label=f'CNN Test ({cnn_test_acc:.4f})')
    plt.axhline(y=fcl_test_acc, color='b', linestyle=':', label=f'FCL Test ({fcl_test_acc:.4f})')
    
    plt.title('Model Accuracy Comparison')
    plt.xlabel('Epoch / Round')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)  # Set y-axis range to 0-1
    plt.legend()
    plt.grid(True)
    plt.savefig('accuracy_comparison.png')
    plt.show()
    
    # Print final test accuracies
    print(f"\nANN Test Accuracy: {ann_test_acc:.4f}")
    print(f"CNN Test Accuracy: {cnn_test_acc:.4f}")
    print(f"FCL Test Accuracy: {fcl_test_acc:.4f}")
    
    # Classification report for FCL
    fcl_predictions = np.argmax(fcl_keras_model.predict(X_test), axis=1)
    print("\nFCL Classification Report:")
    print(classification_report(y_test, fcl_predictions, target_names=le.classes_))
    
    return ann_model, cnn_model, fcl_keras_model

# Main execution
if __name__ == "__main__":
    ann_model, cnn_model, fcl_model = train_and_evaluate("/content/drive/MyDrive/ALLFLOWMETER_HIKARI2021 - Copy.csv")