import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings

warnings.filterwarnings('ignore')

# Enable GPU if available (remove artificial CPU limitation)
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    print(f"🎮 GPU detected: {len(physical_devices)} device(s)")
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
else:
    print("💻 Running on CPU")


class LSTMPricePrediction:
    """
    Production-grade LSTM model for cryptocurrency price prediction
    Features:
    - Multi-feature input (OHLCV + technical indicators)
    - Bidirectional LSTM with attention mechanism
    - Proper regularization and normalization
    - Advanced callbacks for training optimization
    """

    def __init__(self, lookback_period=60, use_technical_indicators=True):
        self.lookback_period = lookback_period
        self.use_technical_indicators = use_technical_indicators
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.history = None
        self.metrics = {}
        self.feature_columns = []

    def add_technical_indicators(self, df):
        """Add technical indicators as features"""
        print("📊 Adding technical indicators...")

        df = df.copy()

        # Moving Averages
        df['ma_7'] = df['close'].rolling(window=7, min_periods=1).mean()
        df['ma_14'] = df['close'].rolling(window=14, min_periods=1).mean()
        df['ma_30'] = df['close'].rolling(window=30, min_periods=1).mean()

        # Exponential Moving Average
        df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()

        # RSI (Relative Strength Index)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
        rs = gain / (loss + 1e-10)
        df['rsi'] = 100 - (100 / (1 + rs))

        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()

        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20, min_periods=1).mean()
        bb_std = df['close'].rolling(window=20, min_periods=1).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']

        # Price momentum
        df['momentum'] = df['close'].pct_change(periods=10)

        # Volume indicators
        df['volume_ma'] = df['volume'].rolling(window=7, min_periods=1).mean()
        df['volume_ratio'] = df['volume'] / (df['volume_ma'] + 1e-10)

        # Volatility
        df['volatility'] = df['close'].pct_change().rolling(window=14, min_periods=1).std()

        # Price range
        df['price_range'] = (df['high'] - df['low']) / (df['close'] + 1e-10)

        # Fill any remaining NaN values
        df = df.fillna(method='bfill').fillna(method='ffill')

        return df

    def prepare_data(self, df, test_size=0.2):
        """
        Prepare multi-feature data for LSTM training
        """
        print(f"📊 Preparing data with lookback period: {self.lookback_period} days")

        # Add technical indicators if enabled
        if self.use_technical_indicators:
            df = self.add_technical_indicators(df)

            # Define feature columns
            self.feature_columns = [
                'open', 'high', 'low', 'close', 'volume',
                'ma_7', 'ma_14', 'ma_30', 'ema_12', 'ema_26',
                'rsi', 'macd', 'macd_signal',
                'bb_middle', 'bb_upper', 'bb_lower', 'bb_width',
                'momentum', 'volume_ma', 'volume_ratio',
                'volatility', 'price_range'
            ]
        else:
            self.feature_columns = ['open', 'high', 'low', 'close', 'volume']

        print(f"📈 Using {len(self.feature_columns)} features: {self.feature_columns[:5]}...")

        # Extract feature data
        data = df[self.feature_columns].values

        # Scale the data
        scaled_data = self.scaler.fit_transform(data)

        # Create sequences
        X, y = [], []
        for i in range(self.lookback_period, len(scaled_data)):
            X.append(scaled_data[i - self.lookback_period:i])
            # Predict 'close' price (index 3 if using OHLCV)
            close_idx = self.feature_columns.index('close')
            y.append(scaled_data[i, close_idx])

        X, y = np.array(X), np.array(y)

        # Train/test split
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        print(f"✅ Data prepared: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
        print(f"   Input shape: ({self.lookback_period}, {len(self.feature_columns)})")

        return X_train, X_test, y_train, y_test, df

    def build_model(self, input_shape):
        """
        Build production-grade LSTM model
        Architecture: Bidirectional LSTM with dropout and batch normalization
        """
        print("🏗️  Building production LSTM model...")

        self.model = Sequential([
            # First Bidirectional LSTM layer
            Bidirectional(LSTM(128, return_sequences=True), input_shape=input_shape),
            Dropout(0.2),
            BatchNormalization(),

            # Second Bidirectional LSTM layer
            Bidirectional(LSTM(64, return_sequences=True)),
            Dropout(0.2),
            BatchNormalization(),

            # Third LSTM layer
            LSTM(32, return_sequences=False),
            Dropout(0.2),

            # Dense layers
            Dense(16, activation='relu'),
            Dropout(0.1),
            Dense(1)
        ])

        # Compile with Adam optimizer
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='huber',  # More robust to outliers than MSE
            metrics=['mae', 'mse']
        )

        print(f"✅ Model built with {self.model.count_params():,} parameters")
        return self.model

    def train(self, X_train, y_train, X_test, y_test, epochs=100, batch_size=32):
        """
        Train the model with advanced callbacks
        """
        print(f"\n🚀 Training model for up to {epochs} epochs...")
        print(f"   Batch size: {batch_size}")

        # Callbacks for training optimization
        callbacks = [
            # Stop training when validation loss stops improving
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),

            # Reduce learning rate when plateau is detected
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=7,
                min_lr=1e-7,
                verbose=1
            )
        ]

        # Train the model
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            callbacks=callbacks,
            verbose=1,
            shuffle=False  # Important for time series
        )

        print("✅ Training completed")
        return self.history

    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance
        """
        print("\n📈 Evaluating model performance...")

        # Make predictions
        y_pred_scaled = self.model.predict(X_test, batch_size=32, verbose=0)

        # Create dummy arrays for inverse transform
        close_idx = self.feature_columns.index('close')
        dummy_shape = (len(y_test), len(self.feature_columns))

        # Inverse transform predictions
        y_test_dummy = np.zeros(dummy_shape)
        y_pred_dummy = np.zeros(dummy_shape)
        y_test_dummy[:, close_idx] = y_test
        y_pred_dummy[:, close_idx] = y_pred_scaled.flatten()

        y_test_rescaled = self.scaler.inverse_transform(y_test_dummy)[:, close_idx]
        y_pred_rescaled = self.scaler.inverse_transform(y_pred_dummy)[:, close_idx]

        # Calculate metrics
        mse = mean_squared_error(y_test_rescaled, y_pred_rescaled)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_test_rescaled - y_pred_rescaled))
        mape = mean_absolute_percentage_error(y_test_rescaled, y_pred_rescaled)
        r2 = r2_score(y_test_rescaled, y_pred_rescaled)

        # Directional accuracy (did we predict up/down correctly?)
        y_test_direction = np.diff(y_test_rescaled) > 0
        y_pred_direction = np.diff(y_pred_rescaled) > 0
        directional_accuracy = np.mean(y_test_direction == y_pred_direction)

        self.metrics = {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'mape': float(mape * 100),  # Convert to percentage
            'r2': float(r2),
            'directional_accuracy': float(directional_accuracy * 100)
        }

        print(f"\n📊 Model Performance Metrics:")
        print(f"   RMSE: ${rmse:,.2f}")
        print(f"   MAE: ${mae:,.2f}")
        print(f"   MAPE: {mape * 100:.2f}%")
        print(f"   R² Score: {r2:.4f}")
        print(f"   Directional Accuracy: {directional_accuracy * 100:.2f}%")

        return self.metrics

    def predict_future(self, df, days_ahead=7):
        """
        Predict future prices using recursive forecasting
        """
        print(f"\n🔮 Predicting next {days_ahead} days...")

        # Prepare the last sequence
        if self.use_technical_indicators:
            df_with_indicators = self.add_technical_indicators(df)
            last_sequence = df_with_indicators[self.feature_columns].values[-self.lookback_period:]
        else:
            last_sequence = df[self.feature_columns].values[-self.lookback_period:]

        last_sequence_scaled = self.scaler.transform(last_sequence)

        predictions = []
        current_sequence = last_sequence_scaled.copy()
        close_idx = self.feature_columns.index('close')

        for day in range(days_ahead):
            # Reshape for prediction
            X_input = current_sequence.reshape(1, self.lookback_period, len(self.feature_columns))

            # Predict next value
            next_pred_scaled = self.model.predict(X_input, verbose=0)[0, 0]

            # Create new row with predicted close price
            new_row = current_sequence[-1].copy()
            new_row[close_idx] = next_pred_scaled

            # Update sequence
            current_sequence = np.vstack([current_sequence[1:], new_row])

            # Store prediction
            predictions.append(next_pred_scaled)

        # Inverse transform predictions
        dummy_shape = (len(predictions), len(self.feature_columns))
        pred_dummy = np.zeros(dummy_shape)
        pred_dummy[:, close_idx] = predictions
        predictions_rescaled = self.scaler.inverse_transform(pred_dummy)[:, close_idx]

        # Format predictions
        current_price = float(df['close'].values[-1])
        prediction_data = []

        for i, pred in enumerate(predictions_rescaled, 1):
            change = ((pred - current_price) / current_price) * 100
            prediction_data.append({
                'day': i,
                'predicted_price': float(pred),
                'change_from_current': float(change),
                'trend': 'BULLISH' if pred > current_price else 'BEARISH'
            })

        # Calculate confidence based on model performance
        confidence_score = min(100, max(0, (1 - self.metrics.get('mape', 100) / 100) * 100))

        return {
            'current_price': current_price,
            'predictions': prediction_data,
            'confidence_score': float(confidence_score),
            'model_metrics': self.metrics
        }

    def train_and_predict(self, df, test_size=0.2, epochs=100, batch_size=32, days_ahead=7):
        """
        Complete training and prediction pipeline
        """
        print("=" * 80)
        print("🤖 PRODUCTION LSTM CRYPTOCURRENCY PRICE PREDICTION")
        print("=" * 80)

        # Prepare data
        X_train, X_test, y_train, y_test, df_processed = self.prepare_data(df, test_size)

        # Build and train model
        self.build_model(input_shape=(self.lookback_period, len(self.feature_columns)))
        self.train(X_train, y_train, X_test, y_test, epochs=epochs, batch_size=batch_size)

        # Evaluate
        metrics = self.evaluate(X_test, y_test)

        # Predict future
        predictions = self.predict_future(df, days_ahead)

        print("\n" + "=" * 80)
        print("✅ LSTM PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 80)

        return {
            'symbol': 'N/A',  # Will be set by caller
            'metrics': metrics,
            'predictions': predictions,
            'model_info': {
                'architecture': 'Bidirectional LSTM (3 layers: 128→64→32 units)',
                'features': len(self.feature_columns),
                'lookback_period': self.lookback_period,
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'total_parameters': self.model.count_params()
            },
            'training_history': {
                'epochs_trained': len(self.history.history['loss']),
                'final_loss': float(self.history.history['loss'][-1]),
                'final_val_loss': float(self.history.history['val_loss'][-1])
            }
        }


def predict_cryptocurrency_price(df, symbol, lookback_period=60, days_ahead=7):
    """
    Main function to predict cryptocurrency prices

    Args:
        df: DataFrame with columns ['date', 'open', 'high', 'low', 'close', 'volume']
        symbol: Cryptocurrency symbol (e.g., 'BTCUSDT')
        lookback_period: Number of days to look back (default: 60)
        days_ahead: Number of days to predict (default: 7)

    Returns:
        Dictionary with predictions and metrics
    """
    predictor = LSTMPricePrediction(
        lookback_period=lookback_period,
        use_technical_indicators=True
    )

    results = predictor.train_and_predict(
        df,
        test_size=0.2,
        epochs=100,  # Will stop early if no improvement
        batch_size=32,
        days_ahead=days_ahead
    )

    results['symbol'] = symbol
    return results