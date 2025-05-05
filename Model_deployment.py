# encoding=utf8
import os
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk

import matplotlib
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv1D, Bidirectional, LSTM, Dense, Flatten, concatenate, Softmax, multiply
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback
import threading


def validate_time(time_str):
    try:
        pd.to_datetime(time_str, format="%H:%M:%S")
        return True
    except ValueError:
        return False


class SolarPredictionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Solar Radiation Prediction System")
        self.root.geometry("900x700")  # Set initial window size

        self.train_history = {'loss': [], 'val_loss': [], 'mae': [], 'val_mae': [], 'mape': [], 'val_mape': []}
        self.model = None
        self.step_entry = None

        # Load data early to have it ready
        df = pd.read_csv("SolarPrediction.csv")
        self.original_df = df
        self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test = self.preprocess_data(df)
        self.input_shape = (self.X_train.shape[1], self.X_train.shape[2])
        self.load_saved_model()

        # Create welcome screen
        self.create_welcome_screen()

    def create_welcome_screen(self):
        """Create the welcome screen with solar radiation image"""
        # Main welcome frame
        self.welcome_frame = ttk.Frame(self.root, padding="20")
        self.welcome_frame.pack(fill=tk.BOTH, expand=True)

        # Title
        title_label = ttk.Label(
            self.welcome_frame,
            text="Solar Radiation Prediction System",
            font=("Arial", 24, "bold")
        )
        title_label.pack(pady=20)

        # Solar radiation image
        try:
            image_path = "solar_radiation.png"
            if os.path.exists(image_path):
                img = Image.open(image_path)
                # Resize image to fit the window
                img = img.resize((500, 300), Image.LANCZOS)
                self.photo = ImageTk.PhotoImage(img)
                img_label = ttk.Label(self.welcome_frame, image=self.photo)
                img_label.pack(pady=20)
            else:
                # If image doesn't exist, show placeholder text
                placeholder = ttk.Label(
                    self.welcome_frame,
                    text="[Solar Radiation Image Placeholder]",
                    font=("Arial", 16),
                    background="#f0f0f0",
                    padding=100
                )
                placeholder.pack(pady=20)
        except Exception as e:
            print(f"Error loading image: {e}")
            # Fallback if image loading fails
            placeholder = ttk.Label(
                self.welcome_frame,
                text="[Solar Radiation Image Placeholder]",
                font=("Arial", 16),
                background="#f0f0f0",
                padding=100
            )
            placeholder.pack(pady=20)

        # Description
        description = ttk.Label(
            self.welcome_frame,
            text=(
                "Welcome to the Solar Radiation Prediction System.\n"
                "This application uses a deep learning model combining CNN, BiLSTM, and Attention\n"
                "to predict solar radiation based on meteorological data.\n\n"
                "Click 'Train Model' to start or load a pre-trained model."
            ),
            font=("Arial", 12),
            justify="center"
        )
        description.pack(pady=20)

        # Train model button
        self.welcome_train_btn = ttk.Button(
            self.welcome_frame,
            text="Train Model",
            command=self.start_main_application,
            style="Accent.TButton",  # Custom style for emphasis
            padding=10
        )
        self.welcome_train_btn.pack(pady=20)

        # Create a custom style for the accent button
        self.style = ttk.Style()
        self.style.configure("Accent.TButton", font=("Arial", 12, "bold"))

    def start_main_application(self):
        """Transition from welcome screen to main application"""
        # Remove welcome frame
        self.welcome_frame.destroy()

        # Create main application widgets
        self.create_widgets()

        # Start training if no model exists
        if self.model is None:
            self.start_training()

    def create_widgets(self):
        """Create all main application widgets"""
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Control panel frame
        control_frame = ttk.Frame(main_frame, padding="5")
        control_frame.pack(fill=tk.X, side=tk.TOP, pady=10)

        # Training section
        training_frame = ttk.LabelFrame(control_frame, text="Model Training", padding=5)
        training_frame.pack(fill=tk.X, side=tk.TOP, padx=5, pady=5)

        self.train_btn = ttk.Button(training_frame, text="Train Model", command=self.start_training)
        self.train_btn.pack(side=tk.LEFT, padx=5, pady=5)

        # Prediction section
        prediction_frame = ttk.LabelFrame(control_frame, text="Prediction", padding=5)
        prediction_frame.pack(fill=tk.X, side=tk.TOP, padx=5, pady=5)

        self.predict_btn = ttk.Button(prediction_frame, text="Predict", command=self.predict)
        self.predict_btn.pack(side=tk.LEFT, padx=5, pady=5)

        self.step_label = ttk.Label(prediction_frame, text="Steps:")
        self.step_label.pack(side=tk.LEFT, padx=5, pady=5)

        self.step_entry = ttk.Entry(prediction_frame, width=5)
        self.step_entry.pack(side=tk.LEFT, padx=5, pady=5)
        self.step_entry.insert(0, "12")  # Default value

        # Analysis section - initially disabled until training is complete
        self.analysis_frame = ttk.LabelFrame(control_frame, text="Model Analysis", padding=5)
        self.analysis_frame.pack(fill=tk.X, side=tk.TOP, padx=5, pady=5)

        self.attention_single_btn = ttk.Button(
            self.analysis_frame,
            text="Attention (Single)",
            command=self.show_single_attention,
            state=tk.DISABLED
        )
        self.attention_single_btn.grid(row=0, column=0, padx=5, pady=5)

        self.attention_avg_btn = ttk.Button(
            self.analysis_frame,
            text="Attention (Average)",
            command=self.show_average_attention,
            state=tk.DISABLED
        )
        self.attention_avg_btn.grid(row=0, column=1, padx=5, pady=5)

        self.activation_btn = ttk.Button(
            self.analysis_frame,
            text="Activation Pattern",
            command=self.but1_fun,
            state=tk.DISABLED
        )
        self.activation_btn.grid(row=0, column=2, padx=5, pady=5)

        self.attention_weight_btn = ttk.Button(
            self.analysis_frame,
            text="Attention Weight",
            command=self.but2_fun,
            state=tk.DISABLED
        )
        self.attention_weight_btn.grid(row=0, column=3, padx=5, pady=5)

        self.global_perf_btn = ttk.Button(
            self.analysis_frame,
            text="Global Performance",
            command=self.but3_fun,
            state=tk.DISABLED
        )
        self.global_perf_btn.grid(row=0, column=4, padx=5, pady=5)

        # Data exploration section - initially disabled
        self.data_frame = ttk.LabelFrame(control_frame, text="Data Exploration", padding=5)
        self.data_frame.pack(fill=tk.X, side=tk.TOP, padx=5, pady=5)

        self.radiation_btn = ttk.Button(
            self.data_frame,
            text="Radiation Distribution",
            command=self.show_radiation_distribution,
            state=tk.DISABLED
        )
        self.radiation_btn.pack(side=tk.LEFT, padx=5, pady=5)

        # Training visualization frame
        self.figure_frame = ttk.Frame(main_frame, padding="5")
        self.figure_frame.pack(fill=tk.BOTH, expand=True, side=tk.TOP)

        self.train_fig = Figure(figsize=(8, 3))
        self.train_ax_loss = self.train_fig.add_subplot(131)
        self.train_ax_mae = self.train_fig.add_subplot(132)
        self.train_ax_mape = self.train_fig.add_subplot(133)
        self.train_canvas = FigureCanvasTkAgg(self.train_fig, master=self.figure_frame)
        self.train_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Results visualization frame
        self.figure_frame2 = ttk.Frame(main_frame, padding="5")
        self.figure_frame2.pack(fill=tk.BOTH, expand=True, side=tk.TOP)
        self.fig = Figure(figsize=(8, 4))
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.figure_frame2)
        self.canvas.get_tk_widget().pack_forget()  # Initially hidden

    def load_saved_model(self):
        """Load pre-trained model if available"""
        if os.path.exists('best_model.h5'):
            self.model = load_model('best_model.h5')
            print("Loaded pre-trained model from 'best_model.h5'")
        else:
            print("No pre-trained model found. Please train a new model.")

    def start_training(self):
        """Start the model training in a separate thread"""
        self.train_btn.config(state=tk.DISABLED, text="Training...")
        threading.Thread(target=self.train_model).start()

    def train_model(self):
        """Train the deep learning model with real-time visualization updates"""
        self.root.after(0, lambda: messagebox.showinfo("Training Started",
                                                       "Model training has started. This may take a few minutes."))
        model = self.build_model(self.input_shape)

        self.train_history = {'loss': [], 'val_loss': [], 'mae': [], 'val_mae': [], 'mape': [], 'val_mape': []}

        class CustomCallback(Callback):
            def __init__(self, parent):
                super().__init__()
                self.parent = parent

            def on_epoch_end(self, epoch, logs=None):
                self.parent.train_history['loss'].append(logs.get('loss', 0))
                self.parent.train_history['val_loss'].append(logs.get('val_loss', 0))
                self.parent.train_history['mae'].append(logs.get('mae', 0))
                self.parent.train_history['val_mae'].append(logs.get('val_mae', 0))
                self.parent.train_history['mape'].append(logs.get('mape', 0))
                self.parent.train_history['val_mape'].append(logs.get('val_mape', 0))
                self.parent.root.after(0, self.parent.update_train_plot)

        callback = CustomCallback(self)
        model.fit(self.X_train, self.y_train, validation_data=(self.X_val, self.y_val),
                  epochs=20, batch_size=64, callbacks=[callback], verbose=0)

        model.save('best_model.h5')
        self.model = model

        # Enable all analysis buttons after training completes
        self.root.after(0, self.enable_analysis_buttons)
        self.root.after(0, lambda: messagebox.showinfo("Training Completed", "Model training has finished!"))
        self.root.after(0, lambda: self.train_btn.config(state=tk.NORMAL, text="Retrain Model"))
        self.root.after(0, self.update_train_plot)

    def enable_analysis_buttons(self):
        """Enable all analysis buttons after model training"""
        # Enable all buttons in the analysis section
        for button in [self.attention_single_btn, self.attention_avg_btn,
                       self.activation_btn, self.attention_weight_btn,
                       self.global_perf_btn, self.radiation_btn]:
            button.config(state=tk.NORMAL)

    def preprocess_data(self, df):
        """Preprocess the solar radiation dataset for deep learning"""
        # Convert date-time fields
        df["Datetime"] = pd.to_datetime(df["Data"], format="%m/%d/%Y %I:%M:%S %p")
        df["DatePart"] = df["Datetime"].dt.date

        # Validate and filter time records
        valid_time_mask = df["Time"].apply(validate_time)
        df = df[valid_time_mask].copy()
        df["Time"] = pd.to_datetime(df["Time"], format="%H:%M:%S").dt.time
        df["Datetime"] = pd.to_datetime(df["DatePart"].astype(str) + " " + df["Time"].astype(str))

        # Process sunrise/sunset data
        df["SunRise"] = pd.to_datetime(df["DatePart"].astype(str) + " " + df["TimeSunRise"], format="%Y-%m-%d %H:%M:%S")
        df["SunSet"] = pd.to_datetime(df["DatePart"].astype(str) + " " + df["TimeSunSet"], format="%Y-%m-%d %H:%M:%S")
        mask = df["SunSet"] < df["SunRise"]
        df.loc[mask, "SunSet"] += pd.Timedelta(days=1)

        # Feature engineering
        df['Hour'] = df['Datetime'].dt.hour
        df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
        df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
        df['DaylightDuration'] = (df['SunSet'] - df['SunRise']).dt.total_seconds() / 3600
        df['IsDaylight'] = ((df['Datetime'] >= df['SunRise']) & (df['Datetime'] <= df['SunSet'])).astype(int)
        df['SinceSunrise'] = (df['Datetime'] - df['SunRise']).dt.total_seconds() / 3600
        df['ToSunset'] = (df['SunSet'] - df['Datetime']).dt.total_seconds() / 3600
        df['WindDirection_sin'] = np.sin(np.radians(df['WindDirection(Degrees)']))
        df['WindDirection_cos'] = np.cos(np.radians(df['WindDirection(Degrees)']))

        # Standardize numeric features
        numeric_cols = ['Temperature', 'Pressure', 'Humidity', 'Speed']
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

        # Create sequence data for time series prediction
        LOOKBACK = 24
        FORECAST = 12
        features = ['Radiation', 'Hour_sin', 'Hour_cos', 'Temperature', 'Pressure',
                    'Humidity', 'WindDirection_sin', 'WindDirection_cos', 'Speed',
                    'IsDaylight', 'SinceSunrise', 'ToSunset']

        def create_sequences(data):
            """Create sliding window sequences for time series data"""
            X, y = [], []
            for i in range(len(data) - LOOKBACK - FORECAST + 1):
                X.append(data.iloc[i:i + LOOKBACK][features].values)
                y.append(data.iloc[i + LOOKBACK:i + LOOKBACK + FORECAST]['Radiation'].values)
            return np.array(X), np.array(y)

        X, y = create_sequences(df)

        # Split data into train, validation, and test sets
        total = len(X)
        train_size = int(0.8 * total)
        val_size = int(0.9 * total)

        X_train, y_train = X[:train_size], y[:train_size]
        X_val, y_val = X[train_size:val_size], y[train_size:val_size]
        X_test, y_test = X[val_size:], y[val_size:]

        return X_train, y_train, X_val, y_val, X_test, y_test

    def build_model(self, input_shape):
        """Build the CNN-BiLSTM-Attention deep learning model"""
        inputs = Input(shape=input_shape)

        # Dual channel CNN
        conv3 = Conv1D(64, 3, activation='relu', padding='same')(inputs)
        conv5 = Conv1D(64, 5, activation='relu', padding='same')(inputs)
        merged = concatenate([conv3, conv5])

        # BiLSTM layer
        bilstm = Bidirectional(LSTM(128, return_sequences=True))(merged)

        # Attention mechanism
        attention = Dense(1, activation='tanh')(bilstm)
        attention = Softmax(axis=1)(attention)
        context = multiply([bilstm, attention])

        # Output layers
        flat = Flatten()(context)
        dense = Dense(256, activation='relu')(flat)
        outputs = Dense(12)(dense)

        model = Model(inputs, outputs)
        model.compile(optimizer=Adam(), loss=tf.keras.losses.MeanSquaredError(), metrics=['mae', 'mape'])
        return model

    def update_train_plot(self):
        """Update the training metrics visualization"""
        self.train_ax_loss.clear()
        self.train_ax_loss.plot(self.train_history['loss'], label='Train Loss')
        self.train_ax_loss.plot(self.train_history['val_loss'], label='Validation Loss')
        self.train_ax_loss.set_title('Loss')
        self.train_ax_loss.legend()

        self.train_ax_mae.clear()
        self.train_ax_mae.plot(self.train_history['mae'], label='Train MAE')
        self.train_ax_mae.plot(self.train_history['val_mae'], label='Validation MAE')
        self.train_ax_mae.set_title('MAE')
        self.train_ax_mae.legend()

        self.train_ax_mape.clear()
        self.train_ax_mape.plot(self.train_history['mape'], label='Train MAPE')
        self.train_ax_mape.plot(self.train_history['val_mape'], label='Validation MAPE')
        self.train_ax_mape.set_title('MAPE')
        self.train_ax_mape.legend()
        self.train_fig.tight_layout()
        self.train_canvas.draw()

    def predict(self):
        """Make predictions and visualize comparison with actual values"""
        if not self.model:
            messagebox.showerror("Error", "No trained model found. Please train or load a model first")
            return
        try:
            steps = int(self.step_entry.get())
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter a valid integer for steps.")
            return

        if steps <= 0 or steps > len(self.y_test[0]):
            messagebox.showerror("Invalid Input",
                                 f"Steps must be between 1 and {len(self.y_test[0])} (max available steps)")
            return

        # Get a sample and make prediction
        sample = self.X_val[0:1]
        prediction = self.model.predict(sample)
        actual = self.y_val[0][:steps]
        predicted = prediction[0][:steps]

        # Enhanced visualization with error analysis
        self.fig.clear()

        # Create two subplots
        ax1 = self.fig.add_subplot(211)
        ax2 = self.fig.add_subplot(212)

        # Plot prediction vs actual values
        ax1.plot(actual, label='Actual', marker='o')
        ax1.plot(predicted, label='Predicted', marker='x')
        ax1.set_title(f'{steps}-step Radiation Prediction')
        ax1.legend()
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Radiation Value')

        # Plot prediction errors
        errors = actual - predicted
        ax2.bar(range(len(errors)), errors)
        ax2.set_title('Prediction Errors')
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Error (Actual - Predicted)')
        ax2.axhline(y=0, color='r', linestyle='-', alpha=0.3)

        self.fig.tight_layout()
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def show_single_attention(self):
        """Visualize attention weights for a single sample"""
        attention_model = Model(
            inputs=self.model.input,
            outputs=self.model.get_layer("softmax").output
        )
        attention_weights = attention_model.predict(self.X_val[:100])  # (100, 24, 1)
        sample_idx = 0
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.plot(attention_weights[sample_idx].squeeze(), 'o-', color='#FF6B6B', linewidth=2)
        ax.set_title(f"Sample {sample_idx} Temporal Attention (LOOKBACK={self.X_val.shape[1]})")
        ax.set_xlabel("Historical Time Steps (5-min interval)")
        ax.set_ylabel("Attention Weight")
        ax.set_xticks(range(0, self.X_val.shape[1], 3))
        ax.set_xticklabels([f"t-{self.X_val.shape[1] - i}" for i in range(0, self.X_val.shape[1], 3)])
        self.fig.tight_layout()
        self.canvas.get_tk_widget().pack_forget()
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def show_average_attention(self):
        """Visualize average attention weights across samples"""
        attention_model = Model(
            inputs=self.model.input,
            outputs=self.model.get_layer("softmax").output
        )
        attention_weights = attention_model.predict(self.X_val[:100])  # (100, 24, 1)
        mean_weights = np.mean(attention_weights, axis=0).squeeze()
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.plot(mean_weights, 's-', color='#4ECDC4', linewidth=2)
        ax.set_title("Average Attention Weights Across Samples")
        ax.set_xlabel("Historical Time Steps (5-min interval)")
        ax.set_ylabel("Mean Weight")
        ax.set_xticks(range(0, self.X_val.shape[1], 3))
        ax.set_xticklabels([f"t-{self.X_val.shape[1] - i}" for i in range(0, self.X_val.shape[1], 3)])
        self.fig.tight_layout()
        self.canvas.get_tk_widget().pack_forget()
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def but1_fun(self):
        """Visualize BiLSTM neuron activation patterns"""
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        bilstm_model = Model(inputs=self.model.input, outputs=self.model.get_layer("bidirectional").output)
        bilstm_output = bilstm_model.predict(self.X_val[:10])  # (10, 24, 256)
        ax.imshow(bilstm_output[0].T, cmap='viridis', aspect='auto')
        ax.set_xlabel("Time Steps")
        ax.set_ylabel("Neuron Index")
        ax.set_title("BiLSTM Layer Activation Pattern (Sample 0)")
        ax.set_xticks(range(0, 24, 3))
        self.fig.tight_layout()
        self.canvas.get_tk_widget().pack_forget()
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def but2_fun(self):
        """Visualize average attention weights across time steps"""
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        attention_model = Model(
            inputs=self.model.input,
            outputs=self.model.get_layer("softmax").output
        )
        att_weights = attention_model.predict(self.X_val[:100])  # (100, 24, 1)
        ax.plot(np.mean(att_weights, axis=0).squeeze(), 'o-', color='#E64A45')
        ax.set_title("Average Attention Weights Across Time Steps")
        ax.set_xlabel("Time Step (5-min interval)")
        ax.set_ylabel("Attention Weight")
        ax.set_xticks(range(0, 24, 3))
        ax.set_xticklabels([f"t-{24 - i}" for i in range(0, 24, 3)])
        ax.grid(alpha=0.3)
        self.fig.tight_layout()
        self.canvas.get_tk_widget().pack_forget()
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def but3_fun(self):
        """Visualize global prediction performance metrics"""
        self.fig.clear()
        y_pred = self.model.predict(self.X_test)
        ax = self.fig.add_subplot(111)
        test_rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        test_r2 = r2_score(self.y_test, y_pred)
        # Draw joint distribution map
        sns.histplot(x=self.y_test.flatten(), y=y_pred.flatten(), bins=30, cmap="Blues_d", ax=ax)
        ax.plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()], 'r--', lw=2)

        ax.set_xlabel('True Values', fontsize=12)
        ax.set_ylabel('Predictions', fontsize=12)
        ax.set_title(f'Global Prediction Performance (RMSE={test_rmse:.2f}, R²={test_r2:.2f})', fontsize=14)

        # Add the ideal diagonal
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        lims = [max(x0, y0), min(x1, y1)]
        ax.plot(lims, lims, 'r--', alpha=0.7, linewidth=1.5)

        self.fig.tight_layout()
        self.canvas.get_tk_widget().pack_forget()
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def show_radiation_distribution(self):
        """Visualize radiation feature distribution and characteristics"""
        self.fig.clear()

        # Create multiple subplots to show different aspects of radiation data
        ax1 = self.fig.add_subplot(221)  # Time series
        ax2 = self.fig.add_subplot(222)  # Distribution histogram
        ax3 = self.fig.add_subplot(223)  # Daily pattern
        ax4 = self.fig.add_subplot(224)  # Actual vs predicted

        # Load original data for visualization
        df = pd.read_csv("SolarPrediction.csv")

        # 1. Time series of radiation values (100 samples)
        sample_indices = np.arange(100)
        ax1.plot(df['Radiation'].iloc[sample_indices], 'b-', linewidth=1)
        ax1.set_title("Radiation Time Series (First 100 points)")
        ax1.set_xlabel("Sample Index")
        ax1.set_ylabel("Radiation Value")

        # 2. Distribution histogram of radiation values
        ax2.hist(df['Radiation'], bins=50, color='green', alpha=0.7)
        ax2.axvline(df['Radiation'].mean(), color='red', linestyle='--',
                    label=f'Mean: {df["Radiation"].mean():.2f}')
        ax2.axvline(df['Radiation'].median(), color='blue', linestyle='--',
                    label=f'Median: {df["Radiation"].median():.2f}')
        ax2.set_title("Radiation Value Distribution")
        ax2.set_xlabel("Radiation Value")
        ax2.set_ylabel("Frequency")
        ax2.legend(fontsize=8)

        # 3. Radiation by hour of day
        df["Datetime"] = pd.to_datetime(df["Data"], format="%m/%d/%Y %I:%M:%S %p")
        df['Hour'] = df['Datetime'].dt.hour
        hourly_radiation = df.groupby('Hour')['Radiation'].mean()
        ax3.bar(hourly_radiation.index, hourly_radiation.values, color='orange')
        ax3.set_title("Average Radiation by Hour of Day")
        ax3.set_xlabel("Hour")
        ax3.set_ylabel("Average Radiation")
        ax3.set_xticks(range(0, 24, 3))

        # 4. Scatter plot of radiation vs temperature
        ax4.scatter(df['Temperature'], df['Radiation'], alpha=0.3, s=10, color='purple')
        ax4.set_title("Radiation vs Temperature")
        ax4.set_xlabel("Temperature")
        ax4.set_ylabel("Radiation")

        # Add statistical information as text
        stats_text = (f"Radiation Statistics:\n"
                      f"Min: {df['Radiation'].min():.2f}\n"
                      f"Max: {df['Radiation'].max():.2f}\n"
                      f"Mean: {df['Radiation'].mean():.2f}\n"
                      f"Std: {df['Radiation'].std():.2f}")
        ax4.annotate(stats_text, xy=(0.05, 0.05), xycoords='axes fraction',
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

        self.fig.tight_layout()
        self.canvas.get_tk_widget().pack_forget()
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)


if __name__ == "__main__":
    root = tk.Tk()
    app = SolarPredictionApp(root)
    root.mainloop()

