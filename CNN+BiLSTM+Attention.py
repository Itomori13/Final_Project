#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import *
from tensorflow.keras.layers import Input, concatenate
from tensorflow.keras.layers import LSTM, Bidirectional, Conv1D, Flatten
from tensorflow.keras.models import Model

# In[2]:
# In[3]:

# load data
df = pd.read_csv("SolarPrediction.csv")

# basic time analyse
df["Datetime"] = pd.to_datetime(df["Data"], format="%m/%d/%Y %I:%M:%S %p")
df["DatePart"] = df["Datetime"].dt.date

print("Step 1 Complete - Basic Time Analysis")
print("Current column:", df.columns.tolist())
print("Data Sample:")
display(df[["Data", "Time", "Datetime"]].head(2))


# In[4]:


# validate the time format
def validate_time(time_str):
    try:
        pd.to_datetime(time_str, format="%H:%M:%S")
        return True
    except ValueError:
        return False

# delete irrelevant time
valid_time_mask = df["Time"].apply(validate_time)
df = df[valid_time_mask].copy()

# standardize time format
df["Time"] = pd.to_datetime(df["Time"], format="%H:%M:%S").dt.time

# combine accurate timestamp
df["Datetime"] = pd.to_datetime(
    df["DatePart"].astype(str) + " " + df["Time"].astype(str),
    format="%Y-%m-%d %H:%M:%S"
)

print("Step 2 Complete - Time Standardization")
print("Remaining Records:", len(df))
print("Time range:", df["Datetime"].min(), "~", df["Datetime"].max())


# In[5]:


# generate sunrise time set
df["SunRise"] = pd.to_datetime(
    df["DatePart"].astype(str) + " " + df["TimeSunRise"],
    format="%Y-%m-%d %H:%M:%S"
)

# generate sunset time set
df["SunSet"] = pd.to_datetime(
    df["DatePart"].astype(str) + " " + df["TimeSunSet"],
    format="%Y-%m-%d %H:%M:%S"
)

# generate cross time problem
mask = df["SunSet"] < df["SunRise"]
df.loc[mask, "SunSet"] += pd.Timedelta(days=1)

print("\nStep 3 Complete - Sunrise and Sunset Time Processing")
print("Example of Sunrise and Sunset Time:")
display(df[["SunRise", "SunSet"]].head(2))
print("Is there an abnormal time", df["SunRise"].gt(df["SunSet"]).any())


# In[6]:


# time feature
df['Hour'] = df['Datetime'].dt.hour
df['DayOfYear'] = df['Datetime'].dt.dayofyear
df['Season'] = df['Datetime'].dt.month % 12 // 3 + 1

# Periodic coding
df['Hour_sin'] = np.sin(2 * np.pi * df['Hour']/24)
df['Hour_cos'] = np.cos(2 * np.pi * df['Hour']/24)

# Astronomical characteristics
df['DaylightDuration'] = (df['SunSet'] - df['SunRise']).dt.total_seconds() / 3600
df['IsDaylight'] = ((df['Datetime'] >= df['SunRise']) & (df['Datetime'] <= df['SunSet'])).astype(int)
df['SinceSunrise'] = (df['Datetime'] - df['SunRise']).dt.total_seconds() / 3600
df['ToSunset'] = (df['SunSet'] - df['Datetime']).dt.total_seconds() / 3600

# Wind direction coding
df['WindDirection_sin'] = np.sin(np.radians(df['WindDirection(Degrees)']))
df['WindDirection_cos'] = np.cos(np.radians(df['WindDirection(Degrees)']))

# standardization
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
numeric_cols = ['Temperature', 'Pressure', 'Humidity', 'Speed']
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

print("\nStep4 Completion - Feature Engineering")
print("Add feature column:", [c for c in df.columns if c not in ["Data", "Time", "DatePart"]])
print("Standardized statistics:")
display(df[numeric_cols].describe().loc[["mean", "std"]])


# In[7]:


LOOKBACK = 24  # use 2-hour data before
FORECAST = 12  # predict 1 hour later

feature_columns = [
    'Radiation',
    'Hour_sin', 'Hour_cos',
    'Temperature', 'Pressure', 'Humidity',
    'WindDirection_sin', 'WindDirection_cos',
    'Speed', 'IsDaylight',
    'SinceSunrise', 'ToSunset'
]

def create_sequences(data, lookback, forecast):
    X, y = [], []
    for i in range(len(data) - lookback - forecast + 1):
        X.append(data.iloc[i:i+lookback][feature_columns].values)
        y.append(data.iloc[i+lookback:i+lookback+forecast]['Radiation'].values)
    return np.array(X), np.array(y)

X, y = create_sequences(df, LOOKBACK, FORECAST)

print("\nStep5 Complete - Dataset Construction")
print("Input shape:", X.shape)
print("Output shape:", y.shape)
print("Verification of input dimension for the first sample:", X[0].shape == (LOOKBACK, len(feature_columns)))


# In[8]:


def build_model(input_shape, forecast_steps):
    inputs = Input(shape=input_shape)
    
    # dual-path CNN
    conv3 = Conv1D(64, 3, activation='relu', padding='same')(inputs)
    conv5 = Conv1D(64, 5, activation='relu', padding='same')(inputs)
    merged_conv = concatenate([conv3, conv5])
    
    # double linked BiLSTM
    bilstm = Bidirectional(LSTM(128, return_sequences=True))(merged_conv)
    
    # time attention
    attention = Dense(1, activation='tanh')(bilstm)  # time attention weight
    attention = Softmax(axis=1)(attention)
    context = multiply([bilstm, attention])
    
    # output layer
    flattened = Flatten()(context)
    dense = Dense(256, activation='relu')(flattened)
    outputs = Dense(forecast_steps)(dense)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse', metrics=['mae','mape'])
    return model

model = build_model((LOOKBACK, len(feature_columns)), FORECAST)
print("\nStep6 Completion - Model Construction")
model.summary()


# In[9]:


# data split
total_samples = len(X)
train_size = int(0.7 * total_samples)
val_size = int(0.85 * total_samples)

X_train, y_train = X[:train_size], y[:train_size]
X_val, y_val = X[train_size:val_size], y[train_size:val_size]
X_test, y_test = X[val_size:], y[val_size:]

# train set
callbacks = [
    EarlyStopping(patience=15, restore_best_weights=True),
    ModelCheckpoint("best_model.h5", save_best_only=True)
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=64,
    callbacks=callbacks,
    verbose=1
)


# In[11]:


# load best model
from tensorflow.keras.models import load_model
best_model = load_model("best_model.h5")

# directly use trained model if possible
val_results = model.evaluate(X_val, y_val, verbose=0)

print("\nTest results:")
print(f"Loss (MSE): {val_results[0]:.4f}")
print(f"MAE: {val_results[1]:.4f}")
print(f"MAPE: {val_results[2]:.2f}")


# In[12]:


import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.title('Loss')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(history.history['mae'], label='train')
plt.plot(history.history['val_mae'], label='val')
plt.title('MAE')
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(history.history['mape'], label='train')
plt.plot(history.history['val_mape'], label='val')
plt.title('MAPE')
plt.legend()

plt.tight_layout()
plt.show()


# In[13]:


# Extract attention weights submodel
attention_model = Model(
    inputs=model.input,
    outputs=model.get_layer("softmax").output  # match layer with model
)

# Get attention weights for validation samples
attention_weights = attention_model.predict(X_val[:100])  # shape: (100, 24, 1)

# Visualization
plt.figure(figsize=(12, 6))

# Single sample
plt.subplot(2, 1, 1)
sample_idx = 0
plt.plot(attention_weights[sample_idx].squeeze(), 'o-', color='#FF6B6B', linewidth=2)
plt.title(f"Sample {sample_idx} Temporal Attention (LOOKBACK=24)")
plt.xlabel("Historical Time Steps (5-min interval)")
plt.ylabel("Attention Weight")
plt.xticks(range(0, 24, 3), [f"t-{24-i}" for i in range(0, 24, 3)])

# Average across samples
plt.subplot(2, 1, 2)
mean_weights = np.mean(attention_weights, axis=0).squeeze()
plt.plot(mean_weights, 's-', color='#4ECDC4', linewidth=2)
plt.title("Average Attention Weights Across Samples")
plt.xlabel("Historical Time Steps (5-min interval)")
plt.ylabel("Mean Weight")
plt.xticks(range(0, 24, 3), [f"t-{24-i}" for i in range(0, 24, 3)])

plt.tight_layout()
plt.show()


# In[14]:



# Extract CNN outputs
conv3_model = Model(inputs=model.input, outputs=model.get_layer("conv1d").output)
conv5_model = Model(inputs=model.input, outputs=model.get_layer("conv1d_1").output)

# Calculate activation magnitudes
conv3_act = np.mean(np.abs(conv3_model.predict(X_val[:100])))
conv5_act = np.mean(np.abs(conv5_model.predict(X_val[:100])))

# Visualization
plt.figure(figsize=(6, 4))
plt.bar(['3x3 Conv', '5x5 Conv'], [conv3_act, conv5_act], color=['#FF9F40', '#55CBCD'])
plt.ylabel("Mean Activation Magnitude")
plt.title("Dual-path CNN Contribution Comparison")
plt.grid(axis='y', alpha=0.3)


# In[15]:



# Extract BiLSTM outputs
bilstm_model = Model(inputs=model.input, outputs=model.get_layer("bidirectional").output)
bilstm_output = bilstm_model.predict(X_val[:10])  # (10, 24, 256)

# Activation heatmap
plt.figure(figsize=(12, 6))
plt.imshow(bilstm_output[0].T, cmap='viridis', aspect='auto')
plt.colorbar(label="Activation Strength")
plt.xlabel("Time Steps")
plt.ylabel("Neuron Index")
plt.title("BiLSTM Layer Activation Pattern (Sample 0)")
plt.xticks(range(0, 24, 3))


# In[16]:


# obtain attention weight
attention_model = Model(
    inputs=model.input,
    outputs=model.get_layer("softmax").output
)
att_weights = attention_model.predict(X_val[:100])  # (100, 24, 1)

# Visualize the average attention distribution
plt.figure(figsize=(10, 4))
plt.plot(np.mean(att_weights, axis=0).squeeze(), 'o-', color='#E64A45')
plt.title("Average Attention Weights Across Time Steps")
plt.xlabel("Time Step (5-min interval)")
plt.ylabel("Attention Weight")
plt.xticks(range(0, 24, 3), [f"t-{24-i}" for i in range(0, 24, 3)])
plt.grid(alpha=0.3)
plt.show()


# In[17]:


# predict model construction
y_pred = model.predict(X_test)


plt.figure(figsize=(15, 9))
for i in np.random.choice(range(len(y_test)), 2):
    plt.subplot(3, 1, (i%3)+1)
    plt.plot(y_test[i], label='True')
    plt.plot(y_pred[i], label='Predicted')
    plt.title(f'Sample {i} Forecast Comparison')
    plt.xlabel('Time Steps')
    plt.ylabel('Radiation')
    plt.legend()
plt.tight_layout()
plt.show()


# In[18]:


# calculation
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
test_r2 = r2_score(y_test, y_pred)

# Create  joint distribution map
g = sns.jointplot(x=y_test.flatten(), 
                y=y_pred.flatten(), 
                kind='hex',
                height=8,
                ratio=5,
                space=0.2,
                joint_kws={'gridsize': 50},
                marginal_kws={'bins': 30, 'fill': True})


g.set_axis_labels('True Values', 'Predictions', fontsize=12)
g.fig.suptitle(f'Global Prediction Performance (RMSE={test_rmse:.2f}, R²={test_r2:.2f})', 
             y=1.02,
             fontsize=14)

# Add the ideal diagonal
x0, x1 = g.ax_joint.get_xlim()
y0, y1 = g.ax_joint.get_ylim()
lims = [max(x0, y0), min(x1, y1)]
g.ax_joint.plot(lims, lims, 'r--', alpha=0.7, linewidth=1.5)


plt.colorbar(g.ax_joint.collections[0], ax=g.ax_joint, label='Data density')


plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.show()


# In[19]:


errors = y_test - y_pred
plt.figure(figsize=(10, 5))
sns.histplot(errors.flatten(), bins=50, kde=True)
plt.title('Prediction Error Distribution')
plt.xlabel('Prediction Error')
plt.ylabel('Frequency')
plt.axvline(x=0, color='r', linestyle='--')
plt.show()


# In[20]:


step_rmse = np.sqrt(np.mean((y_test - y_pred)**2, axis=0))
step_mae = np.mean(np.abs(y_test - y_pred), axis=0)

plt.figure(figsize=(12, 6))
plt.plot(step_rmse, 's-', label='RMSE per Step')
plt.plot(step_mae, 'o-', label='MAE per Step')
plt.title('Error Analysis by Forecast Step')
plt.xlabel('Forecast Time Step')
plt.ylabel('Error Value')
plt.xticks(range(FORECAST), [f'T+{i+1}' for i in range(FORECAST)])
plt.legend()
plt.grid(True)
plt.show()


# In[21]:


plt.figure(figsize=(10, 6))
plt.scatter(y_test.flatten(), y_pred.flatten(), 
            alpha=0.3, 
            c='blue',
            edgecolors='w')
plt.plot([y_test.min(), y_test.max()], 
         [y_test.min(), y_test.max()], 
         'r--', lw=2)
plt.xlabel('True Values', fontsize=12)
plt.ylabel('Predictions', fontsize=12)
plt.title(f'True vs Predicted Values (R²={test_r2:.2f})', fontsize=14)
plt.grid(True)
plt.colorbar(label='Data density')
plt.show()


# In[22]:


plt.figure(figsize=(10, 6))
quantiles = np.percentile(y_test.flatten(), np.linspace(0, 100, 20))
pred_quantiles = np.percentile(y_pred.flatten(), np.linspace(0, 100, 20))

plt.plot(quantiles, pred_quantiles, 'bo-')
plt.plot([quantiles.min(), quantiles.max()], 
         [quantiles.min(), quantiles.max()], 
         'r--', lw=2)
plt.xlabel('True Value Quantiles', fontsize=12)
plt.ylabel('Predicted Value Quantiles', fontsize=12)
plt.title('Quantile-Quantile Comparison', fontsize=14)
plt.grid(True)
plt.show()


# In[23]:


errors = y_pred - y_test
plt.figure(figsize=(12, 6))
sns.boxplot(data=pd.DataFrame(errors, 
                            columns=[f'T+{i+1}' for i in range(FORECAST)]),
           orient='v',
           palette='Set2')
plt.axhline(0, color='r', linestyle='--')
plt.title('Prediction Error Distribution per Time Step')
plt.ylabel('Prediction Error')
plt.xlabel('Forecast Step')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()


# In[24]:


plt.figure(figsize=(10, 6))
sns.kdeplot(y_test.flatten(), 
           label='True Values', 
           color='blue',
           linewidth=2)
sns.kdeplot(y_pred.flatten(), 
           label='Predictions', 
           color='red',
           linestyle='--',
           linewidth=2)
plt.xlabel('Radiation Value')
plt.ylabel('Density')
plt.title('Probability Distribution Comparison')
plt.legend()
plt.grid(True)
plt.show()


# In[25]:


plt.figure(figsize=(12, 8))

# Use hexbin to handle high-density data
hb = plt.hexbin(y_test.flatten(), y_pred.flatten(), 
                gridsize=100, 
                cmap='viridis',
                mincnt=1,
                bins='log')

plt.plot([y_test.min(), y_test.max()], 
         [y_test.min(), y_test.max()], 
         'r--', lw=2)
plt.colorbar(hb, label='log10(N)')
plt.xlabel('True Values', fontsize=12)
plt.ylabel('Predictions', fontsize=12)
plt.title(f'Complete Value Comparison (N={len(y_test.flatten()):,} points)', fontsize=14)
plt.grid(alpha=0.3)
plt.show()


# In[29]:


fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(18, 12), sharey=True)
axes = axes.flatten()
for step in range(y_test.shape[1]):
    try:
        ax = axes[step]
        ax.scatter(y_test[:, step], y_pred[:, step], 
                   alpha=0.3, 
                   c='teal',
                   edgecolors='none')
        ax.plot([y_test.min(), y_test.max()], 
                [y_test.min(), y_test.max()], 
                'r--', lw=1.5)
        ax.set_title(f'Step T+{step+1}')
        ax.set_xlabel('True')
        ax.grid(alpha=0.3)

        # add index
        rmse = np.sqrt(mean_squared_error(y_test[:, step], y_pred[:, step]))
        ax.text(0.05, 0.85, f'RMSE: {rmse:.2f}', 
                transform=ax.transAxes,
                backgroundcolor='white')
    except:
        pass

plt.suptitle('Per-Step Prediction Performance', y=1.02, fontsize=16)
plt.tight_layout()
plt.show()


# In[30]:


plt.figure(figsize=(15, 8))

# use the sample
sample_idx = np.random.choice(len(y_test), 200, replace=False)

# generate curve
for idx in sample_idx:
    plt.plot(y_test[idx], color='blue', alpha=0.03, lw=1)
    plt.plot(y_pred[idx], color='red', alpha=0.03, lw=1)

# generate average curve
plt.plot(np.mean(y_test, axis=0), 'b-', lw=3, label='True Mean')
plt.plot(np.mean(y_pred, axis=0), 'r--', lw=3, label='Predicted Mean')

plt.xlabel('Forecast Steps', fontsize=12)
plt.ylabel('Radiation', fontsize=12)
plt.title('Complete Temporal Comparison', fontsize=14)
plt.legend()
plt.grid(alpha=0.3)
plt.show()


# In[31]:

#matrix generation
error_matrix = np.abs(y_pred - y_test)

plt.figure(figsize=(15, 8))
sns.heatmap(error_matrix.T,
            cmap='YlOrRd',
            cbar_kws={'label': 'Absolute Error'},
            vmin=0,
            vmax=np.percentile(error_matrix, 95))

plt.xlabel('Sample Index', fontsize=12)
plt.ylabel('Forecast Steps', fontsize=12)
plt.title('Spatiotemporal Error Distribution', fontsize=14)
plt.xticks([])
plt.yticks(ticks=range(0, y_test.shape[1], 2), 
           labels=[f'T+{i+1}' for i in range(0, y_test.shape[1], 2)])
plt.show()


# In[32]:


error = y_pred - y_test

plt.figure(figsize=(15, 8))
sns.boxplot(data=pd.DataFrame(error, 
                            columns=[f'T+{i+1}' for i in range(error.shape[1])]),
           orient='v',
           palette='coolwarm',
           showfliers=False,
           notch=True)

plt.axhline(0, color='red', linestyle='--', linewidth=1.5)
plt.title('Error Distribution Across Forecast Steps')
plt.ylabel('Prediction Error')
plt.xlabel('Forecast Step')
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)
plt.show()


# In[33]:


plt.figure(figsize=(12, 7))

# calculate CDF
def ecdf(data):
    x = np.sort(data)
    y = np.arange(1, len(x)+1)/len(x)
    return x, y

x_true, y_true = ecdf(y_test.flatten())
x_pred, y_pred_cdf = ecdf(y_pred.flatten())

plt.plot(x_true, y_true, label='True CDF', lw=2)
plt.plot(x_pred, y_pred_cdf, label='Predicted CDF', lw=2, linestyle='--')
plt.fill_betweenx(y_true, x_true, x_pred, 
                 where=(x_pred >= x_true), 
                 color='green', alpha=0.1,
                 label='Over-prediction Area')
plt.fill_betweenx(y_true, x_true, x_pred,
                 where=(x_pred < x_true),
                 color='red', alpha=0.1,
                 label='Under-prediction Area')

plt.title('Empirical Cumulative Distribution Function Comparison')
plt.xlabel('Radiation Value')
plt.ylabel('Cumulative Probability')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.show()


# In[34]:


def plot_time_step_distribution(data, title):
    plt.figure(figsize=(15, 6))
    
    # Use kernel density estimation
    sns.violinplot(data=pd.DataFrame(data, 
                                   columns=[f'T+{i+1}' for i in range(data.shape[1])]),
                  palette="Spectral",
                  inner="quartile",
                  cut=0)
    
    plt.title(title)
    plt.xlabel('Forecast Step')
    plt.ylabel('Radiation Value')
    plt.xticks(rotation=45)
    plt.grid(alpha=0.3)

plot_time_step_distribution(y_test, 'True Values Distribution Across Steps')
plot_time_step_distribution(y_pred, 'Predicted Values Distribution Across Steps')


# In[35]:


import pandas as pd
from scipy import stats

# create data frame
df = pd.DataFrame({
    'True': y_test.flatten(),
    'Predicted': y_pred.flatten(),
    'Residual': (y_pred - y_test).flatten()
})

# Calculate statistical indicators
pearson_r, pearson_p = stats.pearsonr(df['True'], df['Predicted'])
spearman_r = stats.spearmanr(df['True'], df['Predicted']).correlation

# Draw the enhanced scatter plot
g = sns.jointplot(data=df, x='True', y='Predicted',
                 kind='reg', 
                 scatter_kws={'alpha':0.3, 's':5},
                 line_kws={'color':'red', 'lw':2},
                 ratio=4,
                 marginal_ticks=True)


text = (f"Pearson r = {pearson_r:.3f} (p={pearson_p:.2e})\n"
        f"Spearman ρ = {spearman_r:.3f}\n"
        f"N = {len(df):,}")
g.ax_joint.text(0.05, 0.95, text, 
               transform=g.ax_joint.transAxes,
               ha='left', va='top',
               bbox=dict(facecolor='white', alpha=0.8))

# Add quantile reference lines
for q in [0.1, 0.5, 0.9]:
    g.ax_joint.axhline(np.quantile(df['Predicted'], q), 
                      color='grey', ls=':', alpha=0.5)
    g.ax_joint.axvline(np.quantile(df['True'], q), 
                      color='grey', ls=':', alpha=0.5)

plt.suptitle('Enhanced Correlation Analysis', y=1.02)
plt.show()


# In[36]:


# Calculate the correlation coefficients of each time step
corr_matrix = np.array([[
    np.corrcoef(y_test[:,i], y_pred[:,j])[0,1] 
    for j in range(y_pred.shape[1])]
    for i in range(y_test.shape[1])
])

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix,
           annot=True,
           fmt=".2f",
           cmap='coolwarm',
           vmin=-1, vmax=1,
           mask=np.triu(np.ones_like(corr_matrix)),
           linewidths=0.5,
           cbar_kws={'label': 'Pearson Correlation'})

plt.title('Spatiotemporal Correlation Matrix')
plt.xlabel('Prediction Steps')
plt.ylabel('True Value Steps')
plt.xticks(ticks=np.arange(0.5, y_pred.shape[1]+0.5), 
          labels=[f'P_T+{i+1}' for i in range(y_pred.shape[1])])
plt.yticks(ticks=np.arange(0.5, y_test.shape[1]+0.5), 
          labels=[f'T_T+{i+1}' for i in range(y_test.shape[1])])
plt.grid(False)
plt.show()


# In[37]:


quantiles = np.linspace(0, 1, 100)
q_true = np.quantile(y_test, quantiles)
q_pred = np.quantile(y_pred, quantiles)

plt.figure(figsize=(10, 8))
sns.regplot(x=q_true, y=q_pred, 
           scatter_kws={'alpha':0.6},
           line_kws={'color':'red', 'lw':2})

# Draw the confidence interval
sns.regplot(x=q_true, y=q_pred, 
           ci=99, 
           scatter=False, 
           line_kws={'color':'red', 'alpha':0.2})

plt.plot([q_true.min(), q_true.max()], 
        [q_true.min(), q_true.max()], 
        'k--', alpha=0.5)
plt.title('Quantile Correlation Plot')
plt.xlabel('True Values Quantiles')
plt.ylabel('Predicted Values Quantiles')
plt.grid(alpha=0.3)
plt.text(0.05, 0.9, 
        f'Kendall τ = {stats.kendalltau(q_true, q_pred)[0]:.3f}',
        transform=plt.gca().transAxes,
        bbox=dict(facecolor='white', alpha=0.8))
plt.show()


# In[40]:


import lime.lime_tabular
import matplotlib.pyplot as plt

# Define the feature name (combining time steps and features)
feature_names = [
    f"t-{LOOKBACK-i-1}_{feat}"
    for i in range(LOOKBACK)
    for feat in feature_columns
]

# Lime creation
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train.reshape(-1, LOOKBACK*len(feature_columns))[:1000],  # use trained data
    feature_names=feature_names,
    mode="regression",
    discretize_continuous=False,
    verbose=True
)

# generate predict function
def predict_fn(x):
    # Convert the 2D input into a 3D time series format
    x_reshaped = x.reshape(-1, LOOKBACK, len(feature_columns))
    return model.predict(x_reshaped)

# explanation
sample_idx = 0
sample_to_explain = X_test[sample_idx].flatten()


explanation = explainer.explain_instance(
    data_row=sample_to_explain,
    predict_fn=predict_fn,
    num_features=20,
    num_samples=5000
)

# visible lime explanation
plt.figure(figsize=(10, 6))
explanation.as_pyplot_figure()
plt.title(f"LIME Explanation for Sample {sample_idx}")
plt.show()

# Output the temporal feature importance map
exp_list = explanation.as_list()
print("\nTop important characteristics ：")
for feature, importance in exp_list:
    print(f"{feature}: {importance:.4f}")


# In[ ]:


