from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
tf.random.set_seed(3)
from tensorflow import keras

app = Flask(__name__)

# Train model on startup
print("Loading dataset and training model...")
breast_cancer_dataset = sklearn.datasets.load_breast_cancer()
data_frame = pd.DataFrame(breast_cancer_dataset.data, columns=breast_cancer_dataset.feature_names)
data_frame['label'] = breast_cancer_dataset.target

X = data_frame.drop(columns='label')
Y = data_frame['label']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

model = keras.Sequential([
    keras.layers.Input(shape=(30,)),
    keras.layers.Dense(20, activation='relu'),
    keras.layers.Dense(2, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train_std, Y_train, validation_split=0.1, epochs=10, verbose=0)

_, test_accuracy = model.evaluate(X_test_std, Y_test, verbose=0)
print(f"Model ready. Test accuracy: {test_accuracy*100:.2f}%")

FEATURE_NAMES = list(breast_cancer_dataset.feature_names)
FEATURE_DESCRIPTIONS = {
    'mean radius': 'Mean of distances from center to points on perimeter',
    'mean texture': 'Standard deviation of gray-scale values',
    'mean perimeter': 'Mean perimeter of the cell nucleus',
    'mean area': 'Mean area of the cell nucleus',
    'mean smoothness': 'Local variation in radius lengths',
    'mean compactness': 'Perimeter² / area - 1.0',
    'mean concavity': 'Severity of concave portions of the contour',
    'mean concave points': 'Number of concave portions of the contour',
    'mean symmetry': 'Symmetry of the cell nucleus',
    'mean fractal dimension': 'Coastline approximation - 1',
}

# Sample benign and malignant values for prefill
SAMPLE_BENIGN = (11.76,21.6,74.72,427.9,0.08637,0.04966,0.01657,0.01115,0.1495,0.05888,
                 0.4062,1.21,2.635,28.47,0.005857,0.009758,0.01168,0.007445,0.02406,0.001769,
                 12.98,25.72,82.98,516.5,0.1085,0.08615,0.05523,0.03715,0.2433,0.06563)

SAMPLE_MALIGNANT = (17.99,10.38,122.8,1001.0,0.1184,0.2776,0.3001,0.1471,0.2419,0.07871,
                    1.095,0.9053,8.589,153.4,0.006399,0.04904,0.05373,0.01587,0.03003,0.006193,
                    25.38,17.33,184.6,2019.0,0.1622,0.6656,0.7119,0.2654,0.4601,0.1189)


@app.route('/')
def index():
    features = list(zip(FEATURE_NAMES, SAMPLE_BENIGN))
    return render_template('index.html',
                           features=FEATURE_NAMES,
                           sample_benign=SAMPLE_BENIGN,
                           sample_malignant=SAMPLE_MALIGNANT,
                           accuracy=round(test_accuracy * 100, 2),
                           descriptions=FEATURE_DESCRIPTIONS)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        values = [float(data[f]) for f in FEATURE_NAMES]
        input_df = pd.DataFrame([values], columns=FEATURE_NAMES)
        input_std = scaler.transform(input_df)
        prediction = model.predict(input_std, verbose=0)
        label = int(np.argmax(prediction[0]))
        confidence = float(np.max(prediction[0])) * 100
        result = 'Benign' if label == 1 else 'Malignant'
        return jsonify({
            'result': result,
            'confidence': round(confidence, 1),
            'benign_prob': round(float(prediction[0][1]) * 100, 1),
            'malignant_prob': round(float(prediction[0][0]) * 100, 1)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True)
