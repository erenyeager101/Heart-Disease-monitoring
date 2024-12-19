

from flask import Flask, render_template, request, jsonify
import pickle
import os
import numpy as np

app = Flask(__name__)

with open('stacking_modelfile', 'rb') as f:
    model = pickle.load(f)


@app.route('/')
def mainindex():
    return render_template('mainindex.html', prediction=None)


@app.route('/templates/index.html')
def index():
    return render_template('index.html', prediction=None)


@app.route('/templates/appointment.html')
def appointment():
    return render_template('appointment.html', prediction=None)


@app.route('/templates/treatment.html')
def treatment():
    return render_template('treatment.html', prediction=None)


@app.route('/templates/change_in_lifestyle.html')
def change_in_lifestyle():
    return render_template('change_in_lifestyle.html', prediction=None)


@app.route('/templates/medications.html')
def medications():
    return render_template('medications.html', prediction=None)


@app.route('/predict', methods=['POST'])
def predict():
    try:

        age = int(request.form['age'])
        sex = int(request.form['sex'])
        cpt = int(request.form['cpt'])
        trestbps = int(request.form['trestbps'])
        chol = int(request.form['chol'])
        fbs = int(request.form['fbs'])
        restecg = int(request.form['restecg'])
        thalach = int(request.form['thalach'])
        exang = int(request.form['exang'])
        oldpeak = float(request.form['oldpeak'])
        slope = int(request.form['slope'])
        ca = int(request.form['ca'])
        thal = int(request.form['thal'])

        data = np.array([[age, sex, cpt, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])

        prediction = model.predict(data)

        return render_template('result.html', prediction=prediction[0])

    except Exception as e:
        # Handle exceptions
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)), debug=True)

