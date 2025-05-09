import os
from flask import Flask, render_template, request
from joblib import load
import numpy as np

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("About.html")

@app.route("/kidney")
def kidney():
    return render_template("Kidney.html")

@app.route("/servic")
def servic():
    return render_template("servic.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route("/kidney_predict", methods=["POST"])
def kidney_predict():
    model = load('Models_joblib/kidney_Model.joblib')

    if request.method == 'POST':
        white_blood_cell_count = float(request.form.get('white_blood_cell_count'))
        blood_glucose_random = float(request.form.get('blood_glucose_random'))
        blood_urea = float(request.form.get('blood_urea'))
        serum_creatinine = float(request.form.get('serum_creatinine'))
        packed_cell_volume = float(request.form.get('packed_cell_volume'))
        albumin = float(request.form.get('albumin'))
        haemoglobin = float(request.form.get('haemoglobin'))
        age = float(request.form.get('age'))
        sugar = float(request.form.get('sugar'))
        hypertension = float(request.form.get('hypertension'))

        features = np.array([[white_blood_cell_count, blood_glucose_random, blood_urea,
                              serum_creatinine, packed_cell_volume, albumin,
                              haemoglobin, age, sugar, hypertension]])

        results_predict = int(model.predict(features))

        if results_predict:
            prediction = "There are chances of Chronic Kidney Disease! Consult your doctor soon."
        else:
            prediction = "No dangerous symptoms of Chronic Kidney Disease. But a consultation won't hurt."

    return render_template("result.html", prediction_text=prediction)

@app.route("/diabetes")
def diabetes():
    return render_template("Diabetes.html")

@app.route('/diabetes_predict', methods=['POST'])
def diabetes_predict():
    clf = load('Models_joblib/model.pkl')

    if request.method == 'POST':
        age = float(request.form.get('Age'))
        gender = float(request.form.get('Gender'))
        polyuria = float(request.form.get('Polyuria'))
        polydipsia = float(request.form.get('Polydipsia'))
        weight_loss = float(request.form.get('sudden weight loss'))
        weakness = float(request.form.get('weakness'))
        polyphagia = float(request.form.get('Polyphagia'))
        thrush = float(request.form.get('Genital thrush'))
        blurring = float(request.form.get('visual blurring'))
        itching = float(request.form.get('Itching'))
        irritability = float(request.form.get('Irritability'))
        healing = float(request.form.get('delayed healing'))
        paresis = float(request.form.get('partial paresis'))
        stiffness = float(request.form.get('muscle stiffness'))
        alopecia = float(request.form.get('Alopecia'))
        obesity = float(request.form.get('Obesity'))

        features = np.array([[age, gender, polyuria, polydipsia, weight_loss, weakness, polyphagia,
                              thrush, blurring, itching, irritability, healing, paresis, stiffness,
                              alopecia, obesity]])
        
        result = clf.predict(features)

        if result[0] == 1:
            prediction = "There are chances of Diabetes! Consult your doctor soon."
        else:
            prediction = "No signs of Diabetes. You're good, but regular checkups are always wise!"

    return render_template("result.html", prediction_text=prediction)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # fallback to 5000 for local dev
    app.run(host='0.0.0.0', port=port, debug=True)
