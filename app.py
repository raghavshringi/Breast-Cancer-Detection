import pickle
import numpy as np
from flask import Flask, render_template, request, redirect

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('C:/Users/ragha/OneDrive/Desktop/breast cancer/model.pkl', 'rb'))

@app.route('/')
def home():
    return redirect('/predict')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the input values from the form
        input_features = [float(x) for x in request.form.to_dict().values()]
        # Convert the input into a numpy array
        input_array = np.array(input_features).reshape(1, -1)
        # Make predictions using the loaded model
        prediction = model.predict(input_array)
        # Display the prediction result on a new page
        return render_template('predict.html', prediction=prediction)
    else:
        # Handle GET request
        return render_template('predict.html')  # Replace 'predict.html' with the appropriate HTML template name

@app.route('/predictbreast', methods=['POST'])
def predictbreast():
    if request.method == "POST":
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(float, to_predict_list))
        result =0
        if len(to_predict_list) == 30:
            result = ValuePredictor(to_predict_list, 30)
        


    if int(result) == 1:
        prediction = "Patient is predicted to have breast cancer."
    else:
        prediction = "Patient is predicted to be cancer-free."

    return render_template("predictbreast.html", prediction_text=prediction)


def ValuePredictor(to_predict_list, size):
    to_predict = np.array(to_predict_list).reshape(1, size)
    loaded_model = pickle.load(open('C:/Users/ragha/OneDrive/Desktop/breast cancer/model.pkl', 'rb'))
    result = loaded_model.predict(to_predict)
    return result[0]

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
