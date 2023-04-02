#import Flask class from flask library
from flask import Flask, render_template, request
import numpy as np
import pickle


#create an instance of the class named app
app = Flask(__name__, template_folder = 'C:/Users/samdk/PycharmProjects/webapp/template_folder')

model = pickle.load(open('models/model.pk1', 'rb'))

#use route to tell flask what URL to use and what method to use, get: a message is sent and server returns data
@app.route('/')

def home():
    return render_template('index.html')

#POST sends HTML form data to the server
@app.route('/predict', methods=['POST'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    features = [np.array(int_features)]
    prediction = model.predict(features)

    output = round(prediction[0], 2)
    return render_template('index.html', prediction_text = 'Percent with heart disease:\t {}'.format(output))

  
#run from   http://localhost:3000/
if __name__ == '__main__':
    app.run(port=3000, debug=True)
