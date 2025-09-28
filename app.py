import pickle
from flask import Flask ,request, jsonify, render_template
import numpy as np
app=Flask(__name__)
model =pickle.load(open('model.pkl','rb'))
scaler = pickle.load(open('scaling_new.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')
@app.route('/predict_api',methods=['POST'])
def predict_api():
    data = [float(x) for x in request.form.values()]
    final_user_value=np.array(data).reshape(1,-1)
    transformed_user=scaler.transform(final_user_value)
    model_prediction = model.predict(transformed_user)
    return jsonify(model_prediction[0])

if __name__=="__main__":
    app.run(debug=True)
