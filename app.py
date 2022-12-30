from flask import Flask,request,jsonify
from sklearn.ensemble import RandomForestClassifier
import pickle
import numpy as np
model = pickle.load(open('model.pkl','rb'))
# form ke through jabb bhi koi data aata hai aapko usko request ke through handle krna hota hai

app = Flask(__name__)

@app.route('/')
def home():
    return "Hello World"

## post ke through aap without url ke input le sakte ho
## we can handle two kind of request here get and post
## get mei aap url ke through input lete ho post mie without url lete ho
@app.route('/predict',methods=['POST'])
def predict():
    cgpa = request.form.get('cgpa')
    iq = request.form.get('iq')
    profile_score = request.form.get('profile_score')

    input_query = np.array([[cgpa,iq,profile_score]])
    result = model.predict(input_query)[0]
    #result = {'cgpa':cgpa,'iq':iq,'profile_score':profile_score}
    return jsonify({'placement':str(result)})



if __name__=="__main__":
    app.run(debug=True)