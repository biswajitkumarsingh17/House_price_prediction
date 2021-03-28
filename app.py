from flask import Flask, render_template, request
import requests
import flask
import numpy as np
import pickle


app = Flask(__name__)

model = pickle.load(open('lasso_model.pkl', 'rb'))


@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        MiscVal = request.form['MiscVal'].replace(" ", "")
        BsmtHalfBath = request.form['BsmtHalfBath'].replace(" ", "")
        LowQualFinSF = request.form['LowQualFinSF'].replace(" ", "")
        BsmtFullBath = request.form['BsmtFullBath'].replace(" ", "")
        HalfBath = request.form['HalfBath'].replace(" ", "")

        MiscVal, BsmtHalfBath, LowQualFinSF, BsmtFullBath, HalfBath  = int(MiscVal), int(BsmtHalfBath), int(LowQualFinSF), int(BsmtFullBath), int(HalfBath)

        def rfr_predict(MiscVal, BsmtHalfBath, LowQualFinSF,BsmtFullBath,HalfBath):
            x = np.zeros(120)
            x[0] = MiscVal
            x[1] = BsmtHalfBath
            x[2] = LowQualFinSF
            x[4] = BsmtFullBath
            x[5] = HalfBath

            return model.predict([x])[0]

        result = rfr_predict(MiscVal, BsmtHalfBath, LowQualFinSF,BsmtFullBath,HalfBath)

        print(result)

        return render_template('results.html', result=result)
    else:
        return render_template('index.html')
        


if __name__ == "__main__":
    app.run(debug=True)