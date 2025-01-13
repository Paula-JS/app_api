import pandas as pd
from sqlalchemy import create_engine
from flask import Flask, jsonify, request, render_template
import json
import datetime
import pickle
import sklearn
import psycopg2
import io
import base64

app = Flask(__name__)

conexion = "postgresql://postgres:titanicapi@104.199.11.65:5432/postgres"
engine = create_engine(conexion)


def get_ts():
    import datetime
    timestamp =datetime.datetime.now().isoformat()
    return timestamp[0:19]

@app.route("/", methods=["GET"])
def formulario():
    return render_template("formulario.html")


@app.route("/predict", methods=["POST"])
def predict():
    pclass = int(request.form().get("pclass"))
    sex = int(request.form().get("sex"))
    age = int(request.form().get("age"))
    
    inputs = [pclass, sex, age]
    
    with open("titanic_model.pkl", "rb") as f:
        modelito = pickle.load(f)
    
    outputs = modelito.predict([inputs])[0]
    timestamp = get_ts()
    
    logs = pd.DataFrame({"inputs": [inputs],
              "predictions": [outputs],
              "timestamp":[timestamp]})
    logs.to_sql("predictions", con=engine, index=False, if_exists="append")
    
    import matplotlib.pyplot as plt 
    fig =plt.figure()
    logs_leidos.predictions.value_counts().plot(kind="bar")
    plt.title(f"PREDICTIONS UP TO: {logs_leidos.timestamp.max()}")
    
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    plt.close(fig)
    
    img_base64 = base64.b64decode(buffer.getvalue()).decode("utf-8")
    
    return render_template("resultado.html", prediccion=outputs, grafica=img_base64)
    

@app.route("/results", methods=["GET"])
def results():
    logs_leidos = pd.read_sql("""SELECT * FROM predictions""", con=engine)
    return json.loads(logs_leidos.to_json(orient="records"))



if __name__ == "__main__":
    app.run(debug=True)