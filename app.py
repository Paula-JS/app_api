import pandas as pd
from sqlalchemy import create_engine
from flask import Flask, jsonify, request, render_template
import json
import datetime
import pickle
import sklearn
import psycopg2
import io
from io import BytesIO
import base64
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt 
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

conexion = os.environ["CONEXION"]
engine = create_engine(conexion)

with open("titanic_model.pkl", "rb") as f:
        modelito = pickle.load(f)

def get_ts():
    timestamp =datetime.datetime.now().isoformat()
    return timestamp[0:19]


@app.route('/', methods=['GET'])
def home():
    return render_template("formulario.html")


@app.route("/predict", methods=["POST"])
def predict():
    pclass = int(request.form.get("pclass"))
    sex = int(request.form.get("sex"))
    age = int(request.form.get("age"))
    
    
    outputs = modelito.predict([[pclass, sex, age]])
    timestamp = get_ts()

    
    logs = pd.DataFrame({"pclass": [pclass],
                         "sex": [sex],
                         "age": [age],
                         "prediccion": [int(outputs[0])],
                         "timestamp":[timestamp]})
    
    logs.to_sql("predictions", con=engine, index=False, if_exists="append")
    
    
    read_predicciones = pd.read_sql("SELECT * FROM predictions", con=engine)
    fig =plt.figure()
    read_predicciones.prediccion.value_counts().plot(kind="bar")
    plt.title("Predicciones totales")
    
    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    plt.close(fig)
    
    img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    
    return render_template("resultado.html", prediccion=int(outputs[0]), grafica=img_base64)
    

@app.route("/results", methods=["GET"])
def results():
    logs_leidos = pd.read_sql("""SELECT * FROM predictions""", con=engine)
    return json.loads(logs_leidos.to_json(orient="records"))


if __name__ == "__main__":
    app.run(debug=True)