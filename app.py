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
import os 
import google.generativeai as genai
from utils import get_prompt, get_ts, generar_texto

load_dotenv()

app = Flask(__name__)

conexion = os.environ["CONEXION"]
engine = create_engine(conexion)

with open("titanic_model.pkl", "rb") as f:
        modelito = pickle.load(f)

@app.route('/', methods=['GET'])
def home():
    return render_template("formulario.html")


@app.route("/predict", methods=["POST"])
def predict():
    pclass = int(request.form.get("pclass"))
    sex = int(request.form.get("sex"))
    age = int(request.form.get("age"))
    
    inputs = pclass, sex, age
    
    outputs = modelito.predict([[pclass, sex, age]])
    output = outputs[0]
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
    
    #GENERAR TEXTO IA
    
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"]) 
    model = genai.GenerativeModel("gemini-2.0-flash-exp")
    prompt = get_prompt(inputs, output)
    generacion = generar_texto(model, prompt, temperature=0.7, top_p=1.0, top_k=40, max_output_tokens=512)
    
    
    return render_template("resultado.html", prediccion=int(outputs[0]), grafica=img_base64, texto_ia=generacion)
    

@app.route("/results", methods=["GET"])
def results():
    logs_leidos = pd.read_sql("""SELECT * FROM predictions""", con=engine)
    return json.loads(logs_leidos.to_json(orient="records"))


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)