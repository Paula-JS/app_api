import google.generativeai as genai
import datetime


def get_ts():
    timestamp =datetime.datetime.now().isoformat()
    return timestamp[0:19]

def get_prompt(inputs, output):
    """
    Genera un prompt para el modelo generativo basado en los datos de entrada y la predicción del modelo.
    """
    mapita = {0: "no superviviente",
            1: "superviviente"}
    pclass, sex, age = inputs
    prediction = mapita[output]

    prompt = f"""
    Hola gemini! estoy haciendo una API de predicción de supervivencia o no con el dataset del Titanic.
    He usado 3 features: Pclass, Age y Sex. Lo que necesito es pasarte los inputs y la predicción del modelo,
    y que me generes un texto especulando, a través de los inputs dados y la predicción del modelo, los motivos por los cuales el modelo ha hecho esa predicción
    y si tiene sentido o no la misma, dado el contexto. Quiero que lo escribas de forma muy narrada, como si fuera una historia de aventuras.
    Pero quiero un texto conciso, entre 100 y 500 palabras máximo.
    Se creativo, mojate.
    Importante 1: el formato de salida ha de ser ÚNICA Y EXCLUSIVAMENTE el texto narrado. No me des saludos, metadatos, ni nada,
    aparte del breve texto.
    Importante 2: Además, omite todo tipo de formato enriquecido (markdown, HTML, etc...). Dame solo texto plano.
    Importante 3: Para que el texto no resulte muy horizontal, incluye numerosos saltos de línea.
    EL CONTEXTO ES EL SIGUIENTE: 
    Inputs:
    Pclass = {pclass}
    Sex = {sex} (siendo 0 Male y 1 Female)
    Age = {age} (edad en años)
    Predicción = {prediction}
    Tu respuesta aquí:
    """
    return prompt

def generar_texto(model, prompt, top_k=40, stop_sequences=None, temperature=0.7, top_p=1.0, max_output_tokens=512):
    """
    Genera un texto narrativo basado en el prompt utilizando el modelo generativo.
    """
    response = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            top_p=top_p,
            top_k=top_k,
            stop_sequences=stop_sequences
        )
    )
    return response.text