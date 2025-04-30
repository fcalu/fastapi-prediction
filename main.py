from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import os
import numpy as np
import logging
import openai

from dotenv import load_dotenv
dotenv_path = os.path.join(os.path.dirname(__file__), "data", "api.env")
load_dotenv(dotenv_path=dotenv_path)


openai.api_key = os.getenv("OPENAI_API_KEY")

def obtener_prediccion_chatgpt(prompt: str) -> str:
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Eres un experto en análisis de partidos de fútbol."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error en IA: {str(e)}"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_FOLDER = os.path.join(os.path.dirname(__file__), 'data')
logging.basicConfig(level=logging.INFO)

class PartidoRequest(BaseModel):
    liga: str
    equipo_local: str
    equipo_visitante: str

def promedio_seguro(lista):
    return float(np.mean(lista)) if lista else 1.0

def calcular_metricas(hist, equipo):
    goles, xg, tiros_arco, posesion, tarjetas = [], [], [], [], []
    for _, row in hist.iterrows():
        if row['home_team_name'] == equipo:
            goles.append(row['home_team_goal_count'])
            xg.append(row['team_a_xg'])
            tiros_arco.append(row['home_team_shots_on_target'])
            posesion.append(row['home_team_possession'])
            tarjetas.append(row['home_team_yellow_cards'] + row['home_team_red_cards'])
        elif row['away_team_name'] == equipo:
            goles.append(row['away_team_goal_count'])
            xg.append(row['team_b_xg'])
            tiros_arco.append(row['away_team_shots_on_target'])
            posesion.append(row['away_team_possession'])
            tarjetas.append(row['away_team_yellow_cards'] + row['away_team_red_cards'])
    return {
        "goles_favor": promedio_seguro(goles),
        "xg": promedio_seguro(xg),
        "tiros_arco": promedio_seguro(tiros_arco),
        "posesion": promedio_seguro(posesion),
        "tarjetas": promedio_seguro(tarjetas)
    }

def calcular_puntuacion(stats):
    return (
        stats['goles_favor'] * 5 +
        stats['xg'] * 4 +
        stats['tiros_arco'] * 1.5 +
        stats['posesion'] * 0.2 -
        stats['tarjetas'] * 2
    )

@app.get("/ligas")
def listar_ligas():
    ligas = [f.replace('.csv', '') for f in os.listdir(DATA_FOLDER) if f.endswith('.csv')]
    return {"ligas": ligas}

@app.get("/equipos")
def listar_equipos(liga: str):
    try:
        path = os.path.join(DATA_FOLDER, f"{liga}.csv")
        df = pd.read_csv(path, skipinitialspace=True)

        if 'home_team_name' not in df.columns or 'away_team_name' not in df.columns:
            return {"error": "No se encontraron columnas 'home_team_name' o 'away_team_name'"}

        equipos = sorted(set(df['home_team_name'].dropna()).union(set(df['away_team_name'].dropna())))
        return {"equipos": equipos}
    except Exception as e:
        return {"error": str(e)}

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/predecir")
def predecir_partido(data: PartidoRequest, simulaciones: int = 1000):
    try:
        logging.info(f"Predicción solicitada para: {data}")
        path = os.path.join(DATA_FOLDER, f"{data.liga}.csv")
        df = pd.read_csv(path, skipinitialspace=True)

        required_cols = [
            'home_team_name', 'away_team_name',
            'home_team_goal_count', 'away_team_goal_count',
            'team_a_xg', 'team_b_xg',
            'home_team_shots_on_target', 'away_team_shots_on_target',
            'home_team_possession', 'away_team_possession',
            'home_team_yellow_cards', 'away_team_yellow_cards',
            'home_team_red_cards', 'away_team_red_cards'
        ]

        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            return {"error": f"Faltan columnas: {missing}"}

        df = df.dropna(subset=required_cols)

        local_total = pd.concat([df[df['home_team_name'] == data.equipo_local], df[df['away_team_name'] == data.equipo_local]])
        visitante_total = pd.concat([df[df['home_team_name'] == data.equipo_visitante], df[df['away_team_name'] == data.equipo_visitante]])

        if local_total.empty or visitante_total.empty:
            return {
                "probabilidades": {"local": 33.33, "empate": 33.33, "visitante": 33.33},
                "prediccion_final": "No hay suficiente historial para generar una predicción confiable"
            }

        stats_local = calcular_metricas(local_total, data.equipo_local)
        stats_visitante = calcular_metricas(visitante_total, data.equipo_visitante)

        puntuacion_local = calcular_puntuacion(stats_local)
        puntuacion_visitante = calcular_puntuacion(stats_visitante)

        local_wins = 0
        draws = 0
        visitante_wins = 0

        for _ in range(simulaciones):
            rl = np.random.normal(loc=puntuacion_local, scale=3)
            rv = np.random.normal(loc=puntuacion_visitante, scale=3)
            if rl > rv + 1.5:
                local_wins += 1
            elif rv > rl + 1.5:
                visitante_wins += 1
            else:
                draws += 1

        prob_local = round((local_wins / simulaciones) * 100, 2)
        prob_empate = round((draws / simulaciones) * 100, 2)
        prob_visitante = round((visitante_wins / simulaciones) * 100, 2)

        # Prompt para ChatGPT
        prompt = f"""
Eres un experto en predicción de partidos de fútbol usando inteligencia artificial avanzada y lógica basada en datos históricos.

Analiza el siguiente partido:

📌 EQUIPO LOCAL: {data.equipo_local}
- Goles promedio: {stats_local['goles_favor']:.2f}
- xG promedio: {stats_local['xg']:.2f}
- Tiros al arco: {stats_local['tiros_arco']:.2f}
- Posesión promedio: {stats_local['posesion']:.2f}%
- Tarjetas promedio: {stats_local['tarjetas']:.2f}

📌 EQUIPO VISITANTE: {data.equipo_visitante}
- Goles promedio: {stats_visitante['goles_favor']:.2f}
- xG promedio: {stats_visitante['xg']:.2f}
- Tiros al arco: {stats_visitante['tiros_arco']:.2f}
- Posesión promedio: {stats_visitante['posesion']:.2f}%
- Tarjetas promedio: {stats_visitante['tarjetas']:.2f}

📈 Con base en estos datos y el historial de ambos equipos:

1. ¿Quién ganará el partido? Responde únicamente con:
   - '1' si gana el equipo local,
   - '2' si gana el visitante,
   - 'X' si será empate.
   Incluye una **breve razón técnica basada en los datos**.

2. ¿Habrá más de 2.5 goles en el partido? Responde 'Sí' o 'No' con justificación basada en xG y goles históricos.

3. ¿Cuál equipo es más probable que reciba más tarjetas? Responde con el nombre y explica por qué.

4. ¿Qué equipo tendrá más tiros de esquina en promedio? Responde con el nombre y justifica con los datos.

Devuelve las 4 respuestas como un **análisis profesional breve**. Sé concreto, objetivo y no inventes datos que no están presentes.
"""


        prediccion_chatgpt = obtener_prediccion_chatgpt(prompt)

        return {
            "historial_partidos": len(df[(df['home_team_name'] == data.equipo_local) & (df['away_team_name'] == data.equipo_visitante)]),
            "metricas_local": stats_local,
            "metricas_visitante": stats_visitante,
            "puntuacion_ia_local": round(puntuacion_local, 2),
            "puntuacion_ia_visitante": round(puntuacion_visitante, 2),
            "probabilidades": {
                "local": prob_local,
                "empate": prob_empate,
                "visitante": prob_visitante
            },
            "prediccion_chatgpt": prediccion_chatgpt
        }

    except Exception as e:
        logging.error(f"Error al predecir: {str(e)}")
        return {
            "error": str(e),
            "probabilidades": {
                "local": 33.33,
                "empate": 33.33,
                "visitante": 33.33
            },
            "prediccion_final": "Error al procesar los datos, intenta con otra liga o equipos"
        }
