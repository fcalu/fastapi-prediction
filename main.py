from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from contextlib import asynccontextmanager

DATA_FOLDER = os.path.join(os.path.dirname(__file__), "data")
MODEL_OVER25_PATH = os.path.join(DATA_FOLDER, "model_over25.pkl")
MODEL_1X2_PATH = os.path.join(DATA_FOLDER, "model_1x2.pkl")

class PartidoRequest(BaseModel):
    liga: str
    equipo_local: str
    equipo_visitante: str

FEATURES = [
    "team_a_xg", "team_b_xg",
    "home_team_shots_on_target", "away_team_shots_on_target",
    "home_team_possession", "away_team_possession",
    "home_team_yellow_cards", "away_team_yellow_cards",
    "odds_ft_over25", "odds_btts_yes",
    "odds_ft_home_team_win", "odds_ft_draw", "odds_ft_away_team_win"
]

TARGET_OVER25 = "over_25"
TARGET_1X2 = "resultado_1x2"

def preparar_datos(df):
    df = df.dropna(subset=FEATURES + ["total_goal_count", "home_team_goal_count", "away_team_goal_count"])
    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=FEATURES)
    df[TARGET_OVER25] = (df["total_goal_count"] > 2.5).astype(int)
    df[TARGET_1X2] = df.apply(lambda x: 1 if x['home_team_goal_count'] > x['away_team_goal_count'] else (2 if x['away_team_goal_count'] > x['home_team_goal_count'] else 0), axis=1)
    X = df[FEATURES]
    y_over25 = df[TARGET_OVER25]
    y_1x2 = df[TARGET_1X2]
    return X, y_over25, y_1x2

def entrenar_y_guardar_modelos(df):
    X, y_over25, y_1x2 = preparar_datos(df)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model_over25 = RandomForestClassifier(n_estimators=100, random_state=42)
    model_over25.fit(X_scaled, y_over25)
    joblib.dump((model_over25, scaler), MODEL_OVER25_PATH)

    model_1x2 = RandomForestClassifier(n_estimators=100, random_state=42)
    model_1x2.fit(X_scaled, y_1x2)
    joblib.dump((model_1x2, scaler), MODEL_1X2_PATH)

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("📊 Cargando y unificando CSV...")
    try:
        df = pd.concat([
            pd.read_csv(os.path.join(DATA_FOLDER, f))
            for f in os.listdir(DATA_FOLDER) if f.endswith(".csv")
        ])
        print("🔍 Validando columnas y tipos...")
        entrenar_y_guardar_modelos(df)
        print("✅ Modelos entrenados correctamente")
    except Exception as e:
        print(f"❌ Error al entrenar modelos: {e}")
    yield

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/equipos")
def listar_equipos(liga: str):
    try:
        archivos = [f for f in os.listdir(DATA_FOLDER) if f.lower().replace(" ", "") == f"{liga}".lower().replace(" ", "") + ".csv"]
        if not archivos:
            raise HTTPException(status_code=404, detail="Liga no encontrada")
        path = os.path.join(DATA_FOLDER, archivos[0])
        df = pd.read_csv(path)

        equipos_locales = df["home_team_name"].dropna().unique().tolist()
        equipos_visitantes = df["away_team_name"].dropna().unique().tolist()
        equipos = sorted(set(equipos_locales + equipos_visitantes))
        return {"equipos": equipos}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"No se pudieron cargar los equipos: {e}")

@app.post("/predecir-over25")
def predecir_over25(data: PartidoRequest):
    try:
        archivos = [f for f in os.listdir(DATA_FOLDER) if f.lower().replace(" ", "") == f"{data.liga}".lower().replace(" ", "") + ".csv"]
        if not archivos:
            raise HTTPException(status_code=404, detail="Liga no encontrada")
        path = os.path.join(DATA_FOLDER, archivos[0])
        df = pd.read_csv(path)

        partido = df[
            (df["home_team_name"] == data.equipo_local) &
            (df["away_team_name"] == data.equipo_visitante)
        ].tail(1)

        if partido.empty:
            raise HTTPException(status_code=404, detail="Partido no encontrado")

        X_pred = partido[FEATURES]
        model, scaler = joblib.load(MODEL_OVER25_PATH)
        proba = model.predict_proba(scaler.transform(X_pred))[0][1]
        return {"probabilidad_over25": round(proba * 100, 2)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predecir-1x2")
def predecir_1x2(data: PartidoRequest):
    try:
        archivos = [f for f in os.listdir(DATA_FOLDER) if f.lower().replace(" ", "") == f"{data.liga}".lower().replace(" ", "") + ".csv"]
        if not archivos:
            raise HTTPException(status_code=404, detail="Liga no encontrada")
        path = os.path.join(DATA_FOLDER, archivos[0])
        df = pd.read_csv(path)

        partido = df[
            (df["home_team_name"] == data.equipo_local) &
            (df["away_team_name"] == data.equipo_visitante)
        ].tail(1)

        if partido.empty:
            raise HTTPException(status_code=404, detail="Partido no encontrado")

        X_pred = partido[FEATURES]
        model, scaler = joblib.load(MODEL_1X2_PATH)
        probs = model.predict_proba(scaler.transform(X_pred))[0]
        return {
            "1_local": round(probs[1] * 100, 2),
            "X_empate": round(probs[0] * 100, 2),
            "2_visitante": round(probs[2] * 100, 2)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ligas")
def listar_ligas():
    try:
        archivos = [f for f in os.listdir(DATA_FOLDER) if f.endswith(".csv")]
        ligas = [f.replace(".csv", "") for f in archivos]
        return {"ligas": ligas}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"No se pudieron cargar las ligas: {e}")

@app.post("/analisis-avanzado")
def analisis_avanzado(data: PartidoRequest):
    try:
        archivos = [f for f in os.listdir(DATA_FOLDER) if f.lower().replace(" ", "") == f"{data.liga}".lower().replace(" ", "") + ".csv"]
        if not archivos:
            raise HTTPException(status_code=404, detail="Liga no encontrada")
        path = os.path.join(DATA_FOLDER, archivos[0])
        df = pd.read_csv(path)

        historial_local = df[(df['home_team_name'] == data.equipo_local)]
        historial_visitante = df[(df['away_team_name'] == data.equipo_visitante)]
        h2h = df[(df['home_team_name'] == data.equipo_local) & (df['away_team_name'] == data.equipo_visitante)]

        goles_local = historial_local['home_team_goal_count'].mean()
        goles_visitante = historial_visitante['away_team_goal_count'].mean()
        xg_local = historial_local['team_a_xg'].mean()
        xg_visitante = historial_visitante['team_b_xg'].mean()

        model_1x2, scaler_1x2 = joblib.load(MODEL_1X2_PATH)
        model_over25, scaler_over25 = joblib.load(MODEL_OVER25_PATH)

        partido = df[(df['home_team_name'] == data.equipo_local) & (df['away_team_name'] == data.equipo_visitante)].tail(1)
        if partido.empty:
            raise HTTPException(status_code=404, detail="Partido no encontrado")

        X_pred = partido[FEATURES]
        proba_1x2 = model_1x2.predict_proba(scaler_1x2.transform(X_pred))[0]
        proba_over = model_over25.predict_proba(scaler_over25.transform(X_pred))[0][1]

        cuotas = {
            '1': partido['odds_ft_home_team_win'].values[0],
            'X': partido['odds_ft_draw'].values[0],
            '2': partido['odds_ft_away_team_win'].values[0]
        }
        prob_implicitas = {k: 100 / v for k, v in cuotas.items()}
        suma = sum(prob_implicitas.values())
        prob_implicitas = {k: v * 100 / suma for k, v in prob_implicitas.items()}

        value_bets = []
        if proba_1x2[1] * 100 > prob_implicitas['1']: value_bets.append("Local")
        if proba_1x2[0] * 100 > prob_implicitas['X']: value_bets.append("Empate")
        if proba_1x2[2] * 100 > prob_implicitas['2']: value_bets.append("Visitante")

        analisis = []
        if goles_local > 1.5: analisis.append(f"El equipo local promedia {goles_local:.2f} goles como local.")
        if goles_visitante < 1: analisis.append(f"El visitante tiene un promedio bajo: {goles_visitante:.2f} goles fuera de casa.")
        if len(h2h) >= 3 and h2h['total_goal_count'].mean() > 2.5:
            analisis.append("Históricamente, estos equipos superan los 2.5 goles por partido.")

        return {
            "probabilidades": {
                "local": round(proba_1x2[1] * 100, 2),
                "empate": round(proba_1x2[0] * 100, 2),
                "visitante": round(proba_1x2[2] * 100, 2),
                "over_25": round(proba_over * 100, 2)
            },
            "value_bet_detectado": value_bets,
            "recomendacion": value_bets[0] if value_bets else "Ninguna clara",
            "analisis": analisis
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en análisis avanzado: {str(e)}")
