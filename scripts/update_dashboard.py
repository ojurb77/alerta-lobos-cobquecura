from __future__ import annotations

import json
import math
import os
import sys
import tempfile
from datetime import datetime, timedelta, timezone, date
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
DOCS_DATA = ROOT / "docs" / "data"
STATUS_PATH = DOCS_DATA / "status.json"
HISTORY_PATH = DOCS_DATA / "history.json"

DATASET_ID = "cmems_mod_glo_wav_anfc_0.083deg_PT3H-i"
VARIABLE = "VTPK"
MIN_LON, MAX_LON = -73.05, -72.75
MIN_LAT, MAX_LAT = -36.30, -35.95
MIN_VALID_TIMESTEPS = 6  # mínimo de 6 de los 8 registros de 3 h por día

# Modelo logístico provisional LOYO. Objetivo histórico: >=5 crías/día.
B0 = -6.9431299378225955
B_SEASON_DAY = 0.10647496
B_TP = 0.43959415
THRESHOLD_WATCH = 0.45
THRESHOLD_ALERT = 0.50
SEASON_START = (1, 16)
SEASON_END = (2, 15)


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def season_day(target: date) -> int | None:
    start = date(target.year, *SEASON_START)
    end = date(target.year, *SEASON_END)
    if start <= target <= end:
        return (target - start).days
    return None


def probability(target: date, tp_previous_day: float) -> float:
    sd = season_day(target)
    if sd is None:
        raise ValueError("Fecha fuera de la ventana estacional validada.")
    z = B0 + B_SEASON_DAY * sd + B_TP * tp_previous_day
    # Estabilidad numérica
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


def level(p: float) -> str:
    if p >= THRESHOLD_ALERT:
        return "Alerta experimental (>=50%)"
    if p >= THRESHOLD_WATCH:
        return "Vigilancia reforzada experimental (>=45%)"
    return "Bajo los umbrales experimentales"


def fetch_forecast(start: date, end: date) -> Path:
    username = os.environ.get("COPERNICUSMARINE_SERVICE_USERNAME", "").strip()
    password = os.environ.get("COPERNICUSMARINE_SERVICE_PASSWORD", "").strip()
    if not username or not password:
        raise RuntimeError(
            "Faltan los secretos de Copernicus Marine en la configuración del repositorio."
        )

    import copernicusmarine

    tmp = Path(tempfile.mkdtemp(prefix="cobquecura_"))
    output_name = "forecast_cobquecura.nc"
    copernicusmarine.subset(
        dataset_id=DATASET_ID,
        username=username,
        password=password,
        variables=[VARIABLE],
        minimum_longitude=MIN_LON,
        maximum_longitude=MAX_LON,
        minimum_latitude=MIN_LAT,
        maximum_latitude=MAX_LAT,
        start_datetime=start.isoformat(),
        end_datetime=end.isoformat(),
        output_directory=str(tmp),
        output_filename=output_name,
        overwrite=True,
    )
    result = tmp / output_name
    if not result.exists():
        # Algunas versiones de Toolbox pueden añadir sufijos al nombre.
        matches = list(tmp.rglob("*.nc"))
        if not matches:
            raise RuntimeError("Copernicus no produjo un archivo NetCDF.")
        result = matches[0]
    return result


def daily_tp_from_netcdf(path: Path) -> tuple[dict[str, float], dict[str, int]]:
    with xr.open_dataset(path) as ds:
        if VARIABLE not in ds:
            raise RuntimeError(f"El archivo no contiene la variable {VARIABLE}.")
        if "time" not in ds[VARIABLE].dims:
            raise RuntimeError("La variable de período no tiene dimensión temporal.")

        da = ds[VARIABLE].astype("float64")
        non_time_dims = [d for d in da.dims if d != "time"]
        # Promedio espacial por cada hora de validez, evitando que las horas con más
        # celdas válidas pesen más que las demás.
        per_time = da.mean(dim=non_time_dims, skipna=True) if non_time_dims else da
        times = pd.to_datetime(ds["time"].values, utc=True)
        values = np.asarray(per_time.values, dtype=float).reshape(-1)

    frame = pd.DataFrame({"time": times, "tp": values})
    frame = frame[np.isfinite(frame["tp"])]
    if frame.empty:
        raise RuntimeError("La variable VTPK no contiene valores válidos en el área seleccionada.")
    frame["date"] = frame["time"].dt.strftime("%Y-%m-%d")
    daily = frame.groupby("date")["tp"].agg(["mean", "count"])
    means = {str(k): float(v["mean"]) for k, v in daily.iterrows()}
    counts = {str(k): int(v["count"]) for k, v in daily.iterrows()}
    return means, counts


def read_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    run_time = now_utc()
    today = run_time.date()
    status = {
        "schema_version": 1,
        "run_status": "error",
        "run_timestamp_utc": run_time.isoformat(),
        "data_source": "Copernicus Marine Global Ocean Waves Analysis and Forecast",
        "dataset_id": DATASET_ID,
        "variable": VARIABLE,
        "season_active_today": season_day(today) is not None,
        "message": "",
        "predictions": [],
        "daily_wave_data": [],
        "thresholds": {
            "watch": THRESHOLD_WATCH,
            "alert": THRESHOLD_ALERT
        },
        "model": {
            "target": ">=5 crías varadas en un día",
            "equation": "p = logistic(-6.94312994 + 0.10647496*season_day + 0.43959415*Tp_previous_day)",
            "validation": "LOYO histórico 2022–2026; AUC ROC agregada aproximada 0.789; uso con pronóstico aún no validado",
            "operational_status": "experimental_no_official_alert"
        },
        "limitations": [
            "Modelo experimental; no equivale a una alerta oficial.",
            "El rendimiento con pronósticos operacionales todavía debe validarse.",
            "VTPK es el período de pico; la equivalencia de agregación con la serie histórica debe confirmarse.",
            "Si faltan datos, no debe interpretarse la ausencia de predicción como riesgo bajo."
        ]
    }

    try:
        # Recuperar hoy y el horizonte siguiente. La predicción para una fecha usa
        # el período del día calendario anterior.
        start = today
        end = today + timedelta(days=9)
        nc_path = fetch_forecast(start, end)
        means, counts = daily_tp_from_netcdf(nc_path)

        for day_str in sorted(means):
            status["daily_wave_data"].append({
                "date_utc": day_str,
                "tp_peak_mean_s": round(means[day_str], 3),
                "valid_3h_steps": counts[day_str],
                "coverage_ok": counts[day_str] >= MIN_VALID_TIMESTEPS
            })

        # Objetivos de mañana hasta el final del horizonte, solo dentro de la
        # ventana estacional que el modelo estudió.
        predictions = []
        for offset in range(1, 10):
            target = today + timedelta(days=offset)
            sd = season_day(target)
            if sd is None:
                continue
            predictor_day = (target - timedelta(days=1)).isoformat()
            if predictor_day not in means or counts.get(predictor_day, 0) < MIN_VALID_TIMESTEPS:
                predictions.append({
                    "target_date": target.isoformat(),
                    "status": "datos_insuficientes",
                    "message": f"No hay cobertura suficiente de VTPK para {predictor_day}.",
                    "probability": None,
                    "level": "Sin cálculo"
                })
                continue
            p = probability(target, means[predictor_day])
            predictions.append({
                "target_date": target.isoformat(),
                "season_day": sd,
                "predictor_date": predictor_day,
                "tp_previous_day_s": round(means[predictor_day], 3),
                "valid_3h_steps": counts[predictor_day],
                "probability": round(p, 4),
                "probability_percent": round(p * 100, 1),
                "level": level(p),
                "status": "experimental"
            })

        status["predictions"] = predictions
        if predictions and any(p.get("probability") is not None for p in predictions):
            status["run_status"] = "ok"
            status["message"] = "Pronóstico procesado. Las probabilidades son experimentales y requieren validación operacional."
        elif any(p.get("status") == "datos_insuficientes" for p in predictions):
            status["run_status"] = "data_incomplete"
            status["message"] = "Hay fechas de la ventana estudiada sin cobertura suficiente. No interpretar como riesgo bajo."
        else:
            status["run_status"] = "ok_out_of_season"
            status["message"] = "No hay fechas objetivo dentro de la ventana estacional (16 de enero–15 de febrero) en el horizonte actual."
    except Exception as exc:
        # No publicar detalles internos/credenciales en la página pública.
        status["run_status"] = "error"
        status["message"] = "No fue posible actualizar los datos. Revisa el registro privado de GitHub Actions."
        status["technical_error_type"] = type(exc).__name__
        print(f"ERROR de actualización: {type(exc).__name__}: {exc}", file=sys.stderr)

    write_json(STATUS_PATH, status)

    history = read_json(HISTORY_PATH, [])
    if not isinstance(history, list):
        history = []
    history.append({
        "run_timestamp_utc": status["run_timestamp_utc"],
        "run_status": status["run_status"],
        "message": status["message"],
        "predictions": status.get("predictions", [])
    })
    write_json(HISTORY_PATH, history[-400:])

    print(f"Estado publicado: {status['run_status']}")
    print(status["message"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
