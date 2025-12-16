# daedalus/transport_engine.py
from __future__ import annotations

import os
import json
import math
import zipfile
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests  

import pandas as pd
import geopandas as gpd
from shapely import wkt
from shapely.geometry import Point
import networkx as nx

from django.conf import settings

import hashlib
import time

import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt

# ============================================================
# Helpers (paths)
# ============================================================

def transport_media_dir() -> Path:
    """
    Base folder for all transport inputs/outputs inside MEDIA_ROOT.
    """
    return Path(settings.MEDIA_ROOT) / "transport"


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def transport_outputs_dir() -> Path:
    return ensure_dir(transport_media_dir() / "outputs")


def transport_gtfs_clean_dir() -> Path:
    return ensure_dir(transport_media_dir() / "gtfs_clean")


def transport_incidencias_path() -> Path:
    # incidencias.json living next to inputs/outputs
    return transport_media_dir() / "incidencias.json"


def transport_zonas_poligonos_path() -> Path:
    return transport_media_dir() / "zonas_transporte_poligonos.csv"

# ============================================================
# DEMAND / EMT paths
# ============================================================

def transport_inputs_dir() -> Path:
    return ensure_dir(transport_media_dir())

def transport_viajes_estudio_path() -> Path:
    return transport_media_dir() / "viajes_estudio.csv"

def transport_emt_mapa_path() -> Path:
    return transport_outputs_dir() / "emt_mapa.csv"

def transport_cache_dir() -> Path:
    return ensure_dir(transport_media_dir() / "cache")
# ============================================================
# 1) GTFS Processing
# ============================================================

class GTFSProcessor:
    """
    Provider-agnostic GTFS pipeline, with provider hooks.
    """

    def __init__(self, zip_path: Path, out_folder: Path):
        self.zip_path = Path(zip_path)
        self.folder = Path(out_folder)

    def extract_zip(self) -> None:
        ensure_dir(self.folder)
        with zipfile.ZipFile(self.zip_path, "r") as z:
            for member in z.infolist():
                if member.is_dir():
                    continue
                filename = os.path.basename(member.filename)
                if not filename:
                    continue
                dest = self.folder / filename
                with z.open(member) as src, open(dest, "wb") as out:
                    out.write(src.read())

    def process_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df.columns = df.columns.str.strip()
        return df

    def process_agency(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df[["agency_id", "agency_name"]].copy()
        df.loc[:, "agency_id"] = "CRTM"
        df.loc[:, "agency_name"] = "CRTM"
        return df

    def process_routes(self, df: pd.DataFrame) -> pd.DataFrame:
        if "agency_id" not in df.columns:
            df["agency_id"] = "CRTM"
        df = df[["route_id", "agency_id", "route_short_name", "route_long_name"]].copy()
        df.loc[:, "agency_id"] = "CRTM"
        return df

    def process_trips(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in ["route_id", "service_id", "trip_id", "shape_id"] if c in df.columns]
        return df[cols].copy()

    def process_stop_times(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[["trip_id", "arrival_time", "departure_time", "stop_id", "stop_sequence"]].copy()

    def process_stops(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = ["stop_id", "stop_name", "stop_lat", "stop_lon", "route_id", "service_id", "trip_id", "shape_id"]
        df = df[[c for c in cols if c in df.columns]].copy()
        return df

    def propagar_relaciones(
        self,
        routes: Optional[pd.DataFrame],
        trips: Optional[pd.DataFrame],
        stop_times: Optional[pd.DataFrame],
        stops: Optional[pd.DataFrame],
    ) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:

        if stops is not None and stop_times is not None:
            valid_stop_ids = set(stops["stop_id"])
            stop_times = stop_times[stop_times["stop_id"].isin(valid_stop_ids)].copy()

        if stop_times is not None and trips is not None:
            valid_trip_ids = set(stop_times["trip_id"])
            trips = trips[trips["trip_id"].isin(valid_trip_ids)].copy()

        if trips is not None and routes is not None:
            valid_route_ids = set(trips["route_id"])
            routes = routes[routes["route_id"].isin(valid_route_ids)].copy()

        return routes, trips, stop_times, stops

    def post_process_provider(self, routes, trips, stop_times, stops):
        return routes, trips, stop_times, stops

    def procesar(self) -> None:
        self.extract_zip()
        files = set(os.listdir(self.folder))

        if "agency.txt" in files:
            agency = self.process_columns(pd.read_csv(self.folder / "agency.txt"))
            agency = self.process_agency(agency)
            agency.to_csv(self.folder / "agency.txt", index=False)

        routes = None
        if "routes.txt" in files:
            routes = self.process_columns(pd.read_csv(self.folder / "routes.txt"))
            routes = self.process_routes(routes)

        trips = None
        if "trips.txt" in files:
            trips = self.process_columns(pd.read_csv(self.folder / "trips.txt"))
            trips = self.process_trips(trips)

        stop_times = None
        if "stop_times.txt" in files:
            stop_times = self.process_columns(pd.read_csv(self.folder / "stop_times.txt"))
            stop_times = self.process_stop_times(stop_times)

        stops = None
        if "stops.txt" in files:
            stops = self.process_columns(pd.read_csv(self.folder / "stops.txt"))
            stops = self.process_stops(stops)

        routes, trips, stop_times, stops = self.post_process_provider(routes, trips, stop_times, stops)

        if routes is not None:
            routes.to_csv(self.folder / "routes.txt", index=False)
        if trips is not None:
            trips.to_csv(self.folder / "trips.txt", index=False)
        if stop_times is not None:
            stop_times.to_csv(self.folder / "stop_times.txt", index=False)
        if stops is not None:
            stops.to_csv(self.folder / "stops.txt", index=False)


class CercaniasGTFSProcessor(GTFSProcessor):
    def post_process_provider(self, routes, trips, stop_times, stops):
        if trips is not None and "service_id" in trips.columns:
            trips = trips[trips["service_id"] == "1059M"]
        return self.propagar_relaciones(routes, trips, stop_times, stops)


class EMTGTFSProcessor(GTFSProcessor):
    def post_process_provider(self, routes, trips, stop_times, stops):
        if trips is not None and "service_id" in trips.columns:
            trips = trips[trips["service_id"] == "LA"]
        return self.propagar_relaciones(routes, trips, stop_times, stops)


class MetroGTFSProcessor(GTFSProcessor):
    def post_process_provider(self, routes, trips, stop_times, stops):
        if trips is not None and "service_id" in trips.columns:
            trips = trips[trips["service_id"] == "4_I14"].copy()

        freq_path = self.folder / "frequencies.txt"
        if freq_path.exists() and stop_times is not None and trips is not None:
            frequencies = self.process_columns(pd.read_csv(freq_path))
            stop_times, trips = self.expand_stop_times(stop_times, trips, frequencies)

        return self.propagar_relaciones(routes, trips, stop_times, stops)

    def str_to_timedelta(self, t: str) -> timedelta:
        h, m, s = map(int, t.split(":"))
        return timedelta(hours=h, minutes=m, seconds=s)

    def timedelta_to_str(self, td: timedelta) -> str:
        total = int(td.total_seconds())
        h = total // 3600
        m = (total % 3600) // 60
        s = total % 60
        return f"{h:02}:{m:02}:{s:02}"

    def expand_stop_times(self, stop_times: pd.DataFrame, trips: pd.DataFrame, frequencies: pd.DataFrame):
        new_stop_times: List[Dict[str, Any]] = []
        new_trips: List[Dict[str, Any]] = []
        counter = 0

        trip_to_route = dict(zip(trips["trip_id"], trips["route_id"]))
        trip_to_service = dict(zip(trips["trip_id"], trips["service_id"]))

        for _, row in frequencies.iterrows():
            base_trip = row["trip_id"]
            if base_trip not in trip_to_route:
                continue

            route_id_orig = trip_to_route[base_trip]
            service_id_orig = trip_to_service[base_trip]

            base_df = stop_times[stop_times["trip_id"] == base_trip]
            if base_df.empty:
                continue

            start = self.str_to_timedelta(row["start_time"])
            end = self.str_to_timedelta(row["end_time"])
            headway = timedelta(seconds=int(row["headway_secs"]))

            t = start
            while t <= end:
                counter += 1
                new_trip_id = f"{base_trip}__T{counter:05d}"

                new_trips.append({"route_id": route_id_orig, "service_id": service_id_orig, "trip_id": new_trip_id})

                base_start_arrival = self.str_to_timedelta(base_df.iloc[0]["arrival_time"])
                base_start_departure = self.str_to_timedelta(base_df.iloc[0]["departure_time"])

                for _, st in base_df.iterrows():
                    arr = self.str_to_timedelta(st["arrival_time"])
                    dep = self.str_to_timedelta(st["departure_time"])

                    new_stop_times.append(
                        {
                            "trip_id": new_trip_id,
                            "arrival_time": self.timedelta_to_str(t + (arr - base_start_arrival)),
                            "departure_time": self.timedelta_to_str(t + (dep - base_start_departure)),
                            "stop_id": st["stop_id"],
                            "stop_sequence": st["stop_sequence"],
                        }
                    )

                t += headway

        new_stop_times_df = pd.DataFrame(new_stop_times)
        if not new_stop_times_df.empty:
            new_stop_times_df.sort_values(["trip_id", "stop_sequence"], inplace=True)

        new_trips_df = pd.DataFrame(new_trips)
        return new_stop_times_df, new_trips_df


# ============================================================
# 2) Zones + arcs
# ============================================================

class ZoneAssigner:
    def __init__(self, zonas_csv: Path, gtfs_carpetas: Dict[str, Path]):
        self.zonas_csv = Path(zonas_csv)
        self.gtfs_carpetas = {k: Path(v) for k, v in gtfs_carpetas.items()}
        self.zonas_gdf = None
        self.paradas_gdf = None
        self.gtfs_info: Dict[str, Dict[str, pd.DataFrame]] = {}

    def cargar_zonas(self) -> None:
        df = pd.read_csv(self.zonas_csv)
        df["geometry"] = df["geometry"].apply(wkt.loads)
        gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:25830")
        self.zonas_gdf = gdf.to_crs(epsg=4326)

    def leer_stops(self, gtfs_folder: Path, tipo: str) -> gpd.GeoDataFrame:
        path_stops = gtfs_folder / "stops.txt"
        if not path_stops.exists():
            return gpd.GeoDataFrame()

        stops = pd.read_csv(path_stops)
        stops["geometry"] = stops.apply(lambda r: Point(r["stop_lon"], r["stop_lat"]), axis=1)
        stops["tipo"] = tipo
        return gpd.GeoDataFrame(stops, geometry="geometry", crs="EPSG:4326")

    def leer_todos_gtfs(self) -> None:
        list_paradas = []
        for tipo, folder in self.gtfs_carpetas.items():
            paradas = self.leer_stops(folder, tipo)
            if not paradas.empty:
                list_paradas.append(paradas)

                stop_times_path = folder / "stop_times.txt"
                trips_path = folder / "trips.txt"
                routes_path = folder / "routes.txt"
                if stop_times_path.exists() and trips_path.exists() and routes_path.exists():
                    self.gtfs_info[tipo] = {
                        "stop_times": pd.read_csv(stop_times_path),
                        "trips": pd.read_csv(trips_path),
                        "routes": pd.read_csv(routes_path),
                    }

        if list_paradas:
            self.paradas_gdf = pd.concat(list_paradas, ignore_index=True)

    def asignar_zonas(self) -> pd.DataFrame:
        if self.paradas_gdf is None or self.zonas_gdf is None:
            return pd.DataFrame()

        joined = gpd.sjoin(
            self.paradas_gdf,
            self.zonas_gdf[["ZT1259", "geometry"]],
            how="left",
            predicate="within",
        )

        result = joined[["stop_id", "stop_name", "stop_lat", "stop_lon", "tipo", "ZT1259"]]

        result = result[
            ((result["tipo"] == "metro") & (result["stop_id"].astype(str).str.startswith("par_")))
            | (result["tipo"] != "metro")
        ].copy()
        return result

    def procesar(self, out_csv: Path) -> pd.DataFrame:
        self.cargar_zonas()
        self.leer_todos_gtfs()
        df = self.asignar_zonas()
        df.to_csv(out_csv, index=False)
        return df


class ArcGenerator:
    def __init__(self, zonas_csv: Path, paradas_csv: Path):
        self.zonas_csv = Path(zonas_csv)
        self.paradas_csv = Path(paradas_csv)
        self.velocidad = 1.4  # m/s
        self.umbral = 800  # meters

    def cargar_zonas(self) -> None:
        df = pd.read_csv(self.zonas_csv)
        df["geometry"] = df["geometry"].apply(wkt.loads)
        self.gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:25830")
        self.gdf["centroid"] = self.gdf.geometry.centroid

    def cargar_paradas(self) -> None:
        self.paradas = pd.read_csv(self.paradas_csv)

    def calcular_vecinas(self) -> None:
        self.vecinas_dict = {}
        for _, zona in self.gdf.iterrows():
            c = zona["centroid"]
            otros = self.gdf[self.gdf["ZT1259"] != zona["ZT1259"]].copy()
            otros["dist"] = otros["centroid"].distance(c)
            self.vecinas_dict[zona["ZT1259"]] = otros[otros["dist"] <= self.umbral]["ZT1259"].tolist()

    @staticmethod
    def distancia_metros(lat1, lon1, lat2, lon2) -> float:
        R = 6371000
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)
        a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c

    def generar_arcos_caminando(self) -> pd.DataFrame:
        arcos = []
        pares_zonas = set()
        pares_paradas = set()

        for zona, vecinas in self.vecinas_dict.items():
            grupo = [zona] + vecinas
            for i in range(len(grupo)):
                for j in range(i, len(grupo)):
                    z1, z2 = grupo[i], grupo[j]
                    par_z = tuple(sorted([z1, z2]))
                    if par_z in pares_zonas:
                        continue
                    pares_zonas.add(par_z)

                    p_z1 = self.paradas[self.paradas["ZT1259"] == z1].reset_index(drop=True)
                    p_z2 = self.paradas[self.paradas["ZT1259"] == z2].reset_index(drop=True)

                    for p1 in p_z1.itertuples(index=False):
                        for p2 in p_z2.itertuples(index=False):
                            if z1 == z2 and p1.stop_id >= p2.stop_id:
                                continue
                            par_stops = (p1.stop_id, p2.stop_id)
                            if par_stops in pares_paradas:
                                continue
                            pares_paradas.add(par_stops)

                            d = math.ceil(self.distancia_metros(p1.stop_lat, p1.stop_lon, p2.stop_lat, p2.stop_lon))
                            tiempo_teorico = math.ceil(d / self.velocidad)

                            arcos.append(
                                {
                                    "from_stop": p1.stop_id,
                                    "to_stop": p2.stop_id,
                                    "from_lat": p1.stop_lat,
                                    "from_lon": p1.stop_lon,
                                    "to_lat": p2.stop_lat,
                                    "to_lon": p2.stop_lon,
                                    "time_theoretical_s": tiempo_teorico,
                                    "penalizacion_s": 0,
                                    "time_total_s": tiempo_teorico,
                                    "mode": "walking",
                                }
                            )

        return pd.DataFrame(arcos)

    def procesar_caminando(self, out_csv: Path) -> pd.DataFrame:
        self.cargar_zonas()
        self.cargar_paradas()
        self.calcular_vecinas()
        df = self.generar_arcos_caminando()
        df.to_csv(out_csv, index=False)
        return df

    @staticmethod
    def generar_arcos_gtfs(carpeta: Path, modo: str, archivo_salida: Path) -> pd.DataFrame:
        carpeta = Path(carpeta)

        trips = pd.read_csv(carpeta / "trips.txt", dtype=str)
        stop_times = pd.read_csv(carpeta / "stop_times.txt", dtype=str)
        stops = pd.read_csv(carpeta / "stops.txt", dtype=str)
        routes = pd.read_csv(carpeta / "routes.txt", dtype=str)

        route_name_map = dict(zip(routes["route_id"], routes["route_short_name"]))
        stop_coords = dict(zip(stops["stop_id"], zip(stops["stop_lat"], stops["stop_lon"])))

        arcos: List[Dict[str, Any]] = []

        if modo.lower() == "metro":
            for route_id in trips["route_id"].unique():
                trips_ruta = trips[trips["route_id"] == route_id]
                if trips_ruta.empty:
                    continue

                rep_trip_id = trips_ruta["trip_id"].iloc[0]
                trip = stop_times[stop_times["trip_id"] == rep_trip_id].copy()
                trip["stop_sequence"] = trip["stop_sequence"].astype(int)
                trip = trip.sort_values("stop_sequence").reset_index(drop=True)

                n = len(trip)

                # direct
                for i in range(n - 1):
                    p1 = trip.loc[i]
                    p2 = trip.loc[i + 1]
                    lat1, lon1 = stop_coords[p1["stop_id"]]
                    lat2, lon2 = stop_coords[p2["stop_id"]]

                    h1, m1, s1 = map(int, p1["departure_time"].split(":"))
                    h2, m2, s2 = map(int, p2["departure_time"].split(":"))
                    t1 = h1 * 3600 + m1 * 60 + s1
                    t2 = h2 * 3600 + m2 * 60 + s2
                    tiempo_s = max(t2 - t1, 1)

                    arcos.append(
                        {
                            "from_stop": p1["stop_id"],
                            "to_stop": p2["stop_id"],
                            "from_lat": lat1,
                            "from_lon": lon1,
                            "to_lat": lat2,
                            "to_lon": lon2,
                            "time_theoretical_s": tiempo_s,
                            "penalizacion_s": 0,
                            "time_total_s": tiempo_s,
                            "mode": "metro",
                            "linea": route_name_map.get(route_id, route_id),
                        }
                    )

                # reverse
                for i in range(n - 1, 0, -1):
                    p1 = trip.loc[i]
                    p2 = trip.loc[i - 1]
                    lat1, lon1 = stop_coords[p1["stop_id"]]
                    lat2, lon2 = stop_coords[p2["stop_id"]]

                    h1, m1, s1 = map(int, p1["departure_time"].split(":"))
                    h2, m2, s2 = map(int, p2["departure_time"].split(":"))
                    t1 = h2 * 3600 + m2 * 60 + s2
                    t2 = h1 * 3600 + m1 * 60 + s1
                    tiempo_s = max(t2 - t1, 1)

                    arcos.append(
                        {
                            "from_stop": p1["stop_id"],
                            "to_stop": p2["stop_id"],
                            "from_lat": lat1,
                            "from_lon": lon1,
                            "to_lat": lat2,
                            "to_lon": lon2,
                            "time_theoretical_s": tiempo_s,
                            "penalizacion_s": 0,
                            "time_total_s": tiempo_s,
                            "mode": "metro",
                            "linea": route_name_map.get(route_id, route_id),
                        }
                    )

        else:
            # EMT / Cercanias
            for (route_id, shape_id), group in trips.groupby(["route_id", "shape_id"]):
                trip_ids = group["trip_id"].tolist()
                stops_group = stop_times[stop_times["trip_id"].isin(trip_ids)].copy()
                if stops_group.empty:
                    continue

                stops_group["stop_sequence"] = stops_group["stop_sequence"].astype(int)
                stops_group = stops_group.sort_values(["trip_id", "stop_sequence"]).reset_index(drop=True)
                trip = stops_group[stops_group["trip_id"] == stops_group["trip_id"].unique()[0]].reset_index(drop=True)

                n = len(trip)
                for i in range(n - 1):
                    p1 = trip.loc[i]
                    p2 = trip.loc[i + 1]
                    lat1, lon1 = stop_coords[p1["stop_id"]]
                    lat2, lon2 = stop_coords[p2["stop_id"]]

                    h1, m1, s1 = map(int, p1["departure_time"].split(":"))
                    h2, m2, s2 = map(int, p2["departure_time"].split(":"))
                    t1 = h1 * 3600 + m1 * 60 + s1
                    t2 = h2 * 3600 + m2 * 60 + s2
                    tiempo_s = max(t2 - t1, 1)

                    arcos.append(
                        {
                            "from_stop": p1["stop_id"],
                            "to_stop": p2["stop_id"],
                            "from_lat": lat1,
                            "from_lon": lon1,
                            "to_lat": lat2,
                            "to_lon": lon2,
                            "time_theoretical_s": tiempo_s,
                            "penalizacion_s": 0,
                            "time_total_s": tiempo_s,
                            "mode": modo,
                            "linea": route_name_map.get(route_id, route_id),
                        }
                    )

        df_arcos = pd.DataFrame(arcos)
        df_arcos.to_csv(archivo_salida, index=False)
        return df_arcos


# ============================================================
# 3) Orchestrator for Django button: "Process GTFS"
# ============================================================

@dataclass
class ProcessResult:
    ok: bool
    logs: List[str]
    outputs: Dict[str, str]  # label -> relative media path


def process_transport_pipeline(
    emt_zip: Path,
    metro_zip: Path,
    cercanias_zip: Path,
    zonas_poligonos_csv: Path,
) -> ProcessResult:
    logs: List[str] = []
    out: Dict[str, str] = {}

    base = ensure_dir(transport_media_dir())
    gtfs_dir = ensure_dir(base / "gtfs_clean")
    outputs_dir = ensure_dir(base / "outputs")

    providers = [
        ("emt", EMTGTFSProcessor, Path(emt_zip)),
        ("metro", MetroGTFSProcessor, Path(metro_zip)),
        ("cercanias", CercaniasGTFSProcessor, Path(cercanias_zip)),
    ]

    gtfs_folders: Dict[str, Path] = {}
    for name, cls, zip_path in providers:
        folder = ensure_dir(gtfs_dir / name)
        logs.append(f"[GTFS] Processing {name} from {zip_path.name} ...")
        cls(zip_path=zip_path, out_folder=folder).procesar()
        gtfs_folders[name] = folder
        logs.append(f"[GTFS] {name} OK -> {folder}")

    zonas_poligonos_csv = Path(zonas_poligonos_csv)
    zonas_paradas_csv = outputs_dir / "zonas_transporte_paradas.csv"

    logs.append("[ZONES] Assigning zones to stops...")
    ZoneAssigner(zonas_poligonos_csv, gtfs_folders).procesar(zonas_paradas_csv)
    logs.append(f"[ZONES] OK -> {zonas_paradas_csv.name}")
    out["zonas_transporte_paradas.csv"] = str(zonas_paradas_csv.relative_to(settings.MEDIA_ROOT))

    logs.append("[ARCS] Generating walking arcs...")
    arcos_walk_csv = outputs_dir / "arcos_caminando.csv"
    ArcGenerator(zonas_poligonos_csv, zonas_paradas_csv).procesar_caminando(arcos_walk_csv)
    logs.append(f"[ARCS] walking OK -> {arcos_walk_csv.name}")
    out["arcos_caminando.csv"] = str(arcos_walk_csv.relative_to(settings.MEDIA_ROOT))

    logs.append("[ARCS] Generating GTFS arcs: metro/emt/cercanias...")
    arcos_metro = outputs_dir / "arcos_metro.csv"
    arcos_emt = outputs_dir / "arcos_emt.csv"
    arcos_cerc = outputs_dir / "arcos_cercanias.csv"

    ArcGenerator.generar_arcos_gtfs(gtfs_folders["metro"], "metro", arcos_metro)
    ArcGenerator.generar_arcos_gtfs(gtfs_folders["emt"], "emt", arcos_emt)
    ArcGenerator.generar_arcos_gtfs(gtfs_folders["cercanias"], "cercanias", arcos_cerc)

    logs.append("[ARCS] metro OK")
    logs.append("[ARCS] emt OK")
    logs.append("[ARCS] cercanias OK")

    out["arcos_metro.csv"] = str(arcos_metro.relative_to(settings.MEDIA_ROOT))
    out["arcos_emt.csv"] = str(arcos_emt.relative_to(settings.MEDIA_ROOT))
    out["arcos_cercanias.csv"] = str(arcos_cerc.relative_to(settings.MEDIA_ROOT))

    return ProcessResult(ok=True, logs=logs, outputs=out)


# ============================================================
# 4) Geocoding (Mapbox) 
# ============================================================

class GeocodingService:
    """
    Servicio para convertir nombres de lugares en coordenadas usando Mapbox Geocoding API.
    """

    def __init__(self, token: str):
        self.token = token
        self.base_url = "https://api.mapbox.com/geocoding/v5/mapbox.places"

    def get_coordinates(self, place_name: str, limit: int = 1, language: str = "es") -> Optional[Tuple[float, float]]:
        encoded_place = requests.utils.quote(place_name)
        url = f"{self.base_url}/{encoded_place}.json"

        params = {
            "access_token": self.token,
            "limit": limit,
            "language": language,
            # Madrid bbox (para evitar que te devuelva sitios de otro país)
            "bbox": "-3.888,40.312,-3.517,40.643",
        }

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if not data.get("features"):
                return None

            lon, lat = data["features"][0]["geometry"]["coordinates"]
            return (float(lat), float(lon))
        except Exception:
            return None


# ============================================================
# 5) RouteCalculator (tu lógica) + helpers places->zone 
# ============================================================

class RouteCalculator:
    """
    - Construye grafo de transporte (metro, EMT, cercanías, caminando)
    - Aplica penalizaciones de transbordo configurables
    - Calcula ruta mínima entre zonas
    - Permite activar/desactivar incidencias (cortes de tramo en una línea)
      guardadas en MEDIA_ROOT/transport/incidencias.json

    + NUEVO:
      - get_zone_from_coords(lat, lon)
      - find_route_by_places(origin_place, dest_place, geocoder)
    """

    MODE_WALK_STRS = {"caminando", "walk", "walking"}

    def __init__(
        self,
        outputs_dir: Optional[Path] = None,
        default_walk_transfer_min: int = 10,
        transfer_matrix_min: Optional[Dict[str, Dict[str, int]]] = None,
        incidencias_file: Optional[Path] = None,
        zonas_poligonos_csv: Optional[Path] = None,   
    ):
        self.outputs_dir = Path(outputs_dir) if outputs_dir else transport_outputs_dir()

        self.DEFAULT_WALK_TRANSFER = int(default_walk_transfer_min) * 60

        self.TRANSFER_MATRIX = (
            {m1: {m2: int(t) * 60 for m2, t in d.items()} for m1, d in transfer_matrix_min.items()}
            if transfer_matrix_min
            else {
                "metro": {"metro": 6 * 60, "cercanias": 8 * 60, "emt": 15 * 60},
                "cercanias": {"metro": 6 * 60, "cercanias": 8 * 60, "emt": 15 * 60},
                "emt": {"metro": 6 * 60, "cercanias": 12 * 60, "emt": 15 * 60},
            }
        )

        self.F_CAMINANDO = self.outputs_dir / "arcos_caminando.csv"
        self.F_CERCANIAS = self.outputs_dir / "arcos_cercanias.csv"
        self.F_EMT = self.outputs_dir / "arcos_emt.csv"
        self.F_METRO = self.outputs_dir / "arcos_metro.csv"
        self.F_PARADAS_ZONAS = self.outputs_dir / "zonas_transporte_paradas.csv"

        self.incidencias_file = Path(incidencias_file) if incidencias_file else transport_incidencias_path()
        self.incidencias_activas = self._load_incidencias()

        #  polígonos de zonas (para place->zone)
        self.zonas_poligonos_csv = Path(zonas_poligonos_csv) if zonas_poligonos_csv else transport_zonas_poligonos_path()
        self._zonas_gdf_cache: Optional[gpd.GeoDataFrame] = None

        self.stop_zone_map = self._build_stop_zone_map()
        self.G_base = self._build_base_graph()

    # ---------------------- Helpers -------------------------------
    def _node_name(self, stop_id, modo, linea):
        stop = str(stop_id)
        modo = "" if pd.isna(modo) else str(modo).strip()
        linea = "" if (pd.isna(linea) or linea is None) else str(linea).strip()
        if modo.lower() in self.MODE_WALK_STRS:
            return f"{stop}__walk"
        return f"{stop}__{modo}_{linea}"

    def _parse_node(self, node):
        if "__" not in node:
            return node, None, None
        stop, rest = node.split("__", 1)
        if rest == "walk":
            return stop, "walk", ""
        if "_" in rest:
            modo, linea = rest.split("_", 1)
            return stop, modo, linea
        return stop, rest, ""

    def _safe_read_csv(self, path: Path) -> pd.DataFrame:
        if not Path(path).exists():
            return pd.DataFrame()
        return pd.read_csv(path, dtype=str)

    #  cargar/cachar gdf de zonas
    def _load_zonas_gdf(self) -> Optional[gpd.GeoDataFrame]:
        if self._zonas_gdf_cache is not None:
            return self._zonas_gdf_cache

        if not self.zonas_poligonos_csv or not self.zonas_poligonos_csv.exists():
            return None

        try:
            df = pd.read_csv(self.zonas_poligonos_csv)
            if "geometry" not in df.columns or "ZT1259" not in df.columns:
                return None
            df["geometry"] = df["geometry"].apply(wkt.loads)
            gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:25830").to_crs(epsg=4326)
            self._zonas_gdf_cache = gdf
            return gdf
        except Exception:
            return None

    #  coords -> zone
    def get_zone_from_coords(self, lat: float, lon: float) -> Optional[str]:
        gdf = self._load_zonas_gdf()
        if gdf is None or gdf.empty:
            return None

        p = Point(float(lon), float(lat))  # shapely usa (x=lon, y=lat)
        hit = gdf[gdf.contains(p)]
        if hit.empty:
            return None
        return str(hit.iloc[0]["ZT1259"])

    #  places -> route
    def find_route_by_places(
        self,
        origin_place: str,
        dest_place: str,
        geocoding_service: "GeocodingService",
    ) -> Optional[Dict[str, Any]]:
        o = geocoding_service.get_coordinates(origin_place)
        d = geocoding_service.get_coordinates(dest_place)
        if not o or not d:
            return None

        origin_zone = self.get_zone_from_coords(*o)
        dest_zone = self.get_zone_from_coords(*d)
        if not origin_zone or not dest_zone:
            return None

        return self.find_route_zones(origin_zone, dest_zone)

    def generar_id(self, prefix: str = "INC", width: int = 3) -> str:
        incidencias = []
        if self.incidencias_file and self.incidencias_file.exists():
            try:
                with open(self.incidencias_file, "r", encoding="utf-8") as f:
                    incidencias = json.load(f) or []
            except Exception:
                incidencias = []

        usados = set()
        for inc in incidencias:
            _id = str(inc.get("id", "")).strip()
            if _id.startswith(prefix):
                usados.add(_id)

        n = 1
        while True:
            cand = f"{prefix}{n:0{width}d}"
            if cand not in usados:
                return cand
            n += 1

    # ---------------------- Incidencias ---------------------------
    def _load_incidencias(self) -> List[Dict[str, Any]]:
        if not self.incidencias_file or not self.incidencias_file.exists():
            return []
        try:
            with open(self.incidencias_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                return []
            return [inc for inc in data if inc.get("estado") == "activa"]
        except Exception:
            return []

    def activar_incidencia(
        self,
        incidencia_id: Optional[str] = None,
        modo: Optional[str] = None,
        linea: Optional[str] = None,
        stop_desde: Optional[str] = None,
        stop_hasta: Optional[str] = None,
        tipo: str = "corte_linea",
    ) -> bool:
        if not self.incidencias_file:
            return False

        if incidencia_id is None:
            incidencia_id = self.generar_id()

        if self.incidencias_file.exists():
            try:
                with open(self.incidencias_file, "r", encoding="utf-8") as f:
                    incidencias = json.load(f) or []
            except Exception:
                incidencias = []
        else:
            incidencias = []

        incidencia_existente = None
        for inc in incidencias:
            if inc.get("id") == incidencia_id:
                incidencia_existente = inc
                break

        if incidencia_existente:
            incidencia_existente["estado"] = "activa"
        else:
            if not all([modo, linea, stop_desde, stop_hasta]):
                return False

            incidencias.append(
                {
                    "id": incidencia_id,
                    "tipo": tipo,
                    "modo": str(modo).strip().lower(),
                    "linea": str(linea).strip(),
                    "stop_desde": str(stop_desde).strip(),
                    "stop_hasta": str(stop_hasta).strip(),
                    "estado": "activa",
                }
            )

        ensure_dir(self.incidencias_file.parent)
        with open(self.incidencias_file, "w", encoding="utf-8") as f:
            json.dump(incidencias, f, indent=2, ensure_ascii=False)

        self.incidencias_activas = self._load_incidencias()
        return True

    def desactivar_incidencia(self, incidencia_id: str) -> bool:
        if not self.incidencias_file or not self.incidencias_file.exists():
            return False

        try:
            with open(self.incidencias_file, "r", encoding="utf-8") as f:
                incidencias = json.load(f) or []
        except Exception:
            return False

        encontrada = False
        for inc in incidencias:
            if inc.get("id") == incidencia_id:
                inc["estado"] = "inactiva"
                encontrada = True
                break

        if not encontrada:
            return False

        with open(self.incidencias_file, "w", encoding="utf-8") as f:
            json.dump(incidencias, f, indent=2, ensure_ascii=False)

        self.incidencias_activas = self._load_incidencias()
        return True

    def listar_incidencias(self, solo_activas: bool = True) -> pd.DataFrame:
        if not self.incidencias_file or not self.incidencias_file.exists():
            return pd.DataFrame()
        try:
            with open(self.incidencias_file, "r", encoding="utf-8") as f:
                incidencias = json.load(f) or []
        except Exception:
            return pd.DataFrame()

        df = pd.DataFrame(incidencias)
        if solo_activas and not df.empty and "estado" in df.columns:
            df = df[df["estado"] == "activa"]
        return df

    # ---------------------- Stops/Zones ---------------------------
    def _build_stop_zone_map(self) -> pd.DataFrame:
        df = self._safe_read_csv(self.F_PARADAS_ZONAS)
        if df.empty:
            return pd.DataFrame(columns=["stop_id", "zone"])
        df["stop_id"] = df["stop_id"].astype(str)
        df["ZT1259"] = df["ZT1259"].astype(str)
        return df.rename(columns={"ZT1259": "zone"})[["stop_id", "zone"]].copy()

    def get_stop_name(self, stop_id) -> str:
        if not hasattr(self, "_stop_names"):
            df = self._safe_read_csv(self.F_PARADAS_ZONAS)
            if df.empty or "stop_id" not in df.columns or "stop_name" not in df.columns:
                self._stop_names = {}
            else:
                self._stop_names = dict(zip(df["stop_id"].astype(str), df["stop_name"].astype(str)))
        return self._stop_names.get(str(stop_id), "")

    # ---------------------- Graph build ---------------------------
    def _build_base_graph(self) -> nx.DiGraph:
        G = nx.DiGraph()

        dfs = []
        for path in [self.F_CAMINANDO, self.F_METRO, self.F_EMT, self.F_CERCANIAS]:
            df = self._safe_read_csv(path)
            if not df.empty:
                dfs.append(df)

        if not dfs:
            return G

        arcs = pd.concat(dfs, ignore_index=True)

        for col in ["from_stop", "to_stop", "mode", "linea", "time_total_s"]:
            if col not in arcs.columns:
                arcs[col] = ""

        arcs["from_stop"] = arcs["from_stop"].astype(str)
        arcs["to_stop"] = arcs["to_stop"].astype(str)
        arcs["mode"] = arcs["mode"].astype(str).str.strip().str.lower()
        arcs["linea"] = arcs["linea"].astype(str).fillna("").str.strip()

        for r in arcs.itertuples(index=False):
            modo = getattr(r, "mode", "")
            linea = getattr(r, "linea", "")
            u = self._node_name(getattr(r, "from_stop", ""), modo, linea)
            v = self._node_name(getattr(r, "to_stop", ""), modo, linea)

            t = getattr(r, "time_total_s", "0")
            try:
                w = float(t)
            except Exception:
                w = 0.0

            stop_u, modo_u, linea_u = self._parse_node(u)
            stop_v, modo_v, linea_v = self._parse_node(v)

            if not G.has_node(u):
                G.add_node(u, stop=stop_u, modo=modo_u, linea=linea_u)
            if not G.has_node(v):
                G.add_node(v, stop=stop_v, modo=modo_v, linea=linea_v)

            if u != v:
                G.add_edge(u, v, weight=w, is_transfer=False)

        stop_to_nodes: Dict[str, List[str]] = {}
        for n, d in G.nodes(data=True):
            stop_to_nodes.setdefault(d.get("stop"), []).append(n)

        for stop, nodes in stop_to_nodes.items():
            if len(nodes) < 2:
                continue
            for u in nodes:
                for v in nodes:
                    if u == v:
                        continue
                    if not G.has_edge(u, v):
                        G.add_edge(u, v, weight=self.DEFAULT_WALK_TRANSFER, is_transfer=True)

        return G

    def _apply_incidencias(self, G: nx.DiGraph) -> None:
        if not self.incidencias_activas:
            return

        for inc in self.incidencias_activas:
            modo = str(inc.get("modo", "")).strip().lower()
            linea = str(inc.get("linea", "")).strip()
            stop_desde = str(inc.get("stop_desde", "")).strip()
            stop_hasta = str(inc.get("stop_hasta", "")).strip()

            nodo_desde = self._node_name(stop_desde, modo, linea)
            nodo_hasta = self._node_name(stop_hasta, modo, linea)

            if not (G.has_node(nodo_desde) and G.has_node(nodo_hasta)):
                continue

            line_nodes = []
            for n, d in G.nodes(data=True):
                if d.get("modo") == modo and str(d.get("linea", "")).strip() == linea:
                    line_nodes.append(n)

            if not line_nodes:
                continue

            H = G.subgraph(line_nodes).copy()
            to_remove = [(u, v) for u, v, ed in H.edges(data=True) if ed.get("is_transfer")]
            if to_remove:
                H.remove_edges_from(to_remove)

            try:
                camino = nx.shortest_path(H, nodo_desde, nodo_hasta)
            except nx.NetworkXNoPath:
                try:
                    camino = nx.shortest_path(H, nodo_hasta, nodo_desde)
                except nx.NetworkXNoPath:
                    continue

            for i in range(len(camino) - 1):
                n1 = camino[i]
                n2 = camino[i + 1]
                if G.has_edge(n1, n2):
                    G.remove_edge(n1, n2)
                if G.has_edge(n2, n1):
                    G.remove_edge(n2, n1)

    def _adjust_weights_for_query(self, G: nx.DiGraph) -> None:
        for u, v, data in G.edges(data=True):
            if data.get("is_transfer"):
                _, modo_u, _ = self._parse_node(u)
                _, modo_v, _ = self._parse_node(v)

                if modo_u == "walk" or modo_v == "walk":
                    penalty = self.DEFAULT_WALK_TRANSFER
                else:
                    penalty = self.TRANSFER_MATRIX.get(modo_u, {}).get(modo_v, self.DEFAULT_WALK_TRANSFER)

                G[u][v]["weight"] = penalty
    # ========================================================
    # DEMAND helpers (cache + EMT)
    # ========================================================

    def _get_incidencias_id(self) -> str:
        incs = self.listar_incidencias(solo_activas=True)
        if incs is None or incs.empty:
            return "NONE"
        payload = incs.sort_values("id").to_json(orient="records", force_ascii=False)
        return hashlib.md5(payload.encode("utf-8")).hexdigest()[:8].upper()

    def find_route_zones_cached(self, origin_zone: str, dest_zone: str) -> Optional[Dict[str, Any]]:
        cache_file = transport_cache_dir() / "rutas_zonas_cache.csv"
        inc_id = self._get_incidencias_id()

        if cache_file.exists():
            df = pd.read_csv(cache_file, dtype=str)
            hit = df[
                (df["origen_zona"] == str(origin_zone)) &
                (df["destino_zona"] == str(dest_zone)) &
                (df["incidencias_id"] == inc_id)
            ]
            if not hit.empty:
                return json.loads(hit.iloc[0]["ruta_json"])

        result = self.find_route_zones(origin_zone, dest_zone)
        if result is None:
            return None

        row = {
            "origen_zona": origin_zone,
            "destino_zona": dest_zone,
            "incidencias_id": inc_id,
            "ruta_json": json.dumps(result, ensure_ascii=False),
            "tiempo_total_s": int(result.get("total_time_s", 0)),
        }

        if cache_file.exists():
            df = pd.read_csv(cache_file, dtype=str)
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        else:
            df = pd.DataFrame([row])

        df.to_csv(cache_file, index=False)
        return result

    def get_lineas_por_modo(self, modo: str) -> List[str]:
        lineas = set()
        for node in self.G_base.nodes():
            _, m, l = self._parse_node(node)
            if (m or "").lower() == modo.lower() and l:
                lineas.add(str(l))
        return sorted(lineas, key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else x))

    def get_paradas_linea(self, modo: str, linea: str) -> List[Tuple[str, str]]:
        stops = set()
        for node in self.G_base.nodes():
            stop_id, m, l = self._parse_node(node)
            if (m or "").lower() == modo.lower() and str(l) == str(linea):
                stops.add(stop_id)
        return [(s, self.get_stop_name(s)) for s in sorted(stops)]

    # ---------------------- Main query ----------------------------
    def find_route_zones(self, origin_zone: str, dest_zone: str) -> Optional[Dict[str, Any]]:
        G = self.G_base.copy()
        self._apply_incidencias(G)

        ZO = f"Z_ORIG__{origin_zone}"
        ZD = f"Z_DEST__{dest_zone}"
        G.add_node(ZO, zone=origin_zone)
        G.add_node(ZD, zone=dest_zone)

        origin_stops = self.stop_zone_map[self.stop_zone_map.zone == str(origin_zone)]["stop_id"].astype(str)
        dest_stops = self.stop_zone_map[self.stop_zone_map.zone == str(dest_zone)]["stop_id"].astype(str)

        if origin_stops.empty or dest_stops.empty:
            return None

        for s in origin_stops:
            for n, d in G.nodes(data=True):
                if d.get("stop") == s:
                    G.add_edge(ZO, n, weight=0, is_transfer=False)

        for s in dest_stops:
            for n, d in G.nodes(data=True):
                if d.get("stop") == s:
                    G.add_edge(n, ZD, weight=0, is_transfer=False)

        self._adjust_weights_for_query(G)

        try:
            path = nx.shortest_path(G, ZO, ZD, weight="weight")
            length = nx.shortest_path_length(G, ZO, ZD, weight="weight")
        except nx.NetworkXNoPath:
            return None

        steps = []
        for i in range(len(path) - 1):
            u = path[i]
            v = path[i + 1]
            w = float(G[u][v].get("weight", 0))
            is_transfer = bool(G[u][v].get("is_transfer", False))

            if u.startswith("Z_ORIG__") or v.startswith("Z_DEST__"):
                continue

            stop_u, modo_u, linea_u = self._parse_node(u)
            stop_v, modo_v, linea_v = self._parse_node(v)

            steps.append(
                {
                    "from_node": u,
                    "to_node": v,
                    "from_stop": stop_u,
                    "to_stop": stop_v,
                    "from_stop_name": self.get_stop_name(stop_u),
                    "to_stop_name": self.get_stop_name(stop_v),
                    "mode_from": modo_u,
                    "mode_to": modo_v,
                    "line_from": linea_u,
                    "line_to": linea_v,
                    "is_transfer": is_transfer,
                    "time_s": w,
                }
            )

        return {
            "origin_zone": str(origin_zone),
            "dest_zone": str(dest_zone),
            "total_time_s": float(length),
            "path_nodes": path,
            "steps": steps,
        }


# ============================================================
# DEMAND / EMT — build emt_mapa.csv
# ============================================================

class InformationEMT:
    def __init__(self, gtfs_dir: Path, output_csv: Optional[Path] = None):
        self.gtfs_dir = Path(gtfs_dir)
        self.output_csv = output_csv or transport_emt_mapa_path()

        self.routes = pd.read_csv(self.gtfs_dir / "routes.txt", dtype=str)
        self.trips = pd.read_csv(self.gtfs_dir / "trips.txt", dtype=str)
        self.stop_times = pd.read_csv(self.gtfs_dir / "stop_times.txt", dtype=str)
        self.stops = pd.read_csv(self.gtfs_dir / "stops.txt", dtype=str)

        self.stop_names = dict(zip(self.stops["stop_id"], self.stops["stop_name"]))

    def generar_csv(self) -> pd.DataFrame:
        rows = []

        for _, route in self.routes.iterrows():
            linea = route["route_short_name"]
            route_id = route["route_id"]

            trips_r = self.trips[self.trips["route_id"] == route_id]
            if "service_id" in trips_r.columns:
                trips_r = trips_r[trips_r["service_id"] == "LA"]
            if trips_r.empty:
                continue

            shapes = trips_r["shape_id"].dropna().unique()
            for i, shape in enumerate(shapes):
                sentido = "directo" if i == 0 else "inverso"
                trip_id = trips_r[trips_r["shape_id"] == shape]["trip_id"].iloc[0]

                st = self.stop_times[self.stop_times["trip_id"] == trip_id].copy()
                st["stop_sequence"] = st["stop_sequence"].astype(int)
                st = st.sort_values("stop_sequence")

                for j in range(len(st) - 1):
                    a, b = st.iloc[j], st.iloc[j + 1]
                    rows.append({
                        "linea": linea,
                        "modo": "emt",
                        "sentido": sentido,
                        "from_stop": a["stop_id"],
                        "from_stop_name": self.stop_names.get(a["stop_id"], ""),
                        "to_stop": b["stop_id"],
                        "to_stop_name": self.stop_names.get(b["stop_id"], ""),
                    })

        df = pd.DataFrame(rows)
        ensure_dir(self.output_csv.parent)
        df.to_csv(self.output_csv, index=False)
        return df
# ============================================================
# DEMAND ANALYSIS
# ============================================================

class DemandAnalyzer:
    def __init__(self, ruta: RouteCalculator):
        self.ruta = ruta
        self.emt_mapa = transport_emt_mapa_path()

        if not self.emt_mapa.exists():
            InformationEMT(transport_gtfs_clean_dir() / "emt").generar_csv()

        self.df_mapa = pd.read_csv(self.emt_mapa, dtype=str).fillna("")
        self.df_mapa["_key"] = (
            self.df_mapa["linea"] + "|" +
            self.df_mapa["from_stop"] + "|" +
            self.df_mapa["to_stop"]
        )

    def analizar(self, distritos: List[int]) -> pd.DataFrame:
        viajes = transport_viajes_estudio_path()
        if not viajes.exists():
            raise FileNotFoundError("viajes_estudio.csv no encontrado")

        df_v = pd.read_csv(viajes, dtype=str)

        def dist(z):
            return z.split("-")[1] if "-" in z else ""

        df_v["do"] = df_v["zona_origen"].map(dist)
        df_v["dd"] = df_v["zona_destino"].map(dist)

        distritos = [str(d).zfill(2) for d in distritos]
        df_v = df_v[df_v["do"].isin(distritos) & df_v["dd"].isin(distritos)]

        cont = {}
        for _, r in df_v.iterrows():
            ruta = self.ruta.find_route_zones_cached(r["zona_origen"], r["zona_destino"])
            if not ruta:
                continue
            for s in ruta["steps"]:
                if s["mode_from"] != "emt" or s["is_transfer"]:
                    continue
                k = f'{s["line_from"]}|{s["from_stop"]}|{s["to_stop"]}'
                cont[k] = cont.get(k, 0) + 1

        df = self.df_mapa.copy()
        df["transit"] = df["_key"].map(cont).fillna(0).astype(int)
        return df


class VisualizationGenerator:
    def __init__(self, analyzer: DemandAnalyzer):
        self.analyzer = analyzer

    def plot_linea(self, linea: str, sentido: str, distritos: List[int]):
        df = self.analyzer.analizar(distritos)
        df = df[(df["linea"] == linea) & (df["sentido"] == sentido)]
        if df.empty:
            return None

        paradas = df["from_stop_name"].tolist() + [df.iloc[-1]["to_stop_name"]]
        valores = df["transit"].astype(int).tolist()

        fig, ax = plt.subplots(figsize=(max(14, len(paradas)*0.4), 4))
        ax.plot(range(len(paradas)), [0]*len(paradas), linewidth=6)
        ax.set_xticks(range(len(paradas)))
        ax.set_xticklabels(paradas, rotation=90)
        ax.set_yticks([])
        ax.set_title(f"LÍNEA {linea} — {sentido.upper()}")

        for i, v in enumerate(valores):
            ax.text(i+0.5, 0.3, str(v), ha="center", va="center",
                    bbox=dict(boxstyle="round", fc="white"))

        fig.tight_layout()
        return fig
