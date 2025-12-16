from django.http import JsonResponse
from django.views.decorators.http import require_POST
from django.views.decorators.csrf import csrf_exempt

from .engine_runner import run_optimization
from django.contrib.auth.decorators import login_required
import requests
from django.conf import settings

from django.shortcuts import render, redirect
from django.contrib import messages
import os
from pathlib import Path

from .ads_service import list_ads, create_ad, update_ad_active, delete_ad
from .clip_ad_classifier import classify_ad_image

from datetime import datetime, timezone
import requests
from django.views.decorators.http import require_http_methods

print("[DEBUG] Importando daedalus/views.py", flush=True)

@login_required

def dashboard(request):
    return render(request, "daedalus/dashboard.html")

@login_required

def runs_view(request):
    return render(request, "daedalus/runs.html")

@login_required

def dynamic_map(request):
    return render(request, "daedalus/dynamic_map.html")

@login_required
@csrf_exempt
@require_POST
def run_opt_view(request):
    """
    Lanza intelligence_engine_demo.py y devuelve el resultado en JSON.
    """
    result = run_optimization()
    status = "ok" if result["returncode"] == 0 else "error"

    return JsonResponse({
        "status": status,
        "returncode": result["returncode"],
        "stdout": result["stdout"],
        "stderr": result["stderr"],
    })


def home(request):
    """
    Página de inicio de Daedalus: descripción del proyecto, contacto, etc.
    """
    return render(request, "daedalus/home.html")

@login_required

def dashboard1(request):
    """
    Dashboard de KPIs: gráficos por run.
    Todos los datos se cargan vía Supabase desde el frontend.
    """
    return render(request, "daedalus/dashboard1.html")

@login_required

def hotspots_map(request):
    """
    Mapa de hotspots (paradas con alta espera / saturación) por run.
    Todos los datos se cargan desde Supabase en el frontend.
    """
    return render(request, "daedalus/hotspots.html")


SUPABASE_URL = "https://rrbuwzcoiyncvmzctvnr.supabase.co"
SUPABASE_SERVICE_KEY = (
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InJyYnV3emNvaXlu"
    "Y3ZtemN0dm5yIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc2MzE1NzE1MywiZXhwIjoyMDc4NzMzMTUzfQ."
    "7tJ32D_TQeQ9xWVW8lYTv_T2c0N6U08adn2Llk-8iVk"
)

def supabase_headers():
    return {
        "apikey": SUPABASE_SERVICE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
        "Content-Type": "application/json",
    }


@login_required
def ads_panel(request):
    """
    Simple panel that shows, for each stop,
    the last selected ad and how long it has been active.
    """
    district_filter = request.GET.get("district")

    url = f"{SUPABASE_URL}/rest/v1/v_station_current_ad"
    params = {"select": "*"}
    if district_filter:
        params["district_id"] = f"eq.{district_filter}"

    # ============================
    # Query Supabase
    # ============================
    try:
        resp = requests.get(url, headers=supabase_headers(), params=params, timeout=10)
        resp.raise_for_status()
        rows = resp.json()
    except Exception as e:
        print("Error querying v_station_current_ad:", e)
        rows = []

    # ============================
    # Add human-readable "ad_duration_human"
    # ============================
    now = datetime.now(timezone.utc)

    def human_delta(delta):
        mins = int(delta.total_seconds() // 60)
        if mins < 1:
            return "less than 1 min"
        if mins < 60:
            return f"{mins} min"
        hours = mins // 60
        rem = mins % 60
        if hours < 24:
            return f"{hours} h {rem} min" if rem else f"{hours} h"
        days = hours // 24
        hours = hours % 24
        if days == 1:
            base = "1 day"
        else:
            base = f"{days} days"
        if hours:
            base += f" {hours} h"
        return base

    for r in rows:
        ts_str = r.get("ts")
        if not ts_str:
            r["ad_duration_human"] = "—"
            continue
        try:
            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            delta = now - ts
            r["ad_duration_human"] = human_delta(delta)
        except Exception:
            r["ad_duration_human"] = "—"

    # district dropdown
    districts = [f"{i:02d}" for i in range(1, 22)]

    context = {
        "rows": rows,
        "districts": districts,
        "selected_district": district_filter,
    }
    return render(request, "daedalus/ads_panel.html", context)


@login_required
def ads_map(request):
    """
    Page with the advertising map (HTML only, data loaded via /daedalus/api/ads-map-data/).
    """
    # For now we don't pass anything special, everything is loaded via JS
    return render(request, "daedalus/ads_map.html")


@login_required
def ads_map_data(request):
    """
    JSON endpoint for the advertising map.
    Returns a list of stops with their current ad and associated data.
    """
    # Optional: district filter with ?district=04
    district_filter = request.GET.get("district")

    url = f"{SUPABASE_URL}/rest/v1/v_station_current_ad"
    # Select explicitly the fields we need
    params = {
        "select": "station_id,stop_name,district_id,lat,lon,ts,people_count,"
                  "age_segment,gender_segment,product_segment,ad_id,ad_name"
    }
    if district_filter:
        params["district_id"] = f"eq.{district_filter}"

    try:
        resp = requests.get(url, headers=supabase_headers(), params=params, timeout=10)
        resp.raise_for_status()
        rows = resp.json()
    except Exception as e:
        print("Error querying v_station_current_ad:", e)
        rows = []

    return JsonResponse(rows, safe=False)


@login_required
def ads_admin(request):
    if request.method == "POST":
        action = request.POST.get("action")

        try:
            if action == "create":
                ad_id = request.POST.get("ad_id", "").strip()
                name = request.POST.get("name", "").strip()
                active = bool(request.POST.get("active", "on") == "on")

                if not ad_id:
                    messages.error(request, "Ad ID field is required.")
                    return redirect("daedalus-ads-admin")

                # 1) Save image if provided
                upload = request.FILES.get("image")
                image_url = None
                age_segment = None
                gender_segment = None
                product_segment = None

                if upload:
                    ads_dir = Path(settings.MEDIA_ROOT) / "ads"
                    ads_dir.mkdir(parents=True, exist_ok=True)

                    # simple filename: <ad_id>_<original>
                    filename = f"{ad_id}_{upload.name}"
                    file_path = ads_dir / filename

                    # save file
                    with open(file_path, "wb+") as dest:
                        for chunk in upload.chunks():
                            dest.write(chunk)

                    # public URL for Django
                    image_url = f"{settings.MEDIA_URL}ads/{filename}"

                    # 2) Auto-classify with CLIP
                    segments = classify_ad_image(file_path)
                    age_segment = segments["age_segment"]
                    gender_segment = segments["gender_segment"]
                    product_segment = segments["product_segment"]

                    messages.info(
                        request,
                        f"CLIP classification → age: {age_segment}, gender: {gender_segment}, product: {product_segment}",
                    )

                # 3) Create ad in Supabase
                create_ad(
                    ad_id=ad_id,
                    name=name,
                    age_segment=age_segment,
                    gender_segment=gender_segment,
                    product_segment=product_segment,
                    image_url=image_url,
                    active=active,
                )
                messages.success(request, f"Ad {ad_id} created.")
                return redirect("daedalus-ads-admin")

            elif action == "toggle_active":
                ad_id = request.POST.get("ad_id")
                current = request.POST.get("current_active") == "True"
                update_ad_active(ad_id, not current)
                messages.success(
                    request,
                    f"Ad {ad_id} marked as {'active' if not current else 'inactive'}.",
                )
                return redirect("daedalus-ads-admin")

            elif action == "delete":
                ad_id = request.POST.get("ad_id")
                delete_ad(ad_id)
                messages.success(request, f"Ad {ad_id} permanently deleted.")
                return redirect("daedalus-ads-admin")

        except Exception as e:
            messages.error(request, f"Error processing action: {e}")

    ads = list_ads()
    return render(request, "daedalus/ads_admin.html", {"ads": ads})


from .transport_engine import process_transport_pipeline, transport_media_dir, ensure_dir

@login_required
@require_http_methods(["GET", "POST"])
def transport_init_upload(request):
    base = ensure_dir(transport_media_dir())
    uploads_dir = ensure_dir(base / "uploads")

    if request.method == "POST":
        action = request.POST.get("action", "upload")

        if action == "upload":
            # expect 3 files
            emt = request.FILES.get("emt_zip")
            metro = request.FILES.get("metro_zip")
            cerc = request.FILES.get("cercanias_zip")

            if not (emt and metro and cerc):
                messages.error(request, "Please upload the 3 ZIP files (EMT, Metro, Cercanias).")
                return redirect("daedalus-transport-init")

            # save with fixed names (simple)
            (uploads_dir / "emt.zip").write_bytes(emt.read())
            (uploads_dir / "metro.zip").write_bytes(metro.read())
            (uploads_dir / "cercanias.zip").write_bytes(cerc.read())

            messages.success(request, "Uploads OK. Now you can run 'Process GTFS'.")
            return redirect("daedalus-transport-init")

        if action == "process":
            # inputs
            emt_zip = uploads_dir / "emt.zip"
            metro_zip = uploads_dir / "metro.zip"
            cerc_zip = uploads_dir / "cercanias.zip"

            # IMPORTANT: you must place this file somewhere reachable.
            # Recommended: MEDIA_ROOT/transport/zonas_transporte_poligonos.csv
            zonas_csv = base / "zonas_transporte_poligonos.csv"

            missing = [p.name for p in [emt_zip, metro_zip, cerc_zip, zonas_csv] if not p.exists()]
            if missing:
                messages.error(
                    request,
                    "Missing required files: " + ", ".join(missing) +
                    ". Upload ZIPs and make sure zonas_transporte_poligonos.csv exists in media/transport/."
                )
                return redirect("daedalus-transport-init")

            result = process_transport_pipeline(
                emt_zip=emt_zip,
                metro_zip=metro_zip,
                cercanias_zip=cerc_zip,
                zonas_poligonos_csv=zonas_csv,
            )

            return render(request, "daedalus/transport_init_upload.html", {
                "result": result,
                "uploads_present": True,
            })

    # GET
    uploads_present = all((uploads_dir / f).exists() for f in ["emt.zip", "metro.zip", "cercanias.zip"])
    return render(request, "daedalus/transport_init_upload.html", {
        "uploads_present": uploads_present,
    })

import pandas as pd

from .transport_engine import RouteCalculator, GeocodingService


@login_required
@require_http_methods(["GET", "POST"])
def transport_zones(request):
    base = Path(settings.MEDIA_ROOT) / "transport"
    zonas_csv = base / "zonas_transporte_poligonos.csv"
    outputs_dir = base / "outputs"

    required_outputs = [
        outputs_dir / "zonas_transporte_paradas.csv",
        outputs_dir / "arcos_caminando.csv",
        outputs_dir / "arcos_metro.csv",
        outputs_dir / "arcos_emt.csv",
        outputs_dir / "arcos_cercanias.csv",
    ]
    outputs_ready = all(p.exists() for p in required_outputs)

    if not zonas_csv.exists():
        messages.error(
            request,
            "Zones file not found: media/transport/zonas_transporte_poligonos.csv."
        )
        return redirect("daedalus-transport-init")

    df_z = pd.read_csv(zonas_csv)
    if "ZT1259" not in df_z.columns:
        messages.error(request, "Zones CSV does not contain 'ZT1259' column.")
        return redirect("daedalus-transport-init")

    zones = sorted(
        df_z["ZT1259"].dropna().unique().tolist(),
        key=lambda x: int(x) if str(x).isdigit() else str(x),
    )
    zones_str = set(map(str, zones))

    if request.method == "POST":
        mode = request.POST.get("mode", "zones")  # zones | places

        if mode == "places":
            origin_place = (request.POST.get("origin_place") or "").strip()
            dest_place = (request.POST.get("dest_place") or "").strip()

            if not origin_place or not dest_place:
                messages.error(request, "Write both origin and destination place names.")
                return redirect("daedalus-transport-zones")

            token = getattr(settings, "MAPBOX_TOKEN", None) or getattr(settings, "MAPBOX_API_KEY", None)
            if not token:
                messages.error(request, "Missing MAPBOX_TOKEN in settings.")
                return redirect("daedalus-transport-zones")

            calc = RouteCalculator(outputs_dir=outputs_dir, zonas_poligonos_csv=zonas_csv)
            geocoder = GeocodingService(token)

            o = geocoder.get_coordinates(origin_place)
            d = geocoder.get_coordinates(dest_place)
            if not o or not d:
                messages.error(request, "Mapbox couldn't find one of the places. Try a more specific name.")
                return redirect("daedalus-transport-zones")

            origin_zone = calc.get_zone_from_coords(*o)
            dest_zone = calc.get_zone_from_coords(*d)

            if not origin_zone or not dest_zone:
                messages.error(request, "Coordinates were found, but they don't fall inside any transport zone polygon.")
                return redirect("daedalus-transport-zones")

            # saves zones
            request.session["transport_origin_zone"] = str(origin_zone)
            request.session["transport_dest_zone"] = str(dest_zone)

            # 
            request.session["transport_origin_place"] = origin_place
            request.session["transport_dest_place"] = dest_place

            return redirect("daedalus-transport-results")

        # classic mode
        origin = request.POST.get("origin_zone")
        dest = request.POST.get("dest_zone")

        if str(origin) not in zones_str or str(dest) not in zones_str:
            messages.error(request, "Please select valid origin and destination zones.")
            return redirect("daedalus-transport-zones")

        request.session["transport_origin_zone"] = str(origin)
        request.session["transport_dest_zone"] = str(dest)

        # clean places
        request.session.pop("transport_origin_place", None)
        request.session.pop("transport_dest_place", None)

        return redirect("daedalus-transport-results")

    origin_default = request.session.get("transport_origin_zone", str(zones[0]) if zones else "")
    dest_default = request.session.get("transport_dest_zone", str(zones[0]) if zones else "")

    return render(
        request,
        "daedalus/transport_zones.html",
        {
            "zones": [str(z) for z in zones],
            "origin_default": origin_default,
            "dest_default": dest_default,
            "outputs_ready": outputs_ready,
            # prefill places
            "origin_place_default": request.session.get("transport_origin_place", ""),
            "dest_place_default": request.session.get("transport_dest_place", ""),
        },
    )
from pathlib import Path

import pandas as pd
from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.http import HttpResponse
from django.shortcuts import redirect, render
from django.views.decorators.http import require_http_methods

from .transport_engine import RouteCalculator


@login_required
@require_http_methods(["GET"])
def transport_results(request):
    origin_zone = request.session.get("transport_origin_zone")
    dest_zone = request.session.get("transport_dest_zone")

    # NEW: optional place names (when user searched by name)
    origin_place = request.session.get("transport_origin_place")
    dest_place = request.session.get("transport_dest_place")

    if not origin_zone or not dest_zone:
        return redirect("daedalus-transport-zones")

    outputs_dir = Path(settings.MEDIA_ROOT) / "transport" / "outputs"

    required = [
        outputs_dir / "zonas_transporte_paradas.csv",
        outputs_dir / "arcos_caminando.csv",
        outputs_dir / "arcos_metro.csv",
        outputs_dir / "arcos_emt.csv",
        outputs_dir / "arcos_cercanias.csv",
    ]
    missing = [p.name for p in required if not p.exists()]
    if missing:
        return HttpResponse(
            "Missing transport output files: " + ", ".join(missing)
            + ". Go to Transport Init and run Process GTFS first.",
            status=500,
        )

    # --- Load stops->zone mapping to show counts + stop names ---
    mapping_path = outputs_dir / "zonas_transporte_paradas.csv"
    map_df = pd.read_csv(mapping_path, dtype={"stop_id": str, "ZT1259": str})

    origin_stops_df = map_df[map_df["ZT1259"].astype(str) == str(origin_zone)]
    dest_stops_df = map_df[map_df["ZT1259"].astype(str) == str(dest_zone)]
    origin_stops_count = int(len(origin_stops_df))
    dest_stops_count = int(len(dest_stops_df))

    stop_name_by_id = dict(
        zip(
            map_df["stop_id"].astype(str),
            map_df.get("stop_name", pd.Series([""] * len(map_df))).astype(str),
        )
    )

    # 1) Build calculator (your real logic)
    calc = RouteCalculator(
        default_walk_transfer_min=8,
        transfer_matrix_min={
            "metro": {"metro": 8, "cercanias": 8, "emt": 12},
            "cercanias": {"metro": 8, "cercanias": 8, "emt": 12},
            "emt": {"metro": 8, "cercanias": 10, "emt": 12},
        },
    )

    # 2) Compute route
    try:
        result = calc.find_route_zones(str(origin_zone), str(dest_zone))
    except Exception as e:
        context = {
            "origin_zone": origin_zone,
            "dest_zone": dest_zone,
            "origin_place": origin_place,  # NEW
            "dest_place": dest_place,      # NEW
            "origin_stops_count": origin_stops_count,
            "dest_stops_count": dest_stops_count,
            "route": None,
            "route_error": f"RouteCalculator error: {e}",
        }
        return render(request, "daedalus/transport_results.html", context)

    if result is None:
        context = {
            "origin_zone": origin_zone,
            "dest_zone": dest_zone,
            "origin_place": origin_place,  # NEW
            "dest_place": dest_place,      # NEW
            "origin_stops_count": origin_stops_count,
            "dest_stops_count": dest_stops_count,
            "route": None,
            "route_error": None,
        }
        return render(request, "daedalus/transport_results.html", context)

    # 3) Normalize result to the template format
    steps_in = result.get("steps", []) or []
    total_time_s = float(result.get("total_time_s") or 0.0)

    steps_out = []
    transfers_count = 0

    for s in steps_in:
        is_transfer = bool(s.get("is_transfer", False))

        from_stop = s.get("from_stop") or s.get("from") or ""
        to_stop = s.get("to_stop") or s.get("to") or ""

        mode_from = s.get("mode_from") or s.get("mode") or ""
        mode_to = s.get("mode_to") or ""
        line_from = s.get("line_from") or s.get("linea") or s.get("line") or ""
        line_to = s.get("line_to") or ""

        time_s = float(s.get("time_s") or s.get("time_total_s") or s.get("time") or 0.0)

        if is_transfer:
            transfers_count += 1

        steps_out.append(
            {
                "is_transfer": is_transfer,
                "from_stop": str(from_stop),
                "to_stop": str(to_stop),
                "from_stop_name": stop_name_by_id.get(str(from_stop), ""),
                "to_stop_name": stop_name_by_id.get(str(to_stop), ""),
                "mode_from": str(mode_from),
                "mode_to": str(mode_to),
                "line_from": str(line_from),
                "line_to": str(line_to),
                "time_s": int(round(time_s)),
                "time_min": round(time_s / 60.0, 1),
            }
        )

    route = {
        "total_time_s": int(round(total_time_s)),
        "total_time_min": round(total_time_s / 60.0, 1),
        "steps": steps_out,
        "steps_count": len(steps_out),
        "transfers_count": transfers_count,
    }

    context = {
        "origin_zone": origin_zone,
        "dest_zone": dest_zone,
        "origin_place": origin_place,  # NEW
        "dest_place": dest_place,      # NEW
        "origin_stops_count": origin_stops_count,
        "dest_stops_count": dest_stops_count,
        "route": route,
        "route_error": None,
    }
    return render(request, "daedalus/transport_results.html", context)

@login_required
@require_http_methods(["GET", "POST"])
def transport_incidences(request):
    """
    Transport incidences admin:
      - GET: list incidences + show form
      - POST: create/activate or deactivate
    """
    outputs_dir = Path(settings.MEDIA_ROOT) / "transport" / "outputs"

    required = [
        outputs_dir / "zonas_transporte_paradas.csv",
        outputs_dir / "arcos_metro.csv",
        outputs_dir / "arcos_cercanias.csv",
    ]
    missing = [p.name for p in required if not p.exists()]
    if missing:
        return HttpResponse(
            "Missing transport output files: " + ", ".join(missing) +
            ". Go to Transport Init and run Process GTFS first.",
            status=500
        )

    calc = RouteCalculator(
        default_walk_transfer_min=8,
        transfer_matrix_min={
            "metro":     {"metro": 8, "cercanias": 8,  "emt": 12},
            "cercanias": {"metro": 8, "cercanias": 8,  "emt": 12},
            "emt":       {"metro": 8, "cercanias": 10, "emt": 12},
        },
    )

    message = None
    error = None

    if request.method == "POST":
        action = request.POST.get("action", "").strip()

        if action == "create_or_activate":
            incidencia_id = request.POST.get("incidencia_id", "").strip() or None
            modo = request.POST.get("modo", "").strip().lower()
            linea = request.POST.get("linea", "").strip()
            stop_desde = request.POST.get("stop_desde", "").strip()
            stop_hasta = request.POST.get("stop_hasta", "").strip()

            # Si no dan ID, autogenera (como tu CODIGO)
            if incidencia_id is None:
                incidencia_id = calc.generar_id()

            ok = calc.activar_incidencia(
                incidencia_id=incidencia_id,
                modo=modo,
                linea=linea,
                stop_desde=stop_desde,
                stop_hasta=stop_hasta,
            )
            if ok:
                message = f"Incidence {incidencia_id} activated."
            else:
                error = "Could not activate incidence. Check fields (mode/line/from/to)."

        elif action == "deactivate":
            incidencia_id = request.POST.get("incidencia_id", "").strip()
            ok = calc.desactivar_incidencia(incidencia_id)
            if ok:
                message = f"Incidence {incidencia_id} deactivated."
            else:
                error = f"Could not deactivate incidence {incidencia_id}."

        else:
            error = "Unknown action."

    # List incidences (active + all)
    df_all = calc.listar_incidencias(solo_activas=False)
    df_active = calc.listar_incidencias(solo_activas=True)

    incidences_all = df_all.fillna("").to_dict(orient="records") if not df_all.empty else []
    incidences_active = df_active.fillna("").to_dict(orient="records") if not df_active.empty else []

    context = {
        "message": message,
        "error": error,
        "incidences_active": incidences_active,
        "incidences_all": incidences_all,
    }
    return render(request, "daedalus/transport_incidences.html", context)

