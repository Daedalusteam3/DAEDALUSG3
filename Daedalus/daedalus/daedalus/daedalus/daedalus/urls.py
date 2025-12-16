from django.urls import path
from . import views
from django.contrib.auth import views as auth_views

# Ver https://www.bookaapi.com/book-blogging/url-structure/

# daedalus/urls.py

#app_name = 'daedalus'

print("[DEBUG] Importando daedalus/urls.py", flush=True)

urlpatterns = [
    path('', views.home, name='daedalus-home'),
    path("login/",auth_views.LoginView.as_view(template_name="daedalus/login.html"),name="daedalus-login"),
    path("logout/",auth_views.LogoutView.as_view(next_page="daedalus-home"),name="daedalus-logout"),
    path('runs/', views.runs_view, name='daedalus-runs'),
    path('dynamic/', views.dynamic_map, name='daedalus-dynamic'),
    path('run-opt/', views.run_opt_view, name='daedalus-run-opt'),  
    path('dashboard/', views.dashboard, name='daedalus-dashboard'),
    path("dashboard1/", views.dashboard1, name="daedalus-dashboard1"),
    path("hotspots/", views.hotspots_map, name="daedalus-hotspots"),
    path("ads/", views.ads_panel, name="daedalus-ads"),
    path("ads-map/", views.ads_map, name="daedalus-ads-map"),
    path("api/ads-map-data/", views.ads_map_data, name="daedalus-ads-map-data"),
    path("ads-admin/", views.ads_admin, name="daedalus-ads-admin"),
    path("transport/init/", views.transport_init_upload, name="daedalus-transport-init"),
    path("transport/zones/", views.transport_zones, name="daedalus-transport-zones"),
    path("transport/results/", views.transport_results, name="daedalus-transport-results"),
    path("transport/incidences/", views.transport_incidences, name="daedalus-transport-incidences"),
]

