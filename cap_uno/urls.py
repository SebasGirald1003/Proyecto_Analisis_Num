from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='home'),
    path('biseccion/', views.biseccion_view, name='biseccion'),
    path('regla-falsa/', views.regla_falsa_view, name='regla_falsa'),
]