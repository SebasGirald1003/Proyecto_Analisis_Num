from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='home'),
    path('biseccion/', views.biseccion_view, name='biseccion'),
    path('regla-falsa/', views.regla_falsa_view, name='regla_falsa'),
    path('punto-fijo/', views.punto_fijo_view, name='punto_fijo'),
    path('newton/', views.newton_view, name='newton'),
    path('secante/', views.secante_view, name='secante'),
    path('raices-multiples/', views.raices_multiples_view, name='raices_multiples'),
]