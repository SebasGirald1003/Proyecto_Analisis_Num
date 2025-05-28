from django.urls import path
from . import views

urlpatterns = [
    path("lagrange/", views.lagrange_interpolation_view, name="lagrange"),
    path("newton-interpolante/", views.newton_interpolation_view, name="newton-interpolante"),
    path("vandermonde/", views.vandermonde_interpolation_view, name="vandermonde"),
    path("spline-lineal/", views.spline_lineal_interpolation_view, name="spline-lineal"),
    path("spline-cubico/", views.spline_cubico_interpolation_view, name="spline-cubico"),
]