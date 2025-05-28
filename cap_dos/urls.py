from django.urls import path
from . import views

urlpatterns = [
    path('gauss-elimination/', views.eliminacion_gaussiana_simple_view, name='eliminacion_gaussiana_simple'),
    path('gauss-pivoteo-parcial/', views.eliminacion_gaussiana_pivoteo_parcial, name='gauss_pivoteo'),
    path('gauss-pivoteo-total/', views.eliminacion_gaussiana_pivoteo_total, name='gauss_pivoteo_total'),
    path('factorizacion-lu/', views.factorizacion_LU_simple, name='factorizacion_lu'),
    path('factorizacion-lu-parcial/', views.factorizacion_LU_parcial, name='factorizacion_lu_parcial'),
    path('crout/', views.croult, name='crout_method'),
    path('doolittle/', views.doolittle, name='doolittle_method'),
    path('cholesky/', views.cholesky, name='cholesky_method'),
    path('iterative-methods/', views.metodo_iterativo_Jacobi_Gauss_Seidel_SOR, name='iterative_methods'),
]