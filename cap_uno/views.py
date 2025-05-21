from django.shortcuts import render
from django.http import HttpResponse
from django.contrib import messages
from math import isclose
from .forms import BiseccionForm, PuntoFijoForm
import math

def index(request):
    return render(request, 'home.html')

def biseccion_view(request):
    tabla = None
    raiz = None
    if request.method == 'POST':
        form = BiseccionForm(request.POST)
        if form.is_valid():
            f_str    = form.cleaned_data['funcion']
            a        = form.cleaned_data['a']
            b        = form.cleaned_data['b']
            tol      = form.cleaned_data['tolerancia']
            max_iter = form.cleaned_data['max_iter']
            try:
                allowed_names = {k: getattr(math, k) for k in dir(math) if not k.startswith("__")}
                f = lambda x: eval(f_str, {"__builtins__": {}}, {**allowed_names, 'x': x})
            except Exception as e:
                messages.error(request, f"Error en la función: {e}")
            else:
                if f(a) * f(b) > 0:
                    messages.error(request, "f(a) y f(b) deben tener signos opuestos.")
                else:
                    tabla = []
                    for i in range(1, max_iter + 1):
                        xm = (a + b) / 2.0
                        fxm = f(xm)
                        error = abs(b - a) / 2.0
                        tabla.append((i, a, xm, b, fxm, error))
                        if isclose(fxm, 0, abs_tol=tol) or error < tol:
                            raiz = xm
                            break
                        if f(a) * fxm < 0:
                            b = xm
                        else:
                            a = xm
    else:
        form = BiseccionForm()
    
    return render(request, 'biseccion.html', {
        'form': form,
        'tabla': tabla,
        'raiz': raiz
    })

def regla_falsa_view(request):
    tabla = None
    raiz = None
    if request.method == 'POST':
        form = BiseccionForm(request.POST)
        if form.is_valid():
            f_str    = form.cleaned_data['funcion']
            a        = form.cleaned_data['a']
            b        = form.cleaned_data['b']
            tol      = form.cleaned_data['tolerancia']
            max_iter = form.cleaned_data['max_iter']
            try:
                allowed_names = {k: getattr(math, k) for k in dir(math) if not k.startswith("__")}
                f = lambda x: eval(f_str, {"__builtins__": {}}, {**allowed_names, 'x': x})
            except Exception as e:
                messages.error(request, f"Error en la función: {e}")
            else:
                if f(a) * f(b) > 0:
                    messages.error(request, "f(a) y f(b) deben tener signos opuestos.")
                else:
                    tabla = []
                    for i in range(1, max_iter + 1):
                        fa = f(a)
                        fb = f(b)
                        xm = b - (fb * (a - b)) / (fa - fb)  # Regla falsa
                        fxm = f(xm)
                        error = abs(fxm)
                        tabla.append((i, a, xm, b, fxm, error))  
                        if isclose(fxm, 0, abs_tol=tol) or abs(b - a) < tol:
                            raiz = xm
                            break
                        if fa * fxm < 0:
                            b = xm
                        else:
                            a = xm
    else:
        form = BiseccionForm()
    
    return render(request, 'regla_falsa.html', {
        'form': form,
        'tabla': tabla,
        'raiz': raiz
    })

def punto_fijo_view(request):
    tabla = None
    raiz = None
    if request.method == 'POST':
        form = PuntoFijoForm(request.POST)
        if form.is_valid():
            f_str     = form.cleaned_data['funcion_f']
            g_str     = form.cleaned_data['funcion_g']
            x0        = form.cleaned_data['x0']
            tol       = form.cleaned_data['tolerancia']
            max_iter  = form.cleaned_data['max_iter']
            try:
                allowed_names = {k: getattr(math, k) for k in dir(math) if not k.startswith("__")}
                f = lambda x: eval(f_str, {"__builtins__": {}}, {**allowed_names, 'x': x})
                g = lambda x: eval(g_str, {"__builtins__": {}}, {**allowed_names, 'x': x})
            except Exception as e:
                messages.error(request, f"Error en las funciones: {e}")
            else:
                tabla = []
                xi = x0
                for i in range(1, max_iter + 1):
                    gxi = g(xi)
                    fxi = f(gxi)
                    error = abs(gxi - xi)
                    tabla.append((
                        i,
                        round(xi, 6),
                        round(gxi, 6),
                        round(fxi, 6),
                        round(error, 6)
                    ))
                    if error < tol:
                        raiz = round(gxi, 6)
                        break
                    xi = gxi
    else:
        form = PuntoFijoForm()
    
    return render(request, 'punto_fijo.html', {
        'form': form,
        'tabla': tabla,
        'raiz': raiz
    })