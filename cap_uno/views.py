from django.shortcuts import render
from django.http import HttpResponse
from django.contrib import messages
from math import isclose
from .forms import BiseccionForm
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
                # Permitir funciones matemáticas seguras
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