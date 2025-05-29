from django.shortcuts import render
from django.http import HttpResponse
from django.contrib import messages
from math import isclose
from .forms import BiseccionForm, PuntoFijoForm, NewtonForm, SecanteForm, RaicesMultiplesForm
import math
from sympy import symbols, diff, sympify, lambdify
import matplotlib.pyplot as plt
import numpy as np
import io
import base64

def index(request):
    return render(request, 'home.html')

def graficar_funcion(f, a, b):
    x = np.linspace(a, b, 400)
    y = [f(xi) for xi in x]

    fig, ax = plt.subplots()
    ax.plot(x, y, label='f(x)', color='blue')
    ax.axhline(0, color='black', linewidth=0.5)
    ax.grid(True)
    ax.legend()

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close()

    return base64.b64encode(image_png).decode('utf-8')

def biseccion_view(request):
    tabla = None
    raiz = None
    grafico = None
    f_str = None   
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
                    grafico = graficar_funcion(f, form.cleaned_data['a'], form.cleaned_data['b'])  
    else:
        form = BiseccionForm()
    
    return render(request, 'biseccion.html', {
        'form': form,
        'tabla': tabla,
        'raiz': raiz,
        'grafico': grafico,
        'funcion': f_str, 
    })


def regla_falsa_view(request):
    tabla = None
    raiz = None
    grafico = None 
    f_str = None 
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
                        xm = b - (fb * (a - b)) / (fa - fb)
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
                    grafico = graficar_funcion(f, form.cleaned_data['a'], form.cleaned_data['b'])  
    else:
        form = BiseccionForm()
    
    return render(request, 'regla_falsa.html', {
        'form': form,
        'tabla': tabla,
        'raiz': raiz,
        'grafico': grafico,
        'funcion': f_str,
    })


def punto_fijo_view(request):
    tabla = None
    raiz = None
    grafico_f = None
    grafico_g = None
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
                
                rango_min = x0 - 5
                rango_max = x0 + 5
                grafico_f = graficar_funcion(f, rango_min, rango_max)
                grafico_g = graficar_funcion(g, rango_min, rango_max)

    else:
        form = PuntoFijoForm()

    return render(request, 'punto_fijo.html', {
        'form': form,
        'tabla': tabla,
        'raiz': raiz,
        'grafico_f': grafico_f,
        'grafico_g': grafico_g
    })


def newton_view(request):
    tabla = None
    raiz = None
    df_str = None
    grafica_f = None
    grafica_df = None

    if request.method == 'POST':
        form = NewtonForm(request.POST)
        if form.is_valid():
            f_str = form.cleaned_data['funcion']
            x0 = form.cleaned_data['x0']
            tol = form.cleaned_data['tolerancia']
            max_iter = form.cleaned_data['max_iter']

            try:
                x = symbols('x')
                f_expr = sympify(f_str)
                df_expr = diff(f_expr, x)
                df_str = str(df_expr)

                f = lambdify(x, f_expr, modules=["math"])
                df = lambdify(x, df_expr, modules=["math"])

                grafica_f = graficar_funcion(f, x0 - 5, x0 + 5)
                grafica_df = graficar_funcion(df, x0 - 5, x0 + 5)

            except Exception as e:
                messages.error(request, f"Error en las funciones: {e}")
            else:
                tabla = []
                xi = x0
                for i in range(1, max_iter + 1):
                    fxi = f(xi)
                    dfxi = df(xi)
                    if dfxi == 0:
                        messages.error(request, "La derivada se volvió cero, no se puede continuar.")
                        break
                    xi_next = xi - fxi / dfxi
                    error = abs(xi_next - xi)
                    tabla.append((i, round(xi, 6), round(fxi, 6), round(error, 6)))
                    if error < tol:
                        raiz = round(xi_next, 6)
                        break
                    xi = xi_next
    else:
        form = NewtonForm()

    return render(request, 'newton.html', {
        'form': form,
        'tabla': tabla,
        'raiz': raiz,
        'derivada_str': df_str,
        'grafica_f': grafica_f,
        'grafica_df': grafica_df
    })

def secante_view(request):
    tabla = None
    raiz = None
    grafica_f = None  

    if request.method == 'POST':
        form = SecanteForm(request.POST)
        if form.is_valid():
            f_str = form.cleaned_data['funcion']
            x0 = form.cleaned_data['x0']
            x1 = form.cleaned_data['x1']
            tol = form.cleaned_data['tolerancia']
            max_iter = form.cleaned_data['max_iter']

            try:
                allowed_names = {k: getattr(math, k) for k in dir(math) if not k.startswith("__")}
                f = lambda x: eval(f_str, {"__builtins__": {}}, {**allowed_names, 'x': x})
                grafica_f = graficar_funcion(f, min(x0, x1) - 5, max(x0, x1) + 5)  
            except Exception as e:
                messages.error(request, f"Error en la función: {e}")
            else:
                tabla = []
                for i in range(1, max_iter + 1):
                    fx0 = f(x0)
                    fx1 = f(x1)
                    if fx1 - fx0 == 0:
                        messages.error(request, "División por cero en la iteración. No se puede continuar.")
                        break
                    xi = x1 - fx1 * (x1 - x0) / (fx1 - fx0)
                    fxi = f(xi)
                    error = abs(xi - x1)
                    tabla.append((i, round(xi, 6), round(fxi, 6), round(error, 6)))
                    if error < tol:
                        raiz = round(xi, 6)
                        break
                    x0, x1 = x1, xi
    else:
        form = SecanteForm()

    return render(request, 'secante.html', {
        'form': form,
        'tabla': tabla,
        'raiz': raiz,
        'grafica_f': grafica_f  
    })


def raices_multiples_view(request):
    tabla = None
    raiz = None
    grafica_f = grafica_df = grafica_ddf = None  
    df_str = ddf_str = ''  

    if request.method == 'POST':
        form = RaicesMultiplesForm(request.POST)
        if form.is_valid():
            f_str = form.cleaned_data['funcion']
            x0 = form.cleaned_data['x0']
            tol = form.cleaned_data['tolerancia']
            max_iter = form.cleaned_data['max_iter']

            try:
                x = symbols('x')
                f_expr = sympify(f_str)
                df_expr = diff(f_expr, x)
                ddf_expr = diff(df_expr, x)

                df_str = str(df_expr)
                ddf_str = str(ddf_expr)

                f = lambdify(x, f_expr, modules=['math'])
                df = lambdify(x, df_expr, modules=['math'])
                ddf = lambdify(x, ddf_expr, modules=['math'])

                a, b = x0 - 5, x0 + 5
                grafica_f = graficar_funcion(f, a, b)
                grafica_df = graficar_funcion(df, a, b)
                grafica_ddf = graficar_funcion(ddf, a, b)

            except Exception as e:
                messages.error(request, f"Error al procesar funciones: {e}")
            else:
                tabla = []
                n = 1
                while n <= max_iter:
                    fx0 = f(x0)
                    dfx0 = df(x0)
                    ddfx0 = ddf(x0)

                    denominador = dfx0**2 - fx0 * ddfx0
                    if denominador == 0:
                        messages.error(request, "División por cero detectada en el denominador.")
                        break

                    xn = x0 - (fx0 * dfx0) / denominador
                    fxn = f(xn)
                    error = abs(xn - x0)

                    tabla.append((n, round(xn, 6), round(fxn, 6), round(error, 6)))

                    if abs(fxn) == 0 or error < tol:
                        raiz = round(xn, 6)
                        break

                    x0 = xn
                    n += 1

                if raiz is None:
                    raiz = round(xn, 6)
                    messages.warning(request, "Se alcanzó el número máximo de iteraciones.")

    else:
        form = RaicesMultiplesForm()

    return render(request, 'raices_multiples.html', {
        'form': form,
        'tabla': tabla,
        'raiz': raiz,
        'grafica_f': grafica_f,
        'grafica_df': grafica_df,
        'grafica_ddf': grafica_ddf,
        'df_str': df_str,
        'ddf_str': ddf_str,
    })