from django.shortcuts import render
import numpy as np
from scipy.interpolate import CubicSpline

def lagrange_interpolation_view(request):
    context = {}
    
    if request.method == 'POST':
        x_input = request.POST.get("x", "")
        y_input = request.POST.get("y", "")

        try:
            x_values = list(map(float, x_input.strip().split()))
            y_values = list(map(float, y_input.strip().split()))

            if len(x_values) != len(y_values):
                raise ValueError("Las listas deben tener la misma longitud.")

            #Función para calcular el polinomio de Lagrange
            def lagrange_polynomial(x_vals, y_vals):
                def basis(j):
                    p = np.poly1d([1.0])
                    for m in range(len(x_vals)):
                        if m != j:
                            p *= np.poly1d([1.0, -x_vals[m]]) / (x_vals[j] - x_vals[m])
                    return p
                polynomial = sum(y_vals[j] * basis(j) for j in range(len(x_vals)))
                return polynomial

            # Función para dar formato al polinomio
            def format_polynomial_python_style(poly):
                coeffs = poly.coefficients
                degree = len(coeffs) - 1
                terms = []

                for i, coef in enumerate(coeffs):
                    power = degree - i
                    if abs(coef) < 1e-12:
                        continue
                    sign = "+" if coef >= 0 else "-"
                    coef_str = f"{abs(coef)}"
                    if power == 0:
                        term = f"{sign}{coef_str}"
                    elif power == 1:
                        term = f"{sign}{coef_str}*x**1"
                    else:
                        term = f"{sign}{coef_str}*x**{power}"
                    terms.append(term)
                return " ".join(terms) if terms else "0"

            P = lagrange_polynomial(x_values, y_values)
            polynomial_str = format_polynomial_python_style(P)

            context["polynomial"] = polynomial_str
            context["success"] = True

        except Exception as e:
            context["error"] = f"Error en la entrada o cálculo: {str(e)}"
    
    return render(request, "lagrange.html", context)

def newton_interpolation_view(request):
    context = {}

    if request.method == 'POST':
        x_input = request.POST.get("x", "")
        y_input = request.POST.get("y", "")

        try:
            x_values = list(map(float, x_input.strip().split()))
            y_values = list(map(float, y_input.strip().split()))

            if len(x_values) != len(y_values):
                raise ValueError("Las listas deben tener la misma longitud.")

            n = len(x_values)
            coef = np.copy(y_values)

            # Cálculo de coeficientes con diferencias divididas
            for j in range(1, n):
                for i in range(n - 1, j - 1, -1):
                    coef[i] = (coef[i] - coef[i - 1]) / (x_values[i] - x_values[i - j])

            # Construcción del polinomio expandido
            poly = np.poly1d([0.0])
            for i in range(n):
                term = np.poly1d([1.0])
                for j in range(i):
                    term *= np.poly1d([1.0, -x_values[j]])
                term *= coef[i]
                poly += term

            # Formateo en estilo Python puro
            def format_polynomial_python_style(poly):
                coeffs = poly.coefficients
                degree = len(coeffs) - 1
                terms = []

                for i, coef in enumerate(coeffs):
                    power = degree - i
                    if abs(coef) < 1e-12:
                        continue
                    sign = "+" if coef >= 0 else "-"
                    coef_str = f"{abs(coef)}"
                    if power == 0:
                        term = f"{sign}{coef_str}"
                    elif power == 1:
                        term = f"{sign}{coef_str}*x**1"
                    else:
                        term = f"{sign}{coef_str}*x**{power}"
                    terms.append(term)
                return " ".join(terms) if terms else "0"

            polynomial_str = format_polynomial_python_style(poly)

            context["polynomial"] = polynomial_str
            context["success"] = True

        except Exception as e:
            context["error"] = f"Error en la entrada o cálculo: {str(e)}"

    return render(request, "interp_newton.html", context)

def vandermonde_interpolation_view(request):
    context = {}

    if request.method == 'POST':
        x_input = request.POST.get("x", "")
        y_input = request.POST.get("y", "")

        try:
            x_values = list(map(float, x_input.strip().split()))
            y_values = list(map(float, y_input.strip().split()))

            if len(x_values) != len(y_values):
                raise ValueError("Las listas deben tener la misma longitud.")

            # Matriz de Vandermonde
            V = np.vander(x_values, increasing=False)
            y = np.array(y_values)

            # Resolver sistema V·a = y
            coef = np.linalg.solve(V, y)

            # Crear polinomio con coeficientes
            poly = np.poly1d(coef)

            # Formatear salida en estilo Python
            def format_polynomial_python_style(poly):
                coeffs = poly.coefficients
                degree = len(coeffs) - 1
                terms = []

                for i, coef in enumerate(coeffs):
                    power = degree - i
                    if abs(coef) < 1e-12:
                        continue
                    sign = "+" if coef >= 0 else "-"
                    coef_str = f"{abs(coef)}"
                    if power == 0:
                        term = f"{sign}{coef_str}"
                    elif power == 1:
                        term = f"{sign}{coef_str}*x**1"
                    else:
                        term = f"{sign}{coef_str}*x**{power}"
                    terms.append(term)
                return " ".join(terms) if terms else "0"

            polynomial_str = format_polynomial_python_style(poly)

            context["polynomial"] = polynomial_str
            context["success"] = True

        except Exception as e:
            context["error"] = f"Error en la entrada o cálculo: {str(e)}"

    return render(request, "vandermonde.html", context)

def spline_lineal_interpolation_view(request):
    context = {}

    if request.method == 'POST':
        x_input = request.POST.get("x", "")
        y_input = request.POST.get("y", "")

        try:
            x_values = list(map(float, x_input.strip().split()))
            y_values = list(map(float, y_input.strip().split()))

            if len(x_values) != len(y_values):
                raise ValueError("Las listas deben tener la misma longitud.")
            if len(x_values) < 2:
                raise ValueError("Se requieren al menos dos puntos.")

            # Aseguramos orden ascendente
            x_values, y_values = zip(*sorted(zip(x_values, y_values)))
            n = len(x_values)

            tramos = []

            for i in range(n - 1):
                x0, x1 = x_values[i], x_values[i + 1]
                y0, y1 = y_values[i], y_values[i + 1]

                # Coeficientes del tramo: y = m*x + b
                m = (y1 - y0) / (x1 - x0)
                b = y0 - m * x0

                # Formato estilo Python puro
                sign_b = "+" if b >= 0 else "-"
                b_str = f"{abs(b)}"
                sign_m = "+" if m >= 0 else "-"
                m_str = f"{abs(m)}"

                expr = f"{sign_m} {m_str}*x {sign_b}{b_str}"
                tramos.append(expr)

            context["splines"] = tramos
            context["success"] = True

        except Exception as e:
            context["error"] = f"Error en la entrada o cálculo: {str(e)}"

    return render(request, "spline_lineal.html", context)

def spline_cubico_interpolation_view(request):
    context = {}

    if request.method == 'POST':
        x_input = request.POST.get("x", "")
        y_input = request.POST.get("y", "")

        try:
            x_values = list(map(float, x_input.strip().split()))
            y_values = list(map(float, y_input.strip().split()))

            if len(x_values) != len(y_values):
                raise ValueError("Las listas deben tener la misma longitud.")
            if len(x_values) < 3:
                raise ValueError("Se requieren al menos tres puntos para un spline cúbico.")

            # Ordenar por x
            x_values, y_values = zip(*sorted(zip(x_values, y_values)))

            # Calcular el spline cúbico natural
            cs = CubicSpline(x_values, y_values, bc_type='natural')

            tramos = []
            for i in range(len(x_values) - 1):
                x0 = x_values[i]
                coefs = cs.c[:, i]  # coeficientes del tramo i

                terms = []
                powers = [3, 2, 1, 0]
                for coef, power in zip(coefs, powers):
                    if abs(coef) < 1e-12:
                        continue
                    sign = "+" if coef >= 0 else "-"
                    coef_str = f"{abs(coef)}"
                    if power == 0:
                        term = f"{sign} {coef_str}"
                    else:
                        term = f"{sign} {coef_str}*(x - {x0})**{power}"
                    terms.append(term)

                expression = " ".join(terms)
                tramos.append(expression)

            context["splines"] = tramos
            context["success"] = True

        except Exception as e:
            context["error"] = f"Error en la entrada o cálculo: {str(e)}"

    return render(request, "spline_cubico.html", context)