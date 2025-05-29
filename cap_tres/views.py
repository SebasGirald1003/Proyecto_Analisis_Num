from django.shortcuts import render
import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import io
import base64

def plot_polynomial_expression(expr, x_vals, y_vals):
    try:
        # Crear función a partir de la expresión
        f = lambda x: eval(expr.replace("^", "**"))

        # Rango de graficación
        x_min, x_max = min(x_vals), max(x_vals)
        x_plot = np.linspace(x_min - 1, x_max + 1, 400)
        y_plot = [f(x) for x in x_plot]

        # Graficar
        fig, ax = plt.subplots()
        ax.plot(x_plot, y_plot, label="Polinomio", color="blue")
        ax.scatter(x_vals, y_vals, color="red", label="Puntos")
        ax.legend()
        ax.set_title("Interpolación Polinómica")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(True)

        # Convertir a base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        return image_base64

    except Exception as e:
        return None

def plot_piecewise_functions(splines, x_vals, y_vals):
    try:
        fig, ax = plt.subplots()
        x_vals_sorted, y_vals_sorted = zip(*sorted(zip(x_vals, y_vals)))

        for i in range(len(splines)):
            x0 = x_vals_sorted[i]
            x1 = x_vals_sorted[i + 1]
            expr = splines[i].replace("^", "**")

            f = lambda x: eval(expr.replace("x", f"(x)"))  # para asegurar precedencia

            x_plot = np.linspace(x0, x1, 100)
            y_plot = [f(x) for x in x_plot]
            ax.plot(x_plot, y_plot, label=f"Tramo {i+1}")

        ax.scatter(x_vals, y_vals, color='red', label='Puntos')
        ax.legend()
        ax.set_title("Interpolación por Splines")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(True)

        # Convertir a base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        return image_base64

    except Exception as e:
        return None

def validate_xy_lists(x_values, y_values):
    if not x_values or not y_values:
        raise ValueError("Las listas de x o y están vacías.")

    if len(x_values) < 2 or len(y_values) < 2:
        raise ValueError("Se requieren al menos dos puntos para interpolar.")

    if len(x_values) != len(y_values):
        raise ValueError("Las listas de x e y deben tener la misma longitud.")

    if any(not isinstance(x, (int, float)) or not np.isfinite(x) for x in x_values):
        raise ValueError("Todos los elementos en x deben ser números finitos.")

    if any(not isinstance(y, (int, float)) or not np.isfinite(y) for y in y_values):
        raise ValueError("Todos los elementos en y deben ser números finitos.")

    if len(set(x_values)) != len(x_values):
        raise ValueError("Hay valores de x repetidos, lo cual no es válido para la interpolación.")
    
    for i, val in enumerate(x_values):
        try:
            x = float(val)
            if not np.isfinite(x):
                raise ValueError
        except:
            raise ValueError(f"El valor en x[{i}] = '{val}' no es un número válido.")

    for i, val in enumerate(y_values):
        try:
            y = float(val)
            if not np.isfinite(y):
                raise ValueError
        except:
            raise ValueError(f"El valor en y[{i}] = '{val}' no es un número válido.")



def lagrange_interpolation_view(request):
    context = {}
    
    if request.method == 'POST':
        x_input = request.POST.get("x", "")
        y_input = request.POST.get("y", "")

        try:
            try:
                x_values = list(map(float, x_input.strip().split()))
                y_values = list(map(float, y_input.strip().split()))
            except ValueError:
                raise ValueError("Los valores de x e y deben ser números separados por espacios.")

            if len(x_values) > 12:
                context["warning"] = "Advertencia: interpolar con muchos puntos puede causar oscilaciones indeseadas."

            # Validación general
            validate_xy_lists(x_values, y_values)

            #Función para calcular el polinomio de Lagrange
            def lagrange_polynomial(x_vals, y_vals):
                def basis(j):
                    p = np.poly1d([1.0])
                    for m in range(len(x_vals)):
                        if m != j:
                            denom = x_vals[j] - x_vals[m]
                            if abs(denom) < 1e-12:
                                raise ZeroDivisionError("Dos valores de x son demasiado cercanos, lo que puede causar errores numéricos.")
                            p *= np.poly1d([1.0, -x_vals[m]]) / denom
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
            context["image"] = plot_polynomial_expression(polynomial_str.replace("**", "^"), x_values, y_values)

        except Exception as e:
            context["error"] = f"Error en la entrada o cálculo: {str(e)}"
    
    return render(request, "lagrange.html", context)


def newton_interpolation_view(request):
    context = {}

    if request.method == 'POST':
        x_input = request.POST.get("x", "")
        y_input = request.POST.get("y", "")

        try:
            try:
                x_values = list(map(float, x_input.strip().split()))
                y_values = list(map(float, y_input.strip().split()))
            except ValueError:
                raise ValueError("Los valores de x e y deben ser números separados por espacios.")

            validate_xy_lists(x_values, y_values)

            if len(x_values) > 12:
                context["warning"] = "Advertencia: interpolar con muchos puntos puede causar oscilaciones indeseadas."

            n = len(x_values)
            coef = np.copy(y_values)

            for j in range(1, n):
                for i in range(n - 1, j - 1, -1):
                    denominador = x_values[i] - x_values[i - j]
                    if abs(denominador) < 1e-12:
                        raise ZeroDivisionError("Se detectó una división por cero en los coeficientes.")
                    coef[i] = (coef[i] - coef[i - 1]) / denominador

            poly = np.poly1d([0.0])
            for i in range(n):
                term = np.poly1d([1.0])
                for j in range(i):
                    term *= np.poly1d([1.0, -x_values[j]])
                term *= coef[i]
                poly += term

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
            context["image"] = plot_polynomial_expression(polynomial_str.replace("**", "^"), x_values, y_values)

        except ValueError as ve:
            context["error"] = f"Error de entrada: {str(ve)}"
        except ZeroDivisionError as zde:
            context["error"] = f"Error matemático: {str(zde)}"
        except Exception as e:
            context["error"] = f"Error inesperado: {str(e)}"

    return render(request, "interp_newton.html", context)


def vandermonde_interpolation_view(request):
    context = {}

    if request.method == 'POST':
        x_input = request.POST.get("x", "")
        y_input = request.POST.get("y", "")

        try:
            try:
                x_values = list(map(float, x_input.strip().split()))
                y_values = list(map(float, y_input.strip().split()))
            except ValueError:
                raise ValueError("Los valores de x e y deben ser números separados por espacios.")

            if len(x_values) > 12:
                context["warning"] = "Advertencia: interpolar con muchos puntos puede causar oscilaciones indeseadas."

            if len(set(x_values)) != len(x_values):
                raise np.linalg.LinAlgError("La matriz de Vandermonde no es invertible porque hay valores de X repetidos.")
            
            validate_xy_lists(x_values, y_values)

            V = np.vander(x_values, increasing=False)
            y = np.array(y_values)
            coef = np.linalg.solve(V, y)
            poly = np.poly1d(coef)

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
            context["image"] = plot_polynomial_expression(polynomial_str.replace("**", "^"), x_values, y_values)

        except ValueError as ve:
            context["error"] = f"Error de entrada: {str(ve)}"
        except np.linalg.LinAlgError as lae:
            context["error"] = f"Error matemático: {str(lae)}"
        except Exception as e:
            context["error"] = f"Error inesperado: {str(e)}"

    return render(request, "vandermonde.html", context)


def spline_lineal_interpolation_view(request):
    context = {}

    if request.method == 'POST':
        x_input = request.POST.get("x", "")
        y_input = request.POST.get("y", "")

        try:
            try:
                x_values = list(map(float, x_input.strip().split()))
                y_values = list(map(float, y_input.strip().split()))
            except ValueError:
                raise ValueError("Los valores de x e y deben ser números separados por espacios.")

            validate_xy_lists(x_values, y_values)

            x_values, y_values = zip(*sorted(zip(x_values, y_values)))
            n = len(x_values)

            tramos = []

            for i in range(n - 1):
                x0, x1 = x_values[i], x_values[i + 1]
                y0, y1 = y_values[i], y_values[i + 1]

                if x1 == x0:
                    raise ValueError(f"Los valores x[{i}] y x[{i+1}] son iguales, lo que genera una división por cero.")

                m = (y1 - y0) / (x1 - x0)
                b = y0 - m * x0

                sign_b = "+" if b >= 0 else "-"
                b_str = f"{abs(b)}"
                sign_m = "+" if m >= 0 else "-"
                m_str = f"{abs(m)}"

                expr = f"{sign_m}{m_str}*x {sign_b}{b_str}"
                tramos.append(expr)

            context["splines"] = tramos
            context["success"] = True
            context["image"] = plot_piecewise_functions(tramos, x_values, y_values)

        except ValueError as ve:
            context["error"] = f"Error de entrada: {str(ve)}"
        except Exception as e:
            context["error"] = f"Error inesperado: {str(e)}"

    return render(request, "spline_lineal.html", context)

def spline_cubico_interpolation_view(request):
    context = {}

    if request.method == 'POST':
        x_input = request.POST.get("x", "")
        y_input = request.POST.get("y", "")

        try:
            try:
                x_values = list(map(float, x_input.strip().split()))
                y_values = list(map(float, y_input.strip().split()))
            except ValueError:
                raise ValueError("Los valores de x e y deben ser números separados por espacios.")

            if len(x_values) < 3:
                raise ValueError("Se requieren al menos tres puntos para calcular un spline cúbico.")

            validate_xy_lists(x_values, y_values)

            x_values, y_values = zip(*sorted(zip(x_values, y_values)))

            cs = CubicSpline(x_values, y_values, bc_type='natural')

            tramos = []
            for i in range(len(x_values) - 1):
                x0 = x_values[i]
                coefs = cs.c[:, i] 

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
            context["image"] = plot_piecewise_functions(tramos, x_values, y_values)

        except ValueError as ve:
            context["error"] = f"Error de entrada: {str(ve)}"
        except Exception as e:
            context["error"] = f"Error inesperado: {str(e)}"

    return render(request, "spline_cubico.html", context)
