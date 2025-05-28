import numpy as np
from math import sqrt
from scipy.linalg import eig
from django.shortcuts import render
from .forms import GaussEliminationForm, LUDecompositionForm, LUPivoteoParcialForm, CroutForm, DoolittleForm, CholeskyForm, IterativeMethodsForm

def eliminacion_gaussiana_simple_view(request):
    if request.method == 'POST':
        form = GaussEliminationForm(request.POST)
        if form.is_valid():
            # Parsear la matriz A y el vector B
            try:
                matrix_a = [
                    [float(item) for item in row.split(',')] 
                    for row in form.cleaned_data['matrix_a'].split(';')
                ]
                vector_b = [
                    [float(item)] 
                    for item in form.cleaned_data['vector_b'].split(',')
                ]
                
                # Validaciones iniciales
                m = len(matrix_a)
                n = len(matrix_a[0]) if m > 0 else 0
                
                if m != n:
                    raise ValueError("La matriz no es cuadrada")
                if m != len(vector_b):
                    raise ValueError("El vector B tiene dimensión diferente")
                
                # Convertir a numpy array para facilitar operaciones
                M = np.array([row.copy() for row in matrix_a], dtype=float)
                B = np.array(vector_b, dtype=float)
                
                # Crear matriz aumentada
                augmented_matrix = np.hstack((M, B))
                
                # Proceso de eliminación gaussiana
                iterations = []
                iterations.append(augmented_matrix.copy())
                
                n = len(augmented_matrix)
                
                for i in range(n-1):
                    # Pivoteo parcial si el elemento diagonal es cero
                    if augmented_matrix[i, i] == 0:
                        for j in range(i+1, n):
                            if augmented_matrix[j, i] != 0:
                                augmented_matrix[[i, j]] = augmented_matrix[[j, i]]
                                break
                        
                    # Eliminación hacia adelante
                    for j in range(i+1, n):
                        if augmented_matrix[j, i] != 0:
                            factor = augmented_matrix[j, i] / augmented_matrix[i, i]
                            augmented_matrix[j, i:] -= factor * augmented_matrix[i, i:]
                    
                    iterations.append(augmented_matrix.copy())
                
                # Sustitución regresiva
                x = np.zeros(n)
                for i in range(n-1, -1, -1):
                    x[i] = (augmented_matrix[i, -1] - np.dot(augmented_matrix[i, i+1:n], x[i+1:])) / augmented_matrix[i, i]
                
                # Preparar resultados para la plantilla
                tabla = []
                for idx, iteration in enumerate(iterations):
                    tabla.append({
                        'iteracion': idx + 1,
                        'matriz': iteration.tolist()
                    })
                
                return render(request, 'eliminacion_gaussiana.html', {
                    'form': form,
                    'tabla': tabla,
                    'solucion': x.tolist(),
                    'success': True
                })
            
            except ValueError as e:
                error_message = str(e)
                return render(request, 'eliminacion_gaussiana.html', {
                    'form': form,
                    'error': error_message
                })
    else:
        form = GaussEliminationForm()
    
    return render(request, 'eliminacion_gaussiana.html', {'form': form})


def eliminacion_gaussiana_pivoteo_parcial(request):
    if request.method == 'POST':
        form = GaussEliminationForm(request.POST)
        if form.is_valid():
            # Parsear la matriz A y el vector B
            try:
                matrix_a = [
                    [float(item) for item in row.split(',')] 
                    for row in form.cleaned_data['matrix_a'].split(';')
                ]
                vector_b = [
                    [float(item)] 
                    for item in form.cleaned_data['vector_b'].split(',')
                ]
                
                # Validaciones iniciales
                m = len(matrix_a)
                n = len(matrix_a[0]) if m > 0 else 0
                
                if m != n:
                    raise ValueError("La matriz no es cuadrada")
                if m != len(vector_b):
                    raise ValueError("El vector B tiene dimensión diferente")
                
                # Convertir a numpy array para facilitar operaciones
                M = np.array([row.copy() for row in matrix_a], dtype=float)
                B = np.array(vector_b, dtype=float)
                
                # Crear matriz aumentada
                augmented_matrix = np.hstack((M, B))
                
                # Proceso de eliminación gaussiana con pivoteo parcial
                iterations = []
                iterations.append({'matriz': augmented_matrix.copy(), 'pivot': None})
                
                n = len(augmented_matrix)
                
                for i in range(n-1):
                    # Pivoteo parcial - encontrar la fila con el máximo valor en la columna actual
                    max_row = i
                    max_val = abs(augmented_matrix[i, i])
                    
                    for j in range(i+1, n):
                        if abs(augmented_matrix[j, i]) > max_val:
                            max_val = abs(augmented_matrix[j, i])
                            max_row = j
                    
                    # Intercambiar filas si es necesario
                    if max_row != i:
                        augmented_matrix[[i, max_row]] = augmented_matrix[[max_row, i]]
                        pivot_info = f"Intercambio de filas: F{i+1} ↔ F{max_row+1}"
                    else:
                        pivot_info = f"No se requirió intercambio en iteración {i+1}"
                    
                    # Eliminación hacia adelante
                    for j in range(i+1, n):
                        if augmented_matrix[j, i] != 0:
                            factor = augmented_matrix[j, i] / augmented_matrix[i, i]
                            augmented_matrix[j, i:] -= factor * augmented_matrix[i, i:]
                    
                    iterations.append({'matriz': augmented_matrix.copy(), 'pivot': pivot_info})
                
                # Sustitución regresiva
                x = np.zeros(n)
                for i in range(n-1, -1, -1):
                    x[i] = (augmented_matrix[i, -1] - np.dot(augmented_matrix[i, i+1:n], x[i+1:])) / augmented_matrix[i, i]
                
                # Preparar resultados para la plantilla
                tabla = []
                for idx, iteration in enumerate(iterations):
                    tabla.append({
                        'iteracion': idx + 1,
                        'matriz': iteration['matriz'].tolist(),
                        'pivot_info': iteration['pivot']
                    })
                
                return render(request, 'eliminacion_gaussiana_pivoteo.html', {
                    'form': form,
                    'tabla': tabla,
                    'solucion': x.tolist(),
                    'success': True
                })
            
            except ValueError as e:
                error_message = str(e)
                return render(request, 'eliminacion_gaussiana_pivoteo.html', {
                    'form': form,
                    'error': error_message
                })
    else:
        form = GaussEliminationForm()
    
    return render(request, 'eliminacion_gaussiana_pivoteo.html', {'form': form})

def find_max_element(matrix, start_row, start_col):
    max_val = abs(matrix[start_row, start_col])
    max_row = start_row
    max_col = start_col
    n = matrix.shape[0]
    
    for i in range(start_row, n):
        for j in range(start_col, n):
            if abs(matrix[i, j]) > max_val:
                max_val = abs(matrix[i, j])
                max_row = i
                max_col = j
    return (max_row, max_col)


def eliminacion_gaussiana_pivoteo_total(request):
    if request.method == 'POST':
        form = GaussEliminationForm(request.POST)
        if form.is_valid():
            # Parsear la matriz A y el vector B
            try:
                matrix_a = [
                    [float(item) for item in row.split(',')] 
                    for row in form.cleaned_data['matrix_a'].split(';')
                ]
                vector_b = [
                    [float(item)] 
                    for item in form.cleaned_data['vector_b'].split(',')
                ]
                
                # Validaciones iniciales
                m = len(matrix_a)
                n = len(matrix_a[0]) if m > 0 else 0
                
                if m != n:
                    raise ValueError("La matriz no es cuadrada")
                if m != len(vector_b):
                    raise ValueError("El vector B tiene dimensión diferente")
                
                # Convertir a numpy array para facilitar operaciones
                M = np.array([row.copy() for row in matrix_a], dtype=float)
                B = np.array(vector_b, dtype=float)
                
                # Crear matriz aumentada
                augmented_matrix = np.hstack((M, B))
                
                # Proceso de eliminación gaussiana con pivoteo total
                iterations = []
                marca = np.arange(n)  # Para rastrear cambios de columnas
                iterations.append({
                    'matriz': augmented_matrix.copy(),
                    'pivot_info': "Matriz inicial",
                    'marca': marca.copy()
                })
                
                n = len(augmented_matrix)
                
                for i in range(n-1):
                    # Pivoteo total - encontrar el elemento máximo en la submatriz restante
                    max_row, max_col = find_max_element(augmented_matrix, i, i)
                    
                    pivot_info = []
                    
                    # Intercambio de columnas si es necesario
                    if i != max_col:
                        augmented_matrix[:, [i, max_col]] = augmented_matrix[:, [max_col, i]]
                        marca[i], marca[max_col] = marca[max_col], marca[i]
                        pivot_info.append(f"Intercambio de columnas: C{i+1} ↔ C{max_col+1}")
                    
                    # Intercambio de filas si es necesario
                    if i != max_row:
                        augmented_matrix[[i, max_row]] = augmented_matrix[[max_row, i]]
                        pivot_info.append(f"Intercambio de filas: F{i+1} ↔ F{max_row+1}")
                    
                    if not pivot_info:
                        pivot_info.append(f"No se requirió intercambio en iteración {i+1}")
                    
                    # Eliminación hacia adelante
                    for j in range(i+1, n):
                        if augmented_matrix[j, i] != 0:
                            factor = augmented_matrix[j, i] / augmented_matrix[i, i]
                            augmented_matrix[j, i:] -= factor * augmented_matrix[i, i:]
                    
                    iterations.append({
                        'matriz': augmented_matrix.copy(),
                        'pivot_info': " | ".join(pivot_info),
                        'marca': marca.copy()
                    })
                
                # Sustitución regresiva
                x = np.zeros(n)
                for i in range(n-1, -1, -1):
                    x[i] = (augmented_matrix[i, -1] - np.dot(augmented_matrix[i, i+1:n], x[i+1:])) / augmented_matrix[i, i]
                
                # Reordenar la solución según los intercambios de columnas
                temp_sol = x.copy()
                for i in range(n):
                    x[marca[i]] = temp_sol[i]
                
                # Preparar resultados para la plantilla
                tabla = []
                for idx, iteration in enumerate(iterations):
                    tabla.append({
                        'iteracion': idx + 1,
                        'matriz': iteration['matriz'].tolist(),
                        'pivot_info': iteration['pivot_info'],
                        'marca': iteration['marca'].tolist()
                    })
                
                return render(request, 'eliminacion_gaussiana_pivoteo_total.html', {
                    'form': form,
                    'tabla': tabla,
                    'solucion': x.tolist(),
                    'marca_final': marca.tolist(),
                    'success': True
                })
            
            except ValueError as e:
                error_message = str(e)
                return render(request, 'eliminacion_gaussiana_pivoteo_total.html', {
                    'form': form,
                    'error': error_message
                })
    else:
        form = GaussEliminationForm()
    
    return render(request, 'eliminacion_gaussiana_pivoteo_total.html', {'form': form})  


def progressive_substitution(L, B):
    n = len(L)
    z = np.zeros(n)
    
    for i in range(n):
        sum_val = 0
        for j in range(i):
            sum_val += L[i][j] * z[j]
        z[i] = (B[i] - sum_val) / L[i][i]
    return z

def factorizacion_LU_simple(request):
    if request.method == 'POST':
        form = LUDecompositionForm(request.POST)
        if form.is_valid():
            # Parsear la matriz A y el vector B
            try:
                matrix_a = [
                    [float(item) for item in row.split(',')] 
                    for row in form.cleaned_data['matrix_a'].split(';')
                ]
                vector_b = [
                    [float(item)] 
                    for item in form.cleaned_data['vector_b'].split(',')
                ]
                
                # Validaciones iniciales
                n = len(matrix_a)
                
                if n != len(matrix_a[0]):
                    raise ValueError("La matriz no es cuadrada")
                if n != len(vector_b):
                    raise ValueError("El vector B tiene dimensión diferente")
                
                # Convertir a numpy array para facilitar operaciones
                M = np.array([row.copy() for row in matrix_a], dtype=float)
                B = np.array(vector_b, dtype=float)
                
                # Inicializar matrices L y U
                L = np.eye(n)
                U = np.zeros((n, n))
                
                # Proceso de factorización LU
                iterations = []
                
                for i in range(n-1):
                    if M[i, i] == 0:
                        raise ValueError("Hay un 0 en la diagonal principal")
                    
                    # Multiplicadores y eliminación
                    for j in range(i+1, n):
                        if M[j, i] != 0:
                            L[j, i] = M[j, i] / M[i, i]
                            M[j, i:] -= (M[j, i] / M[i, i]) * M[i, i:]
                    
                    # Actualizar U
                    for j in range(i, n):
                        U[i, j] = M[i, j]
                    for j in range(i+1, n):
                        U[i+1, j] = M[i+1, j]
                    
                    iterations.append({
                        'M': M.copy(),
                        'L': L.copy(),
                        'U': U.copy(),
                        'iteracion': i+1
                    })
                
                # Completar la última entrada de U
                U[n-1, n-1] = M[n-1, n-1]
                
                # Resolver sistema
                z = progressive_substitution(L, B.flatten())
                x = np.linalg.solve(U, z)
                
                return render(request, 'factorizacion_lu_simple.html', {
                    'form': form,
                    'iterations': iterations,
                    'L_final': L.tolist(),
                    'U_final': U.tolist(),
                    'solucion': x.tolist(),
                    'success': True
                })
            
            except ValueError as e:
                error_message = str(e)
                return render(request, 'factorizacion_lu_simple.html', {
                    'form': form,
                    'error': error_message
                })
    else:
        form = LUDecompositionForm()
    
    return render(request, 'factorizacion_lu_simple.html', {'form': form})


def progressive_substitution(L, B):
    n = len(L)
    z = np.zeros(n)
    
    for i in range(n):
        sum_val = 0
        for j in range(i):
            sum_val += L[i][j] * z[j]
        z[i] = (B[i] - sum_val) / L[i][i]
    return z

def zero_in_diagonal(matrix):
    for i in range(len(matrix)):
        if matrix[i][i] == 0:
            return True
    return False

def factorizacion_LU_parcial(request):
    if request.method == 'POST':
        form = LUPivoteoParcialForm(request.POST)
        if form.is_valid():
            # Parsear la matriz A y el vector B
            try:
                matrix_a = [
                    [float(item) for item in row.split(',')] 
                    for row in form.cleaned_data['matrix_a'].split(';')
                ]
                vector_b = [
                    [float(item)] 
                    for item in form.cleaned_data['vector_b'].split(',')
                ]
                
                # Validaciones iniciales
                n = len(matrix_a)
                
                if n != len(matrix_a[0]):
                    raise ValueError("La matriz no es cuadrada")
                if n != len(vector_b):
                    raise ValueError("El vector B tiene dimensión diferente")
                if zero_in_diagonal(matrix_a):
                    raise ValueError("Hay ceros en la diagonal principal. El método no puede ejecutarse")
                
                # Convertir a numpy array para facilitar operaciones
                M = np.array([row.copy() for row in matrix_a], dtype=float)
                B = np.array(vector_b, dtype=float)
                
                # Inicializar matrices L, U y P
                L = np.eye(n)
                U = np.zeros((n, n))
                P = np.eye(n)
                
                # Proceso de factorización LU con pivoteo parcial
                iterations = []
                
                for i in range(n-1):
                    # Pivoteo parcial - encontrar el máximo en la columna actual
                    max_row = i
                    max_val = abs(M[i, i])
                    
                    for j in range(i+1, n):
                        if abs(M[j, i]) > max_val:
                            max_val = abs(M[j, i])
                            max_row = j
                    
                    # Intercambiar filas si es necesario
                    if max_row != i:
                        # Intercambiar filas en M
                        M[[i, max_row]] = M[[max_row, i]]
                        # Intercambiar filas en P
                        P[[i, max_row]] = P[[max_row, i]]
                        # Intercambiar elementos en L
                        if i > 0:
                            L[[i, max_row], :i] = L[[max_row, i], :i]
                    
                    # Multiplicadores y eliminación
                    for j in range(i+1, n):
                        if M[j, i] != 0:
                            L[j, i] = M[j, i] / M[i, i]
                            M[j, i:] -= L[j, i] * M[i, i:]
                    
                    # Actualizar U
                    for j in range(i, n):
                        U[i, j] = M[i, j]
                    for j in range(i+1, n):
                        U[i+1, j] = M[i+1, j]
                    
                    iterations.append({
                        'M': M.copy(),
                        'L': L.copy(),
                        'U': U.copy(),
                        'P': P.copy(),
                        'iteracion': i+1
                    })
                
                # Completar la última entrada de U
                U[n-1, n-1] = M[n-1, n-1]
                
                # Resolver sistema
                PB = np.dot(P, B)  # Aplicar permutación al vector B
                z = progressive_substitution(L, PB.flatten())
                x = np.linalg.solve(U, z)
                
                return render(request, 'factorizacion_lu_parcial.html', {
                    'form': form,
                    'iterations': iterations,
                    'L_final': L.tolist(),
                    'U_final': U.tolist(),
                    'P_final': P.tolist(),
                    'solucion': x.tolist(),
                    'success': True
                })
            
            except ValueError as e:
                error_message = str(e)
                return render(request, 'factorizacion_lu_parcial.html', {
                    'form': form,
                    'error': error_message
                })
    else:
        form = LUPivoteoParcialForm()
    
    return render(request, 'factorizacion_lu_parcial.html', {'form': form})


def progressive_substitution(L, B):
    n = len(L)
    z = np.zeros(n)
    for i in range(n):
        sum_val = 0
        for j in range(i):
            sum_val += L[i][j] * z[j]
        z[i] = (B[i] - sum_val) / L[i][i]
    return z

def croult(request):
    if request.method == 'POST':
        form = CroutForm(request.POST)
        if form.is_valid():
            try:
                # Parsear datos de entrada
                matrix_a = [
                    [float(item) for item in row.split(',')] 
                    for row in form.cleaned_data['matrix_a'].split(';')
                ]
                vector_b = [
                    [float(item)] 
                    for item in form.cleaned_data['vector_b'].split(',')
                ]
                
                n = len(matrix_a)
                
                # Validaciones
                if n != len(matrix_a[0]):
                    raise ValueError("La matriz no es cuadrada")
                if n != len(vector_b):
                    raise ValueError("El vector B tiene dimensión diferente")
                if np.linalg.det(matrix_a) == 0:
                    raise ValueError("El determinante de la matriz no puede ser cero")
                
                # Inicializar matrices L y U
                L = np.eye(n)
                U = np.eye(n)
                
                # Proceso de factorización de Crout
                iterations = []
                
                for i in range(n-1):
                    # Calcular columna i de L
                    for j in range(i, n):
                        product = sum(L[j][k] * U[k][i] for k in range(i))
                        L[j][i] = matrix_a[j][i] - product
                    
                    # Calcular fila i de U (excepto diagonal)
                    for j in range(i+1, n):
                        product = sum(L[i][k] * U[k][j] for k in range(i))
                        U[i][j] = (matrix_a[i][j] - product) / L[i][i]
                    
                    iterations.append({
                        'iteracion': i+1,
                        'L': [row.copy() for row in L],
                        'U': [row.copy() for row in U]
                    })
                
                # Calcular último elemento de L
                product = sum(L[n-1][k] * U[k][n-1] for k in range(n-1))
                L[n-1][n-1] = matrix_a[n-1][n-1] - product
                
                iterations.append({
                    'iteracion': n,
                    'L': [row.copy() for row in L],
                    'U': [row.copy() for row in U]
                })
                
                # Resolver sistema
                z = progressive_substitution(L, np.array(vector_b).flatten())
                x = np.linalg.solve(U, z)
                
                return render(request, 'crout.html', {
                    'form': form,
                    'iterations': iterations,
                    'L_final': L.tolist(),
                    'U_final': U.tolist(),
                    'solucion': x.tolist(),
                    'success': True
                })
            
            except ValueError as e:
                return render(request, 'crout.html', {
                    'form': form,
                    'error': str(e)
                })
    else:
        form = CroutForm()
    
    return render(request, 'crout.html', {'form': form})


def progressive_substitution(L, B):
    n = len(L)
    z = np.zeros(n)
    for i in range(n):
        sum_val = 0
        for j in range(i):
            sum_val += L[i][j] * z[j]
        z[i] = (B[i] - sum_val) / L[i][i]
    return z

def doolittle(request):
    if request.method == 'POST':
        form = DoolittleForm(request.POST)
        if form.is_valid():
            try:
                # Parsear datos de entrada
                matrix_a = [
                    [float(item) for item in row.split(',')] 
                    for row in form.cleaned_data['matrix_a'].split(';')
                ]
                vector_b = [
                    [float(item)] 
                    for item in form.cleaned_data['vector_b'].split(',')
                ]
                
                n = len(matrix_a)
                
                # Validaciones
                if n != len(matrix_a[0]):
                    raise ValueError("La matriz no es cuadrada")
                if n != len(vector_b):
                    raise ValueError("El vector B tiene dimensión diferente")
                if np.linalg.det(matrix_a) == 0:
                    raise ValueError("El determinante de la matriz no puede ser cero")
                
                # Inicializar matrices L y U
                L = np.eye(n)
                U = np.zeros((n, n))
                
                # Proceso de factorización de Doolittle
                iterations = []
                
                for i in range(n):
                    # Calcular fila i de U
                    for j in range(i, n):
                        sum_val = 0
                        for k in range(i):
                            sum_val += L[i][k] * U[k][j]
                        U[i][j] = matrix_a[i][j] - sum_val
                    
                    # Calcular columna i de L (excepto diagonal)
                    for j in range(i+1, n):
                        sum_val = 0
                        for k in range(i):
                            sum_val += L[j][k] * U[k][i]
                        L[j][i] = (matrix_a[j][i] - sum_val) / U[i][i]
                    
                    iterations.append({
                        'iteracion': i+1,
                        'L': [row.copy() for row in L],
                        'U': [row.copy() for row in U]
                    })
                
                # Resolver sistema
                z = progressive_substitution(L, np.array(vector_b).flatten())
                x = np.linalg.solve(U, z)
                
                return render(request, 'doolittle.html', {
                    'form': form,
                    'iterations': iterations,
                    'L_final': L.tolist(),
                    'U_final': U.tolist(),
                    'solucion': x.tolist(),
                    'success': True
                })
            
            except ValueError as e:
                return render(request, 'doolittle.html', {
                    'form': form,
                    'error': str(e)
                })
    else:
        form = DoolittleForm()
    
    return render(request, 'doolittle.html', {'form': form})

def progressive_substitution(L, B):
    n = len(L)
    z = np.zeros(n)
    for i in range(n):
        sum_val = 0
        for j in range(i):
            sum_val += L[i][j] * z[j]
        z[i] = (B[i] - sum_val) / L[i][i]
    return z

def cholesky(request):
    if request.method == 'POST':
        form = CholeskyForm(request.POST)
        if form.is_valid():
            try:
                # Parsear datos de entrada
                matrix_a = [
                    [float(item) for item in row.split(',')] 
                    for row in form.cleaned_data['matrix_a'].split(';')
                ]
                vector_b = [
                    [float(item)] 
                    for item in form.cleaned_data['vector_b'].split(',')
                ]
                
                n = len(matrix_a)
                
                # Validaciones
                if n != len(matrix_a[0]):
                    raise ValueError("La matriz no es cuadrada")
                if n != len(vector_b):
                    raise ValueError("El vector B tiene dimensión diferente")
                if not np.allclose(matrix_a, np.array(matrix_a).T):
                    raise ValueError("La matriz no es simétrica")
                if np.linalg.det(matrix_a) <= 0:
                    raise ValueError("La matriz no es definida positiva (determinante ≤ 0)")
                
                # Inicializar matriz L (triangular inferior)
                L = np.zeros((n, n))
                
                # Proceso de factorización de Cholesky
                iterations = []
                
                for i in range(n):
                    # Diagonal principal
                    sum_val = 0
                    for k in range(i):
                        sum_val += L[i][k] ** 2
                    L[i][i] = sqrt(matrix_a[i][i] - sum_val)
                    
                    # Elementos debajo de la diagonal
                    for j in range(i+1, n):
                        sum_val = 0
                        for k in range(i):
                            sum_val += L[j][k] * L[i][k]
                        L[j][i] = (matrix_a[j][i] - sum_val) / L[i][i]
                    
                    iterations.append({
                        'iteracion': i+1,
                        'L': [row.copy() for row in L],
                        'L_transpose': [row.copy() for row in L.T]  # U = L^T
                    })
                
                # Resolver sistema
                z = progressive_substitution(L, np.array(vector_b).flatten())
                x = np.linalg.solve(L.T, z)  # L^T x = z
                
                return render(request, 'cholesky.html', {
                    'form': form,
                    'iterations': iterations,
                    'L_final': L.tolist(),
                    'L_transpose_final': L.T.tolist(),
                    'solucion': x.tolist(),
                    'success': True
                })
            
            except ValueError as e:
                return render(request, 'cholesky.html', {
                    'form': form,
                    'error': str(e)
                })
    else:
        form = CholeskyForm()
    
    return render(request, 'cholesky.html', {'form': form})



def metodo_iterativo_Jacobi_Gauss_Seidel_SOR(request):
    if request.method == 'POST':
        form = IterativeMethodsForm(request.POST)
        if form.is_valid():
            try:
                # Parsear datos de entrada
                matrix_a = [
                    [float(item) for item in row.split(',')] 
                    for row in form.cleaned_data['matrix_a'].split(';')
                ]
                vector_b = [
                    float(item) 
                    for item in form.cleaned_data['vector_b'].split(',')
                ]
                x0 = [
                    float(item) 
                    for item in form.cleaned_data['x0'].split(',')
                ]
                tol = form.cleaned_data['tol']
                max_iter = form.cleaned_data['max_iter']
                norm_type = form.cleaned_data['norm_type']
                method = form.cleaned_data['method']
                w = form.cleaned_data['w'] if method == '3' else 1.0
                
                n = len(matrix_a)
                
                # Validaciones
                if n != len(matrix_a[0]):
                    raise ValueError("La matriz no es cuadrada")
                if n != len(vector_b):
                    raise ValueError("El vector B tiene dimensión diferente")
                if n != len(x0):
                    raise ValueError("El vector X0 tiene dimensión diferente")
                if tol < 0:
                    raise ValueError("La tolerancia debe ser positiva")
                if max_iter <= 0:
                    raise ValueError("El número máximo de iteraciones debe ser positivo")
                if method == '3' and (w <= 0 or w >= 2):
                    raise ValueError("El parámetro de relajación ω debe estar entre 0 y 2")
                
                # Convertir a numpy arrays
                A = np.array(matrix_a, dtype=float)
                B = np.array(vector_b, dtype=float)
                x = np.array(x0, dtype=float)
                
                # Verificar diagonal no nula
                if np.any(np.diag(A) == 0):
                    raise ValueError("Algunos elementos de la diagonal son cero")
                
                # Calcular matrices D, L, U
                D = np.diag(np.diag(A))
                L = -np.tril(A, -1)
                U = -np.triu(A, 1)
                
                # Preparar resultados
                results = {
                    'D': D.tolist(),
                    'L': L.tolist(),
                    'U': U.tolist(),
                    'iterations': [],
                    'conclusion': None,
                    'error': None,
                    'final_solution': None,
                    'spectral_radius': None
                }
                
                # Calcular matriz de iteración T y vector C según el método
                if method == '1':  # Jacobi
                    T = np.linalg.inv(D) @ (L + U)
                    C = np.linalg.inv(D) @ B
                elif method == '2':  # Gauss-Seidel
                    T = np.linalg.inv(D - L) @ U
                    C = np.linalg.inv(D - L) @ B
                else:  # SOR
                    T = np.linalg.inv(D - w * L) @ ((1 - w) * D + w * U)
                    C = w * np.linalg.inv(D - w * L) @ B
                
                results['T'] = T.tolist()
                results['C'] = C.tolist()
                
                # Calcular radio espectral
                eigenvalues = eig(T)[0]
                spectral_radius = max(abs(eigenvalues))
                results['spectral_radius'] = spectral_radius
                
                if spectral_radius >= 1:
                    raise ValueError(f"El radio espectral ({spectral_radius:.4f}) es ≥ 1. El método puede no converger")
                
                # Proceso iterativo
                iterations = []
                error = tol + 1
                count = 0
                
                iterations.append({
                    'count': count,
                    'x': x.copy().tolist(),
                    'error': None
                })
                
                while error > tol and count < max_iter:
                    x_new = T @ x + C
                    
                    # Calcular error según norma seleccionada
                    if norm_type == '1':
                        error = np.linalg.norm(x_new - x, 1)
                    elif norm_type == 'inf':
                        error = np.linalg.norm(x_new - x, np.inf)
                    else:  # Norma 2
                        error = np.linalg.norm(x_new - x)
                    
                    x = x_new
                    count += 1
                    
                    iterations.append({
                        'count': count,
                        'x': x.copy().tolist(),
                        'error': error
                    })
                
                results['iterations'] = iterations
                results['final_solution'] = x.tolist()
                
                if count >= max_iter:
                    results['conclusion'] = f"El método no convergió en {max_iter} iteraciones (Error: {error:.2e})"
                else:
                    results['conclusion'] = f"Convergencia alcanzada en {count} iteraciones (Error: {error:.2e})"
                
                return render(request, 'iterative_methods.html', {
                    'form': form,
                    'results': results,
                    'success': True,
                    'method_name': dict(form.fields['method'].choices)[method]
                })
            
            except ValueError as e:
                return render(request, 'iterative_methods.html', {
                    'form': form,
                    'error': str(e)
                })
    else:
        form = IterativeMethodsForm()
    
    return render(request, 'iterative_methods.html', {'form': form})