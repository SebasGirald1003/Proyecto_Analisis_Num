from django import forms

class BiseccionForm(forms.Form):
    funcion      = forms.CharField(label='f(x)',   initial='x**3 - x - 2')
    a            = forms.FloatField(label='a')
    b            = forms.FloatField(label='b')
    tolerancia   = forms.FloatField(label='Tolerancia', initial=1e-6)
    max_iter     = forms.IntegerField(label='Máx iteraciones', initial=50, min_value=1)

class PuntoFijoForm(forms.Form):
    funcion_f = forms.CharField(label='f(x)', max_length=100)
    funcion_g = forms.CharField(label='g(x)', max_length=100)
    x0 = forms.FloatField(label='x0')
    tolerancia   = forms.FloatField(label='Tolerancia', initial=1e-6)
    max_iter     = forms.IntegerField(label='Máx iteraciones', initial=50, min_value=1)

class NewtonForm(forms.Form):
    funcion = forms.CharField(label='f(x)', max_length=100)
    derivada = forms.CharField(label="f'(x)", max_length=100)
    x0 = forms.FloatField(label='x0')
    tolerancia = forms.FloatField(label='Tolerancia')
    max_iter     = forms.IntegerField(label='Máx iteraciones', initial=50, min_value=1)

class SecanteForm(forms.Form):
    funcion = forms.CharField(label='f(x)', max_length=100)
    x0 = forms.FloatField(label='x0')
    x1 = forms.FloatField(label='x1')
    tolerancia = forms.FloatField(label='Tolerancia')
    max_iter     = forms.IntegerField(label='Máx iteraciones', initial=50, min_value=1)

class CholeskyForm(forms.Form):
    matrix_a = forms.CharField(
        label='Matriz A simétrica definida positiva (filas separadas por ";", elementos por ",")',
        widget=forms.Textarea(attrs={'rows': 4, 'cols': 40}),
        help_text='Ejemplo: 4,1,1;1,2,3;1,3,6'
    )
    vector_b = forms.CharField(
        label='Vector B (ingrese elementos separados por ",")',
        widget=forms.Textarea(attrs={'rows': 2, 'cols': 40}),
        help_text='Ejemplo: 1,2,3'
    )