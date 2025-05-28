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
    derivada = forms.CharField(label='f\'(x)', required=False)
    x0 = forms.FloatField(label='x0')
    tolerancia = forms.FloatField(label='Tolerancia')
    max_iter     = forms.IntegerField(label='Máx iteraciones', initial=50, min_value=1)

class SecanteForm(forms.Form):
    funcion = forms.CharField(label='f(x)', max_length=100)
    x0 = forms.FloatField(label='x0')
    x1 = forms.FloatField(label='x1')
    tolerancia = forms.FloatField(label='Tolerancia')
    max_iter     = forms.IntegerField(label='Máx iteraciones', initial=50, min_value=1)

class RaicesMultiplesForm(forms.Form):
    funcion = forms.CharField(label='f(x)', max_length=100)
    derivada_1 = forms.CharField(label="f'(x)", max_length=100, required=False)
    derivada_2 = forms.CharField(label="f''(x)", max_length=100, required=False)
    x0 = forms.FloatField(label='x₀')
    tolerancia = forms.FloatField(label='Tolerancia')
    max_iter = forms.IntegerField(label='Máximo de Iteraciones')
