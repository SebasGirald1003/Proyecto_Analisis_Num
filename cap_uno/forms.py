from django import forms

class BiseccionForm(forms.Form):
    funcion      = forms.CharField(label='f(x)',   initial='x**3 - x - 2')
    a            = forms.FloatField(label='a')
    b            = forms.FloatField(label='b')
    tolerancia   = forms.FloatField(label='Tolerancia', initial=1e-6)
    max_iter     = forms.IntegerField(label='Máx iteraciones', initial=50, min_value=1)
