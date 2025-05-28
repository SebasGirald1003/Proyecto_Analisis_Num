from django import forms

class GaussEliminationForm(forms.Form):
    matrix_a = forms.CharField(
        label='Matriz A (ingrese filas separadas por ";", elementos por ",")',
        widget=forms.Textarea(attrs={'rows': 4, 'cols': 40}),
        help_text='Ejemplo: 2,1,-1;0,1,1;1,0,1'
    )
    vector_b = forms.CharField(
        label='Vector B (ingrese elementos separados por ",")',
        widget=forms.Textarea(attrs={'rows': 2, 'cols': 40}),
        help_text='Ejemplo: 1,2,3'
    )


class LUDecompositionForm(forms.Form):
    matrix_a = forms.CharField(
        label='Matriz A (ingrese filas separadas por ";", elementos por ",")',
        widget=forms.Textarea(attrs={'rows': 4, 'cols': 40}),
        help_text='Ejemplo: 2,1,-1;0,1,1;1,0,1'
    )
    vector_b = forms.CharField(
        label='Vector B (ingrese elementos separados por ",")',
        widget=forms.Textarea(attrs={'rows': 2, 'cols': 40}),
        help_text='Ejemplo: 1,2,3'
    )

class LUPivoteoParcialForm(forms.Form):
    matrix_a = forms.CharField(
        label='Matriz A (ingrese filas separadas por ";", elementos por ",")',
        widget=forms.Textarea(attrs={'rows': 4, 'cols': 40}),
        help_text='Ejemplo: 2,1,-1;0,1,1;1,0,1'
    )
    vector_b = forms.CharField(
        label='Vector B (ingrese elementos separados por ",")',
        widget=forms.Textarea(attrs={'rows': 2, 'cols': 40}),
        help_text='Ejemplo: 1,2,3'
    )

class CroutForm(forms.Form):
    matrix_a = forms.CharField(
        label='Matriz A (ingrese filas separadas por ";", elementos por ",")',
        widget=forms.Textarea(attrs={'rows': 4, 'cols': 40}),
        help_text='Ejemplo: 4,3,2;6,3,4;4,9,2'
    )
    vector_b = forms.CharField(
        label='Vector B (ingrese elementos separados por ",")',
        widget=forms.Textarea(attrs={'rows': 2, 'cols': 40}),
        help_text='Ejemplo: 1,2,3'
    )


class DoolittleForm(forms.Form):
    matrix_a = forms.CharField(
        label='Matriz A (ingrese filas separadas por ";", elementos por ",")',
        widget=forms.Textarea(attrs={'rows': 4, 'cols': 40}),
        help_text='Ejemplo: 4,3,2;6,3,4;4,9,2'
    )
    vector_b = forms.CharField(
        label='Vector B (ingrese elementos separados por ",")',
        widget=forms.Textarea(attrs={'rows': 2, 'cols': 40}),
        help_text='Ejemplo: 1,2,3'
    )

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

class IterativeMethodsForm(forms.Form):
    matrix_a = forms.CharField(
        label='Matriz A (filas separadas por ";", elementos por ",")',
        widget=forms.Textarea(attrs={'rows': 4, 'cols': 40}),
        help_text='Ejemplo: 4,1,1;1,2,3;1,3,6'
    )
    vector_b = forms.CharField(
        label='Vector B (elementos separados por ",")',
        widget=forms.Textarea(attrs={'rows': 2, 'cols': 40}),
        help_text='Ejemplo: 1,2,3'
    )
    x0 = forms.CharField(
        label='Vector inicial X0 (elementos separados por ",")',
        widget=forms.Textarea(attrs={'rows': 1, 'cols': 40}),
        help_text='Ejemplo: 0,0,0'
    )
    tol = forms.FloatField(
        label='Tolerancia',
        initial=1e-6,
        min_value=0
    )
    max_iter = forms.IntegerField(
        label='Máximo de iteraciones',
        initial=100,
        min_value=1
    )
    norm_type = forms.ChoiceField(
        label='Tipo de norma',
        choices=[
            ('1', 'Norma 1'),
            ('2', 'Norma 2'),
            ('inf', 'Norma infinito')
        ],
        initial='inf'
    )
    method = forms.ChoiceField(
        label='Método',
        choices=[
            ('1', 'Jacobi'),
            ('2', 'Gauss-Seidel'),
            ('3', 'SOR')
        ],
        initial='1'
    )
    w = forms.FloatField(
        label='Parámetro de relajación (ω) - Solo para SOR',
        initial=1.0,
        required=False,
        min_value=0,
        max_value=2
    )