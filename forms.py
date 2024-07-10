from django import forms
from .models import CusUser

class LoginForm(forms.Form):
    username = forms.CharField(max_length=100, widget=forms.TextInput(attrs={'class': 'form-control text-dark bg-transparent border-0'}))
    password = forms.CharField(widget=forms.PasswordInput(attrs={'class': 'form-control text-dark bg-transparent border-0'}))

    def clean(self):
        cleaned_data = super().clean()
        username = cleaned_data.get('username')
        password = cleaned_data.get('password')

        if username and password:
            try:
                user = CusUser.objects.get(username=username)
                if user.password != password:  # Compare the plain text password
                    raise forms.ValidationError("Invalid username or password.")
            except CusUser.DoesNotExist:
                raise forms.ValidationError("Invalid username or password.")

        return cleaned_data
