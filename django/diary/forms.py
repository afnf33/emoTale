from django.forms import ModelForm
from diary.models import Content, User, Result

class ContentForm(ModelForm):
    class Meta:
        model = Content
        fields = ['text']

class UserForm(ModelForm):
    class Meta:
        model = User
        fields = ['id', 'name', 'count', 'email']
        
class ResultForm(ModelForm):
    class Meta:
        model = Result
        fields = ['id', 'happiness', 'sadness', 'anger', 'fear', 'email']
