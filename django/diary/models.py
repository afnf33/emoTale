from django.db import models

# Create your models here.

class User(models.Model):
    id = models.IntegerField()
    name = models.CharField(max_length = 5)
    count = models.IntegerField(default = 0)
    email = models.EmailField(primary_key = True)

class Result(models.Model):
    id = models.IntegerField(primary_key = True)
    happiness = models.FloatField()
    sadness = models.FloatField()
    anger = models.FloatField()
    fear = models.FloatField()
    email = models.ForeignKey(User, on_delete=models.PROTECT)
    
class Content(models.Model):
    id = models.IntegerField(primary_key = True)
    email = models.ForeignKey(User, on_delete=models.PROTECT)
    text = models.TextField(max_length=1000)
