# myapp/models.py
from django.db import models

class PressedButton(models.Model):
    press_date = models.DateTimeField(auto_now_add=True)
    result_number = models.IntegerField()
