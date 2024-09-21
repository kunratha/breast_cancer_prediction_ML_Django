from django.db import models


class Register(models.Model):
    username = models.CharField(max_length=255)
    password1 = models.CharField(max_length=255)
    password2 = models.IntegerField(null=True)
    joined_date = models.DateField(null=True)

    def __str__(self):
        return f"{self.username} {self.password1} {self.password2}"
