from django.contrib import admin
from .models import Register

# Register your models here.


class RegisterAdmin(admin.ModelAdmin):
    list_display = (
        "username",
        "password1",
        "password2",
        "joined_date",
    )


admin.site.register(Register, RegisterAdmin)
