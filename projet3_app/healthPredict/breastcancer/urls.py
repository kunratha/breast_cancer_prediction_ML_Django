from django.urls import path
from . import views

urlpatterns = [
    path("", views.home, name="home"),
    path("dataanalysis/", views.dataanalysis, name="dataanalysis"),
    path("predict/", views.predict, name="predict"),
    path("about/", views.about, name="about"),
    path("contact/", views.contact, name="contact"),
    path("login/", views.login, name="login"),
    path("register/", views.register, name="register"),
    path("services/", views.services, name="services"),
]
