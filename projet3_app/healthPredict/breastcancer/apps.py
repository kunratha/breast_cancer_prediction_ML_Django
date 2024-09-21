from django.apps import AppConfig
import pandas as pd


class BreastcancerConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "breastcancer"

    def ready(self):
        # Load the dataset when the app is ready
        data_path = "C:/wildcode_school_courses/projet3_app/cleaned_df.csv"
        global cleaned_df
        cleaned_df = pd.read_csv(data_path)
