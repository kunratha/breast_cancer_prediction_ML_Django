from django.shortcuts import render
from django.http import HttpResponse
from django.template import loader
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import io
import base64
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.forms import UserCreationForm

from .apps import cleaned_df  # Import the dataframe from the app config

# Load and preprocess the data
# data = "C:/wildcode_school_courses/projet3_app/cleaned_df.csv"
# cleaned_df = pd.read_csv(data)


def home(request):
    template = loader.get_template("home.html")
    return HttpResponse(template.render())


def about(request):
    return render(request, "about.html")


def services(request):
    return render(request, "services.html")


def contact(request):
    return render(request, "contact.html")


def login(request):
    return render(request, "login.html")


def register(request):
    if request.method == "POST":
        form = UserCreationForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect("login")
    else:
        form = UserCreationForm()
    return render(request, "register.html", {"form": form})


url = "https://raw.githubusercontent.com/MaskiVal/DataSets/main/cancer_breast.csv"
breast_cancer_df = pd.read_csv(url)


def dataanalysis(request):
    # Section 1: Plotly Bar Chart for Diagnosis Distribution
    # Calculate the percentages
    value_counts = breast_cancer_df["diagnosis"].value_counts(normalize=True) * 100

    # Map the values "Benign" for "B" and "Malignant" for "M"
    value_counts.index = value_counts.index.map({"B": "Benign", "M": "Malignant"})

    # Create the figure
    fig = go.Figure()

    # Add bars
    fig.add_trace(
        go.Bar(x=value_counts.index, y=value_counts, marker_color="darkviolet")
    )

    # Add labels and title with custom colors
    fig.update_layout(
        title={
            "text": "Percentage Distribution of Diagnoses",
            "font": {"color": "darkblue"},  # Title color
            "x": 0.5,  # Center the title
            "xanchor": "center",  # Center the title
        },
        xaxis={
            "title": {
                "text": "Diagnosis",
                "font": {"color": "darkgreen"},  # X-axis label color
            },
            "tickfont": {"color": "darkgreen"},  # X-axis tick labels color
        },
        yaxis={
            "title": {
                "text": "Percentage (%)",
                "font": {"color": "darkred"},  # Y-axis label color
            },
            "tickfont": {"color": "darkred"},  # Y-axis tick labels color
        },
        paper_bgcolor="white",  # Set the background color of the paper
        plot_bgcolor="white",  # Set the background color of the plot
        width=1000,  # Set the width of the chart
        height=600,  # Set the height of the chart
    )

    # Convert the plotly figure to HTML
    plot_div = fig.to_html(full_html=False)

    # .................................................................................
    # Section 2: Histograms of Numeric Features by Diagnosis
    # Select only the numeric columns for plotting
    numeric_cols = cleaned_df.select_dtypes(include=[np.number])

    # Plotting histograms for each numeric feature by diagnosis
    fig, axes = plt.subplots(
        nrows=11, ncols=3, figsize=(15, 40)
    )  # Adjusted the number of rows to accommodate all subplots
    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    axes = axes.ravel()

    for idx, col in enumerate(numeric_cols.columns):
        if cleaned_df[col].nunique() > 1:  # Ensure there are multiple elements
            sns.histplot(
                data=cleaned_df, x=col, hue="diagnosis", kde=True, ax=axes[idx]
            )
            axes[idx].set_title(f"Distribution of {col} by Diagnosis")
            axes[idx].set_xlabel("")
            axes[idx].set_ylabel("")
        else:
            axes[idx].text(0.5, 0.5, f"Not enough data\nin {col}", ha="center")
            axes[idx].set_title(f"Distribution of {col} by Diagnosis")
            axes[idx].set_xlabel("")
            axes[idx].set_ylabel("")

    # Ensure all unused subplots are not visible
    for ax in axes[len(numeric_cols.columns) :]:
        ax.set_visible(False)

    # Save the histogram plot to a PNG image in memory
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png")
    plt.close()
    buffer.seek(0)
    hist_image_png = buffer.getvalue()
    buffer.close()

    # Encode the PNG image to base64 string
    hist_image_base64 = base64.b64encode(hist_image_png).decode("utf-8")
    hist_image_uri = f"data:image/png;base64,{hist_image_base64}"

    # Section 3: Correlation Heatmap of Features
    # Compute the correlation matrix
    corr_matrix = numeric_cols.corr()

    # Set up the matplotlib figure
    plt.figure(figsize=(20, 12))

    # Draw the heatmap with the correct aspect ratio
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        linewidths=1,
        cmap="coolwarm",
        cbar_kws={"shrink": 0.7},
    )
    plt.title("Correlation Heatmap of Features")

    # Save the heatmap plot to a PNG image in memory
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png")
    plt.close()
    buffer.seek(0)
    heatmap_image_png = buffer.getvalue()
    buffer.close()

    # Encode the PNG image to base64 string
    heatmap_image_base64 = base64.b64encode(heatmap_image_png).decode("utf-8")
    heatmap_image_uri = f"data:image/png;base64,{heatmap_image_base64}"

    # Render the template with the plots
    return render(
        request,
        "dataanalysis/dataanalysis.html",
        context={
            "plot_div": plot_div,
            "hist_image_uri": hist_image_uri,
            "heatmap_image_uri": heatmap_image_uri,
        },
    )


# import pandas as pd
# from sklearn.svm import SVC
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from django.http import JsonResponse
# from django.shortcuts import render
# from django.views.decorators.csrf import csrf_exempt

# # Load and preprocess the data
# data = "C:/wildcode_school_courses/wild-project3/scaled_cleaned_df.csv"
# scaled_cleaned_df = pd.read_csv(data)

# # Separate the features and the target variable
# features = scaled_cleaned_df.drop(columns=["diagnosis", "id"])
# target = scaled_cleaned_df["diagnosis"]

# # Scale the features
# scaler = StandardScaler()
# scaled_features = scaler.fit_transform(features)

# # Split the data into training and testing sets
# x_train, x_test, y_train, y_test = train_test_split(
#     scaled_features, target, test_size=0.3, random_state=42
# )

# # Train the SVM model
# svm_model = SVC(probability=True, class_weight="balanced")
# svm_model.fit(x_train, y_train)

# # Default values if the form is empty
# default_values = [
#     14.0,
#     20.5,
#     91.0,
#     500.0,
#     0.09,
#     0.1,
#     0.07,
#     0.045,
#     0.2,
#     0.06,
#     0.4,
#     1.0,
#     2.5,
#     25.0,
#     0.005,
#     0.02,
#     0.025,
#     0.015,
#     0.025,
#     0.001,
#     16.0,
#     25.0,
#     100.0,
#     700.0,
#     0.12,
#     0.2,
#     0.15,
#     0.07,
#     0.3,
#     0.08,
# ]


# def predict_diagnosis(input_data):
#     input_data = scaler.transform([input_data])
#     prediction = svm_model.predict(input_data)
#     print(
#         f"Prediction result for input {input_data}: {prediction[0]}"
#     )  # Log result to the terminal
#     return prediction[0]


# @csrf_exempt
# def predict(request):
#     if request.method == "POST":
#         try:
#             user_input = [
#                 float(request.POST.get("radius_mean") or default_values[0]),
#                 float(request.POST.get("texture_mean") or default_values[1]),
#                 float(request.POST.get("perimeter_mean") or default_values[2]),
#                 float(request.POST.get("area_mean") or default_values[3]),
#                 float(request.POST.get("smoothness_mean") or default_values[4]),
#                 float(request.POST.get("compactness_mean") or default_values[5]),
#                 float(request.POST.get("concavity_mean") or default_values[6]),
#                 float(request.POST.get("concave_points_mean") or default_values[7]),
#                 float(request.POST.get("symmetry_mean") or default_values[8]),
#                 float(request.POST.get("fractal_dimension_mean") or default_values[9]),
#                 float(request.POST.get("radius_se") or default_values[10]),
#                 float(request.POST.get("texture_se") or default_values[11]),
#                 float(request.POST.get("perimeter_se") or default_values[12]),
#                 float(request.POST.get("area_se") or default_values[13]),
#                 float(request.POST.get("smoothness_se") or default_values[14]),
#                 float(request.POST.get("compactness_se") or default_values[15]),
#                 float(request.POST.get("concavity_se") or default_values[16]),
#                 float(request.POST.get("concave_points_se") or default_values[17]),
#                 float(request.POST.get("symmetry_se") or default_values[18]),
#                 float(request.POST.get("fractal_dimension_se") or default_values[19]),
#                 float(request.POST.get("radius_worst") or default_values[20]),
#                 float(request.POST.get("texture_worst") or default_values[21]),
#                 float(request.POST.get("perimeter_worst") or default_values[22]),
#                 float(request.POST.get("area_worst") or default_values[23]),
#                 float(request.POST.get("smoothness_worst") or default_values[24]),
#                 float(request.POST.get("compactness_worst") or default_values[25]),
#                 float(request.POST.get("concavity_worst") or default_values[26]),
#                 float(request.POST.get("concave_points_worst") or default_values[27]),
#                 float(request.POST.get("symmetry_worst") or default_values[28]),
#                 float(
#                     request.POST.get("fractal_dimension_worst") or default_values[29]
#                 ),
#             ]
#             result = predict_diagnosis(user_input)
#             return JsonResponse({"result": result})
#         except ValueError as e:
#             return JsonResponse({"error": f"ValueError: {e}"}, status=400)
#         except Exception as e:
#             return JsonResponse({"error": f"Exception: {e}"}, status=400)

#     return render(request, "predict.html")

# import numpy as np
# import pandas as pd
# from sklearn.svm import SVC
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from django.http import JsonResponse
# from django.shortcuts import render
# from django.views.decorators.csrf import csrf_exempt

# # Load and preprocess the data
# data = "C:/wildcode_school_courses/wild-project3/scaled_cleaned_df.csv"
# scaled_cleaned_df = pd.read_csv(data)

# # Separate the features and the target variable
# features = scaled_cleaned_df.drop(columns=["diagnosis", "id"])
# target = scaled_cleaned_df["diagnosis"]

# # Scale the features
# scaler = StandardScaler()
# scaled_features = scaler.fit_transform(features)

# # Split the data into training and testing sets
# x_train, x_test, y_train, y_test = train_test_split(
#     scaled_features, target, test_size=0.3, random_state=42
# )

# # Train the SVM model
# svm_model = SVC(probability=True, class_weight="balanced")
# svm_model.fit(x_train, y_train)

# # Default values if the form is empty
# default_values = [
#     14.0,
#     20.5,
#     91.0,
#     500.0,
#     0.09,
#     0.1,
#     0.07,
#     0.045,
#     0.2,
#     0.06,
#     0.4,
#     1.0,
#     2.5,
#     25.0,
#     0.005,
#     0.02,
#     0.025,
#     0.015,
#     0.025,
#     0.001,
#     16.0,
#     25.0,
#     100.0,
#     700.0,
#     0.12,
#     0.2,
#     0.15,
#     0.07,
#     0.3,
#     0.08,
# ]


# def predict_diagnosis(input_data):
#     input_data = scaler.transform([input_data])
#     prediction = svm_model.predict(input_data)
#     print(
#         f"Prediction result for input {input_data}: {prediction[0]}"
#     )  # Print the result on the terminal
#     return str(prediction[0])  # Convert to string to ensure JSON serialization


# @csrf_exempt
# def predict(request):
#     if request.method == "POST":
#         try:
#             user_input = [
#                 float(request.POST.get("radius_mean") or default_values[0]),
#                 float(request.POST.get("texture_mean") or default_values[1]),
#                 float(request.POST.get("perimeter_mean") or default_values[2]),
#                 float(request.POST.get("area_mean") or default_values[3]),
#                 float(request.POST.get("smoothness_mean") or default_values[4]),
#                 float(request.POST.get("compactness_mean") or default_values[5]),
#                 float(request.POST.get("concavity_mean") or default_values[6]),
#                 float(request.POST.get("concave_points_mean") or default_values[7]),
#                 float(request.POST.get("symmetry_mean") or default_values[8]),
#                 float(request.POST.get("fractal_dimension_mean") or default_values[9]),
#                 float(request.POST.get("radius_se") or default_values[10]),
#                 float(request.POST.get("texture_se") or default_values[11]),
#                 float(request.POST.get("perimeter_se") or default_values[12]),
#                 float(request.POST.get("area_se") or default_values[13]),
#                 float(request.POST.get("smoothness_se") or default_values[14]),
#                 float(request.POST.get("compactness_se") or default_values[15]),
#                 float(request.POST.get("concavity_se") or default_values[16]),
#                 float(request.POST.get("concave_points_se") or default_values[17]),
#                 float(request.POST.get("symmetry_se") or default_values[18]),
#                 float(request.POST.get("fractal_dimension_se") or default_values[19]),
#                 float(request.POST.get("radius_worst") or default_values[20]),
#                 float(request.POST.get("texture_worst") or default_values[21]),
#                 float(request.POST.get("perimeter_worst") or default_values[22]),
#                 float(request.POST.get("area_worst") or default_values[23]),
#                 float(request.POST.get("smoothness_worst") or default_values[24]),
#                 float(request.POST.get("compactness_worst") or default_values[25]),
#                 float(request.POST.get("concavity_worst") or default_values[26]),
#                 float(request.POST.get("concave_points_worst") or default_values[27]),
#                 float(request.POST.get("symmetry_worst") or default_values[28]),
#                 float(
#                     request.POST.get("fractal_dimension_worst") or default_values[29]
#                 ),
#             ]
#             result = predict_diagnosis(user_input)
#             return JsonResponse({"result": result})
#         except ValueError as e:
#             return JsonResponse({"error": f"ValueError: {e}"}, status=400)
#         except Exception as e:
#             return JsonResponse({"error": f"Exception: {e}"}, status=400)

#     return render(request, "predict.html")


# Separate the features and the target variable
X = cleaned_df.drop(columns=["diagnosis", "id"])
y = cleaned_df["diagnosis"]

# Scale the features
scaler = StandardScaler()
scaled_X = scaler.fit_transform(X)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.3, random_state=42
)

# Train the SVM model
svm_model = SVC(probability=True, class_weight="balanced")
svm_model.fit(x_train, y_train)

# Default values if the form is empty
default_values = [
    14.0,
    20.5,
    91.0,
    500.0,
    0.09,
    0.1,
    0.07,
    0.045,
    0.2,
    0.06,
    0.4,
    1.0,
    2.5,
    25.0,
    0.005,
    0.02,
    0.025,
    0.015,
    0.025,
    0.001,
    16.0,
    25.0,
    100.0,
    700.0,
    0.12,
    0.2,
    0.15,
    0.07,
    0.3,
    0.08,
]


def predict_diagnosis(input_data):
    input_data = scaler.transform([input_data])
    prediction = svm_model.predict(input_data)
    print(f"Prediction result for input {input_data}: {prediction[0]}")

    # Convert numerical result to "M" or "B"
    diagnosis = "M" if prediction[0] == 1 else "B"
    return diagnosis


@csrf_exempt
def predict(request):
    if request.method == "POST":
        try:
            user_input = [
                float(request.POST.get("radius_mean", default_values[0])),
                float(request.POST.get("texture_mean", default_values[1])),
                float(request.POST.get("perimeter_mean", default_values[2])),
                float(request.POST.get("area_mean", default_values[3])),
                float(request.POST.get("smoothness_mean", default_values[4])),
                float(request.POST.get("compactness_mean", default_values[5])),
                float(request.POST.get("concavity_mean", default_values[6])),
                float(request.POST.get("concave_points_mean", default_values[7])),
                float(request.POST.get("symmetry_mean", default_values[8])),
                float(request.POST.get("fractal_dimension_mean", default_values[9])),
                float(request.POST.get("radius_se", default_values[10])),
                float(request.POST.get("texture_se", default_values[11])),
                float(request.POST.get("perimeter_se", default_values[12])),
                float(request.POST.get("area_se", default_values[13])),
                float(request.POST.get("smoothness_se", default_values[14])),
                float(request.POST.get("compactness_se", default_values[15])),
                float(request.POST.get("concavity_se", default_values[16])),
                float(request.POST.get("concave_points_se", default_values[17])),
                float(request.POST.get("symmetry_se", default_values[18])),
                float(request.POST.get("fractal_dimension_se", default_values[19])),
                float(request.POST.get("radius_worst", default_values[20])),
                float(request.POST.get("texture_worst", default_values[21])),
                float(request.POST.get("perimeter_worst", default_values[22])),
                float(request.POST.get("area_worst", default_values[23])),
                float(request.POST.get("smoothness_worst", default_values[24])),
                float(request.POST.get("compactness_worst", default_values[25])),
                float(request.POST.get("concavity_worst", default_values[26])),
                float(request.POST.get("concave_points_worst", default_values[27])),
                float(request.POST.get("symmetry_worst", default_values[28])),
                float(request.POST.get("fractal_dimension_worst", default_values[29])),
            ]
            result = predict_diagnosis(user_input)
            return JsonResponse({"result": result})
        except ValueError as e:
            return JsonResponse({"error": f"ValueError: {e}"}, status=400)
        except Exception as e:
            return JsonResponse({"error": f"Exception: {e}"}, status=400)

    return render(request, "predict.html")
