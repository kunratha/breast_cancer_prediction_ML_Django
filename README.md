# **Breast Cancer Prediction using Machine Learning and Django**

This project is a **breast cancer prediction system** built using **Machine Learning** and **Django**. The system leverages a trained machine learning model to predict whether a tumor is **malignant** or **benign** based on medical features such as *radius*, *texture*, and *smoothness* of the tumor. It provides an intuitive web interface where users can input features and receive a prediction.

## **Features**

- **Django Web Framework**: Built with Django to provide a clean, user-friendly interface.
- **Machine Learning**: Uses a trained machine learning model to predict breast cancer (**SVM**).
- **User Input Form**: Allows users to input medical features manually.
- **Prediction Result**: Displays whether the tumor is predicted to be malignant or benign.
- **Bootstrap Integrated**: Front-end styled using Bootstrap for responsive design.

## **Project Structure**
breast-cancer_prediction_ML_Django/
├── healthPredict
  ├── breastcancer/
  │   ├── migrations/
  │   ├── static/
  │   │   ├── css/              # CSS files
  │   │   ├── js/               # JavaScript files
  │   │   └── images/           # Images
  │   ├── templates/
  │   │   └── breastcancer/     # HTML templates
  │   ├── models.py             # Machine Learning model integration
  │   ├── views.py              # Django views for form submission and results
  │   ├── urls.py               # URL routing for the app
  │   └── ...
├── manage.py                 # Django project management file
├── db.sqlite3                # SQLite database (optional)
└── README.md                 # Project documentation (this file)


## **Installation and Setup**

### **Prerequisites**

- **Python 3.8+**
- **pip** (Python package installer)
- **Django 4.x**
- **Git** (optional, for version control)

### 1. **Clone the Repository**

First, clone the project repository from GitHub:

```bash
git clone https://github.com/kunratha/breast-cancer_prediction_ML_Django.git
cd breast-cancer_prediction_ML_Django 
```

### 2. **Create and Activate Virtual Environment (optional)**

It's recommended to use a virtual environment to manage your dependencies. You can create and activate one as follows:
python -m venv env
source env/bin/activate  # For Windows: env\Scripts\activate

### 3. **Install Dependencies**

Install the required Python libraries by running:
pip install -r requirements.txt

Make sure your requirements.txt file includes all necessary dependencies, such as:
Django>=4.0
scikit-learn
pandas
numpy

### 4. **Run Migrations**
Apply the Django migrations to set up the database:
python manage.py migrate

### 5. **Run the Development Server**
To start the Django development server, run:

**python manage.py runserver**

**Visit http://127.0.0.1:8000/ in your browser to view the application.**

**How It Works**
**Input Features:** The user enters tumor-related medical features like radius, texture, and perimeter.
**Prediction:** The system feeds the input into a pre-trained machine learning model, which returns a prediction of either malignant or benign.
**Result Display:** The result is shown to the user in a modal, providing an easy-to-understand interpretation of the prediction.

**Machine Learning Model**
The prediction model was trained using the Wisconsin Breast Cancer Dataset, and integrated into the Django app for predictions. The model might be trained using algorithms such as:

**Support Vector Machine (SVM)**

The model takes several input features, including:
Radius Mean
Texture Mean
Perimeter Mean
Area Mean
Smoothness Mean
Concavity Mean
Symmetry Mean
etc.

**Dataset**
The dataset used for training the model is sourced from the UCI Machine Learning Repository.
Contributing

**Contributions are welcome!**
If you have any suggestions or improvements, feel free to fork the repository, create a new branch, and submit a pull request.
    **Fork the repo**
    Create a new branch (git checkout -b feature-branch)
    **Make your changes**
    **Commit your changes** (git commit -m 'Add feature')
    **Push to the branch** (git push origin feature-branch)
    **Open a pull request**

**License**
This project is licensed under the MIT License.

**Contact**
For any inquiries or feedback, feel free to reach out:

    GitHub: kunratha
    Email: kean.kun.ratha2020@gmail.com
