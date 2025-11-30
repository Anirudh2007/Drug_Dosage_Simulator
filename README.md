📘 Theophylline Pharmacokinetic Simulator (ODE-Based)

An interactive pharmacokinetic (PK) simulator built using ordinary differential equations, RK4 numerical integration, and real clinical data from the Theophylline dataset.
The project models oral drug absorption and elimination, performs multi-dose simulations, and automatically recommends optimal fixed-interval dosing schedules.

🚀 Features

✔️ One-compartment PK ODE model

✔️ Calibrated using real clinical Theophylline data

✔️ Numerical solution via Runge–Kutta 4 (RK4)

✔️ Multi-dose simulation with user-controlled:

dose

dosing interval

number of doses

total simulation time

✔️ Therapeutic window analysis (MEC–MTC)

✔️ Automated fixed-interval recommendation engine

✔️ Fully interactive Streamlit interface

📂 Repository Structure
src/                 - main Python source code
report/              - full LaTeX report and PDF
plots/               - sample output figures
requirements.txt     - dependencies
README.md            - project documentation

📄 Running the Application
Install dependencies:
pip install -r requirements.txt

Run Streamlit app:
streamlit run src/pk_interactive_app.py

📊 Dataset

This project uses the publicly available Theoph dataset from the R datasets package:

Direct CSV link:
https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/datasets/Theoph.csv

Data is downloaded automatically by the code.

📚 Full Report

A full technical report (PDF + LaTeX source) is included in report.

🧑‍💻 Author

Anirudh Jain, Ayush Pandey, Sourabh, Vedant Kumar
Cluster Innovation Centre, University of Delhi
