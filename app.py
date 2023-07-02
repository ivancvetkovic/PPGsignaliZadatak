from flask import Flask, render_template
import pandas as pd
import numpy as np
import csv
import pywt

app = Flask(__name__)

@app.route('/')
def index():
    # Čitanje transformisanih podataka iz CSV fajla
    df_transformed = pd.read_csv('transformed_data.csv')

    # Konvertovanje 'Coefficients' iz stringa sa zarezima u listu brojeva
    df_transformed['Coefficients'] = df_transformed['Coefficients'].apply(lambda x: list(map(float, x.replace('  ', ',').split(','))))

    # Pretpostavljajući da su ulazni podaci lista listi
    # gde je svaka unutrašnja lista poseban PPG signal
    ppg_signals = df_transformed['Coefficients'].to_list()  # Koristi transformisane PPG signale iz DataFrame-a

    return render_template('index.html', ppg_signals=ppg_signals)

if __name__ == '__main__':
    app.run(debug=True)
