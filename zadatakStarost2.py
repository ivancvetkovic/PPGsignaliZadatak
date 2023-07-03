import pandas as pd
import numpy as np
import csv
import pywt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# Čitanje Excel fajla
df = pd.read_excel('db1.xlsx')

# Izdvajanje kolona sa ID-jem i uzrastom
id_age_df = df[['data_id', 'age']]

# Izdvajanje kolona sa PPG podacima
ppg_data_df = df.iloc[:, 2:]  # Izaberi kolone od C pa nadalje

# Kreiranje novog DataFrame-a sa ID-jem, uzrastom i PPG podacima
output_df = pd.DataFrame(columns=['id', 'age', 'data'])
for idx, row in id_age_df.iterrows():
    data_id = row['data_id']
    age = row['age']
    ppg_data = ppg_data_df.iloc[idx].dropna()
    ppg_data = pd.to_numeric(ppg_data, errors='coerce')
    ppg_data = ppg_data[~np.isnan(ppg_data)].tolist()
    output_df = pd.concat([output_df, pd.DataFrame({'id': data_id, 'age': age, 'data': [ppg_data]})], ignore_index=True)

# Sačuvaj DataFrame u CSV fajl
output_df.to_csv('output.csv', index=False)
print("Podaci sačuvani u output.csv.")

# Definiši wavelet funkciju i nivo dekompozicije
wavelet = 'db3'
level = 5

# Čitanje ulaznog CSV fajla
with open('sredjenExecl.csv', 'r') as file:
    reader = csv.reader(file)
    next(reader)  # Preskoči zaglavlje ako postoji

    # Kreiranje CSV fajla za čuvanje wavelet koeficijenata
    with open('transformed_data.csv', 'w', newline='') as output_file:
        writer = csv.writer(output_file)
        writer.writerow(['Data ID', 'Age', 'Coefficients'])  # Upisivanje zaglavlja

        for row in reader:
            data_id = row[0]
            age = row[1]
            ppg_data = [float(num) for num in row[2][1:-1].split(',') if num.strip() != '']  # Pretvaranje stringa u listu, preskačući prazne vrednosti

            # Izvršavanje wavelet transformacije
            coeffs = pywt.wavedec(ppg_data, wavelet, level=level)

            # Pretvaranje koeficijenata u string reprezentaciju
            coeffs_str = ','.join(str(c) for c in coeffs[0])

            # Upisivanje podataka u izlazni CSV fajl
            writer.writerow([data_id, age, coeffs_str])

    print("Wavelet transformacija završena i rezultati sačuvani u transformed_data.csv.")

# Učitavanje sačuvanog modela
model = load_model('Epohe2000inputNurons512_waveletdb1.h5')

# Čitanje transformisanih podataka iz CSV fajla
df_transformed = pd.read_csv('transformed_data.csv')

# Konvertovanje 'Coefficients' iz stringa sa zarezima u listu brojeva
df_transformed['Coefficients'] = df_transformed['Coefficients'].apply(lambda x: list(map(float, x.replace('  ', ',').split(','))))

# Pretpostavljajući da su ulazni podaci lista listi
# gde je svaka unutrašnja lista poseban PPG signal
ppg_signals = df_transformed['Coefficients'].to_list()  # Koristi transformisane PPG signale iz DataFrame-a

correct_ages = df_transformed['Age'].to_list()  # Koristi odgovarajuće tačne uzraste iz DataFrame-a

# Provera tipova podataka i podizanje izuzetka ako postoji problem
for signal in ppg_signals:
    assert isinstance(signal, list), f"Nevažeći signal: {signal}"

# Ako je model treniran na dopunjenim sekvencama,
# obrati pažnju da dopuniš ulazne podatke na isti način
ppg_signals = pad_sequences(ppg_signals, maxlen=599, truncating='post', padding='post')

# Predvidi uzrast za svaki PPG signal
ages_predicted = model.predict(ppg_signals)

# Izračunavanje metrika greške
mae = mean_absolute_error(correct_ages, ages_predicted)
mse = mean_squared_error(correct_ages, ages_predicted)
print(f"Srednja apsolutna greška: {mae}")
print(f"Srednja kvadratna greška: {mse}")

# Ako želiš da ispiseš svaku pojedinačnu predikciju zajedno sa stvarnim uzrastom
for i in range(len(correct_ages)):
    print(f"Zadata starost: {correct_ages[i]}, Predviđena starost: {ages_predicted[i][0]}")

# Prikaz svakog signala posebno
for i in range(len(ppg_signals)):
    signal = ppg_signals[i]
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    reconstructed_signal = pywt.waverec(coeffs[:level+1], wavelet)
    
    # Kreiranje subplot-a za originalni signal
    plt.subplot(2, 1, 1)
    plt.plot(signal)
    plt.title(f'Originalni signal - Signal {i + 1}')
    plt.xlabel('Vremenski odbirak')
    plt.ylabel('Amplituda')

    # Kreiranje subplot-a za rekonstruisani signal
    plt.subplot(2, 1, 2)
    plt.plot(reconstructed_signal)
    plt.title(f'Rekonstruisani signal - Signal {i + 1}')
    plt.xlabel('Vremenski odbirak')
    plt.ylabel('Amplituda')
    
    plt.tight_layout()
    plt.show()

