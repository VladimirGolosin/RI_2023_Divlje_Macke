## RI_2023_Divlje_Macke

Projekat radili:
- Vladimir Gološin SV2/2020, Konvolucijska neuronska mreža
- Kristina Andrijin SV26/2020, (obična) Neuronska mreža

# Pokretanje konvolucijske neuronske mreže:

Da bi se pokrenuo program potrebno je pokrenuti main.py unutar "convolutional neural network" foldera. Da bi projekat radio potrebno je instalirati odgovarajuće module u virtualno okruženje. Komande za to su:

pip install pandas
pip install numpy
pip install tensorflow
pip install matplotlib
pip install seaborn
pip install scikit-learn

Ako postoji folder sa nazivom "trained_model" unutar model foldera, program će iskoristiti taj model da obavi test fazu i predviđanja. Ako ne postoji, odpočinje se treniranje modela.

Kada se završi treniranje modela, grafikoni, završni model i modeli najboljih performansi se čuvaju unutar export foldera. Ako želite da pokrenete neke od modela iz export foldera potrebno ih je preimenovati u "trained_model" i premestiti u model folder.

# Pokretanje neuronske mreže:


