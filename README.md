# RI_2023_Divlje_Macke

Projekat radili:
- Vladimir Gološin SV2/2020, Konvolucijska neuronska mreža
- Kristina Andrijin SV26/2020, (obična) neuronska mreža (MLP)

## Pokretanje konvolucijske neuronske mreže:

Da bi se pokrenuo program potrebno je pokrenuti main.py unutar "convolutional neural network" foldera. Da bi projekat radio potrebno je instalirati odgovarajuće module u virtualno okruženje. Komande za to su:

pip install pandas

pip install numpy

pip install tensorflow

pip install matplotlib

pip install seaborn

pip install scikit-learn


Ako postoji folder sa nazivom "trained_model" unutar model foldera, program će iskoristiti taj model da obavi test fazu i predviđanja. Ako ne postoji, odpočinje se treniranje modela.

Kada se završi treniranje modela, grafikoni, završni model i modeli najboljih performansi se čuvaju unutar export foldera. Ako želite da pokrenete neke od modela iz export foldera potrebno ih je preimenovati u "trained_model" i premestiti u model folder.

## Pokretanje neuronske mreže:
Da bi projekat radio potrebno je instalirati odgovarajuće module u virtualno okruženje. Komande za to su:
pip install os

pip install torch torchvision

pip install matplotlib

pip install numpy

pip install Pillow


Treniranje neuronske mreže se postiže pokretanjem main funkcije u fajlu neural_network/neural_network.py, dok se učitavnje modela postiže pokretanjem maina u neural_network/load_model.py, preimenovanjem modela u "best_model.pth" i prebacivanjem modela u folder neural_network/best. Moguće je raditi ručno testiranje fotografija njihovim prebacivanjem u folder RI_2023_Divlje_Macke i pisanjem naziva (sa ekstenzijom) kao trećeg parametra funkcije classify, kako je urađeno u primeru. 

classify(model, image_transform, 'african_leopard_8.jpg', classes, 'african_leopard'). 

Poželjno je napisati koja životinja je u pitanju, u suprotnom to neće biti naznačeno jer je nepoznato, s obzirom da neuronska mreža nema previše pameti :-(
Moguće je da se novim treniranjem modela neće dobiti parametri sa postera, s obzirom da je do tog modela došlo uz dosta hyperparameters tunninga.
