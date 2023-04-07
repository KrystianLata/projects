## Dokumnetacja projektu

W zadaniu został stworzony model konwolucyjnej sieci neuronowej klasyfikującej przedstawiane na obrazach znaki amerykańskiego języka migowego.

Dostępne był obrazy w podfolderach, dla każdej z dostępnych klas: 0, A, B lub C

Stworzony skrypt poprwanie wczytuje obrazy z podfolerów, jednocześnie tworząc odpowiednie etykiety dla danych.

Dane zostają znormalizowane, podzielone na zbiory treningowe, walidacyjne i testowe w proporcji 80/10/10, a następnie zaugmentowane za pomocą takich parametów jak:
* losowe obracanie zdjęć pod kątem 10 stopni
* losowe poziome przesunięcie obrazu
* losowe odbicie na płaszczyźnie poziomej

### Stoworzony model

Do rozwiązania problemu klasyfikacji stworzoyn został model zawierający:
* 4 warstwy splotowe
* 4 warstwy MaxPooling
* warstą flatten
* 2 warstwami dense, z 512 neuronami w pierwszej z nich

Model został skompilowany przy użycio opimizera RMSprop, z ustawionym checkpointem zapisującym najlepszy z modeli

Tak skonsturowana sieć przy testowaniu osiągnęła zadawalające metryki już przy trzeciej epoce, które lekko poprawiały się w następnych epokach.
Ostatecznie zapisany został model z ósmej epoki, w której validation loss osiągnęło poziom 0.00477.

Ostateczne metryki na poziomie testowym były na poziomie:
Accuracy: 0.9917
Precision: 0.9919
Recall: 0.9917
F1-score: 0.9917

Co jest wynikiem satysfakcjonującym, ponieważ oznaczają że model prawidłowo klasyfikuje niemal wszystkie znaki na zbiorze danych, których jeszcze nie widział.

Możliwe byłoby spróbowanie trenowania modelu przez większą liczbę epok, która na porzeby zadania została zredukowana do 10 ze względu na ograniczenia czasowe

Ostatecznie skrypt generuje wykres krzywych uczenia na zbiorze trenignowym i walidacyjnym. Zgodnie z wykresem model nie ma problemu z overfittingiem ani underfittingiem

Następnie skrypt generuje 10 losowych predykcji z opisem jaka była wartość rzeczywista, a jaka przewidziana. 


Podsumowując, metryki modelu i wykres krzywych uczenia wskazują na to, że model jest odpowiednio przetrenowany i dobrze radzi sobie z klasyfikacją obrazów z dostarczonego zbioru.
Model sieci neuronowej po uruomieniu skryptu zapisze się w podfolderze best_model (nie został dodany do repozytorium ze wzgledu na zbyt duży rozmiar)





W projekcie zostały wykorzystanne następujace biblioteki

- tensorflow==2.7.0
- scikit-learn
- seaborn 
- matplotlib
- random
- numpy
- os
- Pillow 
