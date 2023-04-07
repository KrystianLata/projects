## Dokumnetacja projektu

W zadaniu został stworzony model konwolucyjnej sieci neuronowej klasyfikującej przedstawiane na obrazach znaki amerykańskiego języka migowego.

Dostępne był obrazy w podfolderach, dla każdej z dostępnych klas: 0, A, B lub C

Stworzony skrypt poprwanie wczytuje obrazy z podfolerów, jednocześnie tworząc odpowiednie etykiety dla danych.

Dane zostają znormalizowane, podzielone na zbiory treningowe, walidacyjne i testowe w proporcji 80/10/10, a następnie zaugmentowane za pomocą takich parametów jak:
* losowe obracanie zdjęć pod kątem 10 stopni
* losowe poziome przesunięcie obrazu
* losowe odbicie na płaszczyźnie poziomej

