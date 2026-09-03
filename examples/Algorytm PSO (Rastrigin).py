import numpy as np

# Funkcja celu: minimalizacja funkcji kwadratowej
def funkcja_celu(x):
    return 10 + x**2 - 10 * np.cos(2 * np.pi * x)

# Klasa cząstki
class Cząstka:
    def __init__(self, granica_min, granica_max):
        # Losowa inicjalizacja pozycji i prędkości (przyjmujemy 1 wymiar)
        self.pozycja = np.random.uniform(granica_min, granica_max)
        self.prędkość = np.random.uniform(-1, 1)
        self.pozycja_najlepsza = np.copy(self.pozycja)
        self.najlepsza_wartość = funkcja_celu(self.pozycja)
    
    def aktualizuj_prędkość(self, globalna_najlepsza, w, c1, c2):
        r1 = np.random.rand()
        r2 = np.random.rand()
        
        # Aktualizacja prędkości cząstki
        część_inercyjna = w * self.prędkość
        część_kognitywna = c1 * r1 * (self.pozycja_najlepsza - self.pozycja)
        część_socjalna = c2 * r2 * (globalna_najlepsza - self.pozycja)
        
        self.prędkość = część_inercyjna + część_kognitywna + część_socjalna

    def aktualizuj_pozycję(self, granica_min, granica_max):
        # Aktualizacja pozycji cząstki
        self.pozycja += self.prędkość
        # Zabezpieczenie przed wyjściem poza granice
        self.pozycja = np.clip(self.pozycja, granica_min, granica_max)

    def aktualizuj_osobiste_najlepsze(self):
        # Sprawdzenie, czy obecna pozycja jest lepsza niż osobista najlepsza
        wartość = funkcja_celu(self.pozycja)
        if wartość < self.najlepsza_wartość:
            self.pozycja_najlepsza = np.copy(self.pozycja)
            self.najlepsza_wartość = wartość

# Klasa PSO
class PSO:
    def __init__(self, rozmiar_populacji, granica_min, granica_max, w=0.5, c1=1.5, c2=1.5):
        # Inicjalizacja parametrów
        self.czastki = [Cząstka(granica_min, granica_max) for _ in range(rozmiar_populacji)]
        self.globalna_najlepsza_pozycja = None
        self.globalna_najlepsza_wartość = float('inf')
        self.w = w  # Współczynnik inercji
        self.c1 = c1  # Współczynnik komponentu kognitywnego
        self.c2 = c2  # Współczynnik komponentu socjalnego
    
    def optymalizuj(self, liczba_iteracji):
        # Pętla główna PSO
        for iteracja in range(liczba_iteracji):
            for czastka in self.czastki:
                # Aktualizacja osobistego najlepszego rozwiązania
                czastka.aktualizuj_osobiste_najlepsze()

                # Aktualizacja globalnego najlepszego rozwiązania
                if czastka.najlepsza_wartość < self.globalna_najlepsza_wartość:
                    self.globalna_najlepsza_wartość = czastka.najlepsza_wartość
                    self.globalna_najlepsza_pozycja = np.copy(czastka.pozycja_najlepsza)
            
            # Aktualizacja prędkości i pozycji cząstek
            for czastka in self.czastki:
                czastka.aktualizuj_prędkość(self.globalna_najlepsza_pozycja, self.w, self.c1, self.c2)
                czastka.aktualizuj_pozycję(granica_min, granica_max)
            
            print(f"Iteracja {iteracja+1}/{liczba_iteracji}, Globalna najlepsza wartość: {self.globalna_najlepsza_wartość}")
        
        return self.globalna_najlepsza_pozycja, self.globalna_najlepsza_wartość

# Parametry algorytmu
rozmiar_populacji = 30
granica_min = -5.12
granica_max = 5.12
liczba_iteracji = 50

# Inicjalizacja i uruchomienie algorytmu PSO
pso = PSO(rozmiar_populacji, granica_min, granica_max, 0.8, 0.1, 0.2)
najlepsza_pozycja, najlepsza_wartość = pso.optymalizuj(liczba_iteracji)

# Wyświetlenie wyników i animacji trajektorii cząstek
print(f"Najlepsze rozwiązanie: {najlepsza_pozycja:.4f}, Wartość funkcji celu: {najlepsza_wartość:.4f}")
