import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib import animation
import tkinter as tk
from tkinter import ttk, messagebox

# Parametry modelu domyślne
masa = 1.0  # masa
sztywnosc = 10.0  # sztywność sprężyny
tlumienie = 0.5  # współczynnik tłumienia
warunki_poczatkowe = [1.0, 0.0]  # warunki początkowe: przemieszczenie 1, prędkość 0
czas = np.linspace(0, 10, 1000)  # przedział czasowy

# System równań różniczkowych
def oscylator_harmoniczny(stan, t, masa, sztywnosc, tlumienie):
    x1, x2 = stan
    dx1_dt = x2
    dx2_dt = -(sztywnosc/masa) * x1 - (tlumienie/masa) * x2
    return [dx1_dt, dx2_dt]

# Funkcja do uruchamiania symulacji
def uruchom_symulacje():
    global masa, sztywnosc, tlumienie, czas
    try:
        masa = float(pole_masa.get())
        sztywnosc = float(pole_sztywnosc.get())
        tlumienie = float(pole_tlumienie.get())
        
        # Rozwiązanie równań
        rozwiazanie = odeint(oscylator_harmoniczny, warunki_poczatkowe, czas, args=(masa, sztywnosc, tlumienie))
        przemieszczenie = rozwiazanie[:, 0]
        predkosc = rozwiazanie[:, 1]

        # Tworzenie wykresów
        fig = plt.figure(figsize=(12, 5))
        
        # Wykres przemieszczenia i prędkości
        plt.subplot(1, 2, 1)
        plt.plot(czas, przemieszczenie, label='Przemieszczenie (x)')
        plt.plot(czas, predkosc, label='Prędkość (v)')
        plt.xlabel('Czas (s)')
        plt.ylabel('Wartość')
        plt.title('Przemieszczenie i prędkość')
        plt.legend()
        plt.grid(True)

        # Portret fazowy
        plt.subplot(1, 2, 2)
        plt.plot(przemieszczenie, predkosc)
        plt.xlabel('Przemieszczenie (x)')
        plt.ylabel('Prędkość (v)')
        plt.title('Portret fazowy')
        plt.grid(True)

        plt.tight_layout()

        # Animacja
        fig_anim = plt.figure(figsize=(6, 4))
        ax = fig_anim.add_subplot(111)
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-0.5, 0.5)
        ax.set_xlabel('Przemieszczenie (x)')
        ax.set_title('Animacja oscylatora harmonicznego')
        linia, = ax.plot([], [], 'bo-', lw=2, markersize=10)  # Niebieski punkt z linią
        linia_sprezyny, = ax.plot([], [], 'b-', lw=1)  # Linia sprężyny

        def init():
            linia.set_data([], [])
            linia_sprezyny.set_data([], [])
            ax.set_ylim(-0.5, 0.5)
            return linia, linia_sprezyny

        def animuj(i):
            pozycja_x = przemieszczenie[i]
            linia.set_data([0, pozycja_x], [0, 0])
            linia_sprezyny.set_data([0, pozycja_x], [0, 0])
            return linia, linia_sprezyny

        ani = animation.FuncAnimation(fig_anim, animuj, init_func=init, frames=len(czas), interval=20, blit=True)
        
        plt.show()
    except ValueError:
        messagebox.showerror("Błąd", "Proszę wprowadzić wartości liczbowe dla parametrów.")

# Tworzenie GUI
okno = tk.Tk()
okno.title("Oscylator harmoniczny")

# Pola wprowadzania parametrów
tk.Label(okno, text="Masa (m):").grid(row=0, column=0, padx=5, pady=5)
pole_masa = tk.Entry(okno)
pole_masa.insert(0, "1.0")
pole_masa.grid(row=0, column=1, padx=5, pady=5)

tk.Label(okno, text="Sztywność (k):").grid(row=1, column=0, padx=5, pady=5)
pole_sztywnosc = tk.Entry(okno)
pole_sztywnosc.insert(0, "10.0")
pole_sztywnosc.grid(row=1, column=1, padx=5, pady=5)

tk.Label(okno, text="Tłumienie (b):").grid(row=2, column=0, padx=5, pady=5)
pole_tlumienie = tk.Entry(okno)
pole_tlumienie.insert(0, "0.5")
pole_tlumienie.grid(row=2, column=1, padx=5, pady=5)

# Przycisk do uruchamiania symulacji
tk.Button(okno, text="Uruchom symulację", command=uruchom_symulacje).grid(row=3, column=0, columnspan=2, pady=10)

# Uruchomienie GUI
okno.mainloop()