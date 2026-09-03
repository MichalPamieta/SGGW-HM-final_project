import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Definicja funkcji celu
def f(x, y):
    return 20 + (x**2 - 10 * np.cos(2 * np.pi * x)) + (y**2 - 10 * np.cos(2 * np.pi * y))

# Obliczenia siatki [-5.12,5.12]x[-5.12,5.12]
x_vals = np.linspace(-5.12, 5.12, 200)
y_vals = np.linspace(-5.12, 5.12, 200)
x, y = np.meshgrid(x_vals, y_vals)
z = f(x, y)

# Tworzenie wykresu 3D (10,-40)
fig_3d_1 = plt.figure(figsize=(10, 7))
ax_3d_1 = fig_3d_1.add_subplot(111, projection='3d')
surf_3d_1 = ax_3d_1.plot_surface(x, y, z, cmap='viridis', edgecolor='none')

# Ustawienie początkowej pozycji widoku
ax_3d_1.view_init(elev=30, azim=-135)  # Ustawienie początkowego kąta widzenia

# Tworzenie wykresu 3D (0,0)
fig_3d_2 = plt.figure(figsize=(10, 7))
ax_3d_2 = fig_3d_2.add_subplot(111, projection='3d')
surf_3d_2 = ax_3d_2.plot_surface(x, y, z, cmap='viridis', edgecolor='none')

# Ustawienie początkowej pozycji widoku
ax_3d_2.view_init(elev=0, azim=-135)  # Ustawienie początkowego kąta widzenia

# Wyświetlenie wykresów
plt.show()  # Zatrzymanie wykonania kodu do momentu zamknięcia wykresu

# Funkcja animacji do pełnego obrotu
def rotate_animation(frame):
    if frame < 180:  # Pierwsza połowa animacji
        ax_anim.view_init(elev=frame, azim=270)
    else:
        ax_anim.view_init(elev=360-frame, azim=270)  # Kontynuacja obrotu

# Tworzenie nowej figury do animacji
fig_anim = plt.figure(figsize=(10, 7))
ax_anim = fig_anim.add_subplot(111, projection='3d')
surf_anim = ax_anim.plot_surface(x, y, z, cmap='viridis', edgecolor='none')

# Zapis animacji
ani_3d = FuncAnimation(fig_anim, rotate_animation, frames=np.linspace(0, 360, 180), interval=50)

# Zapisz animację
ani_3d.save('rotating plot 2d (rastrigin).gif', dpi=120, writer='pillow', fps=15)

# Znajdowanie minimum globalnego
x_min = x.ravel()[z.argmin()]
y_min = y.ravel()[z.argmin()]

# Parametry algorytmu PSO
w = 0.8
c1 = 0.1
c2 = 0.2

# Tworzenie cząstek
n_particles = 30
X = np.random.rand(2, n_particles) * 10.24 - 5.12
V = np.random.randn(2, n_particles) * 0.1

# Inicjalizacja zmiennych
pbest = X.copy()
pbest_obj = f(X[0], X[1])
gbest = pbest[:, pbest_obj.argmin()]
gbest_obj = pbest_obj.min()

# Funkcja wykonująca jedną iterację PSO
def update_pso():
    global V, X, pbest, pbest_obj, gbest, gbest_obj
    # Aktualizacja parametrów
    r1, r2 = np.random.rand(2)
    V = w * V + c1 * r1 * (pbest - X) + c2 * r2 * (gbest.reshape(-1, 1) - X)
    X = X + V
    obj = f(X[0], X[1])
    mask = obj < pbest_obj
    pbest[:, mask] = X[:, mask]
    pbest_obj[mask] = obj[mask]
    if obj.min() < gbest_obj:
        gbest = X[:, obj.argmin()]
        gbest_obj = obj.min()

# Ustawienie podstawowego wykresu: mapa konturowa
fig_contour, ax = plt.subplots(figsize=(8, 6))
fig_contour.set_tight_layout(True)
img = ax.imshow(z, extent=[-5.12, 5.12, -5.12, 5.12], origin='lower', cmap='viridis', alpha=0.5)
fig_contour.colorbar(img, ax=ax)
ax.plot([x_min], [y_min], marker='x', markersize=7, color="white")
contours = ax.contour(x, y, z, 10, colors='black', alpha=0.4)
ax.clabel(contours, inline=True, fontsize=8, fmt="%.0f")

# Aktualizacja pozycji cząstek
pbest_plot = ax.scatter(pbest[0], pbest[1], marker='o', color='black', alpha=0.5)
p_plot = ax.scatter(X[0], X[1], marker='o', color='blue', alpha=0.8)
p_arrow = ax.quiver(X[0], X[1], V[0], V[1], color='blue', width=0.005, angles='xy', scale_units='xy', scale=1)
gbest_plot = ax.scatter([gbest[0]], [gbest[1]], marker='*', s=350, color='black', alpha=0.8)
ax.set_xlim([-5.12, 5.12])
ax.set_ylim([-5.12, 5.12])

# Kroki PSO: aktualizacja algorytmu i wyświetlanie na wykresie
def animate(i):
    title = f'Iteration {i:02d}'
    update_pso()
    ax.set_title(title)
    pbest_plot.set_offsets(pbest.T)
    p_plot.set_offsets(X.T)
    p_arrow.set_offsets(X.T)
    p_arrow.set_UVC(V[0], V[1])
    gbest_plot.set_offsets(gbest.reshape(1, -1))
    return ax, pbest_plot, p_plot, p_arrow, gbest_plot

# Tworzenie animacji PSO
anim_pso = FuncAnimation(fig_contour, animate, frames=range(1, 51), interval=500, blit=False, repeat=True)
anim_pso.save("pso 2d (rastrigin).gif", dpi=120, writer="pillow")

print(f"PSO znalazł najlepsze rozwiązanie w f({gbest[0]:.4f}, {gbest[1]:.4f}) = {gbest_obj:.4f}")