import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Funkcje celu
def f_Example(x, y):
    return (x - 3.14)**2 + (y - 2.72)**2 + np.sin(3 * x + 1.41) + np.sin(4 * y - 1.73)

def f_Rastrigin(x, y):
    return 20 + (x**2 - 10 * np.cos(2 * np.pi * x)) + (y**2 - 10 * np.cos(2 * np.pi * y))

def f_Ackley(x, y):
    return -20 * np.exp(-0.2 * np.sqrt(0.5 * (x**2 + y**2))) - np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y))) + np.e + 20

def f_Sphere(x, y):
    return x**2 + y**2

def f_Rosenbrock(x, y):
    return 100 * (y - x**2)**2 + (1 - x)**2

def f_Beale(x, y):
    return (1.5 - x + x * y)**2 + (2.25 - x + x * y**2)**2 + (2.625 - x + x * y**3)**2

def f_Goldstein_Price(x, y):
    return (1 + (x + y + 1)**2 * (19 - 14 * x + 3 * x**2 - 14 * y + 6 * x * y + 3 * y**2)) * (30 + (2 * x - 3 * y)**2 * (18 - 32 * x + 12 * x**2 + 48 * y - 36 * x * y + 27 * y**2))

def f_Booth(x, y):
    return (x + 2 * y - 7)**2 + (2 * x + y - 5)**2

def f_BukinN6(x, y):
    return 100 * np.sqrt(np.abs(y - 0.01 * x**2)) + 0.01 * np.abs(x + 10)

def f_Matyas(x, y):
    return 0.26 * (x**2 + y**2) - 0.48 * x * y

def f_LeviN13(x, y):
    return np.sin(3 * np.pi * x)**2 + (x - 1)**2 * (1 + np.sin(3 * np.pi * y)**2) + (y - 1)**2 * (1 + np.sin(2 * np.pi * y)**2)

def f_Himmelblau(x, y):
    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2

def f_Three_hump_camel(x, y):
    return 2 * x**2 - 1.05 * x**4 + (x**6)/6 + x * y + y**2

def f_Easom(x, y):
    return -np.cos(x) * np.cos(y) * np.exp(-((x - np.pi)**2 + (y - np.pi)**2))

def f_Cross_in_tray(x, y):
    return -0.0001 * (np.abs(np.sin(x) * np.sin(y) * np.exp(np.abs(100 - (np.sqrt(x**2 + y**2)/np.pi)))) + 1)**0.1

def f_Eggholder(x, y):
    return -(y + 47) * np.sin(np.abs(x / 2 + (y + 47))) - x * np.sin(np.sqrt(np.abs(x - (y + 47))))

def f_Holder_table(x, y):
    return -np.abs(np.sin(x) * np.cos(y) * np.exp(np.abs(1 - (np.sqrt(x**2 + y**2) / np.pi))))

def f_McCormick(x, y):
    return np.sin(x + y) + (x - y)**2 - 1.5 * x + 2.5 * y + 1

def f_SchafferN2(x, y):
    return 0.5 + (np.sin(x**2 - y**2)**2 - 0.5) / (1 + 0.001 * (x**2 + y**2))**2

def f_SchafferN4(x, y):
    return 0.5 + (np.cos(np.sin(np.abs(x**2 - y**2)))**2 - 0.5) / (1 + 0.001 * (x**2 + y**2))**2

def f_Styblinski_Tang(x, y):
    return (x**4 - 16 * x**2 + 5 * x + y**4 - 16 * y**2 + 5 * x) / 2



# Lista dostępnych funkcji
objective_functions = {
    "Example": f_Example,
    "Rastrigin": f_Rastrigin,
    "Ackley": f_Ackley,
    "Sphere": f_Sphere,
    "Rosenbrock": f_Rosenbrock,
    "Beale": f_Beale,
    "Goldstein-Price": f_Goldstein_Price,
    "Booth": f_Booth,
    "Bukin function N.6": f_BukinN6,
    "Matyas": f_Matyas,
    "Levi function N.13": f_LeviN13,
    "Himmelblau": f_Himmelblau,
    "Three hump camel": f_Three_hump_camel,
    "Easom": f_Easom,
    "Cross-in-tray": f_Cross_in_tray,
    "Eggholder": f_Eggholder,
    "Holder table": f_Holder_table,
    "McCormick": f_McCormick,
    "Schaffer function N.2": f_SchafferN2,
    "Schaffer function N.4": f_SchafferN4,
    "Styblinski-Tang": f_Styblinski_Tang
}

# PSO z wizualizacją
def particle_swarm_optimization(func, iterations, n_particles, w, c1, c2, x_bounds, y_bounds, generate_gif):
    x_vals = np.linspace(x_bounds[0], x_bounds[1], 200)
    y_vals = np.linspace(y_bounds[0], y_bounds[1], 200)
    x, y = np.meshgrid(x_vals, y_vals)
    z = func(x, y)

    x_min = x.ravel()[z.argmin()]
    y_min = y.ravel()[z.argmin()]
    func_min_value = z.min()

    X = np.random.rand(2, n_particles)
    X[0] = X[0] * (x_bounds[1] - x_bounds[0]) + x_bounds[0]
    X[1] = X[1] * (y_bounds[1] - y_bounds[0]) + y_bounds[0]
    V = np.random.randn(2, n_particles) * 0.1

    pbest = X.copy()
    pbest_obj = func(X[0], X[1])
    gbest = pbest[:, pbest_obj.argmin()]
    gbest_obj = pbest_obj.min()

    fig_contour, ax = plt.subplots(figsize=(8, 6))
    fig_contour.set_tight_layout(True)
    img = ax.imshow(z, extent=[x_bounds[0], x_bounds[1], y_bounds[0], y_bounds[1]], 
                    origin='lower', cmap='viridis', alpha=0.5)
    fig_contour.colorbar(img, ax=ax)
    ax.plot([x_min], [y_min], marker='x', markersize=5, color="white")
    contours = ax.contour(x, y, z, 10, colors='black', alpha=0.4)
    ax.clabel(contours, inline=True, fontsize=8, fmt="%.0f")

    pbest_plot = ax.scatter(pbest[0], pbest[1], marker='o', color='black', alpha=0.5)
    p_plot = ax.scatter(X[0], X[1], marker='o', color='blue', alpha=0.8)
    p_arrow = ax.quiver(X[0], X[1], V[0], V[1], color='blue', width=0.005, angles='xy', scale_units='xy', scale=1)
    gbest_plot = ax.scatter([gbest[0]], [gbest[1]], marker='*', s=350, color='black', alpha=0.5)
    ax.set_xlim(x_bounds)
    ax.set_ylim(y_bounds)

    def update_pso():
        nonlocal V, X, pbest, pbest_obj, gbest, gbest_obj
        r1, r2 = np.random.rand(2)
        V = w * V + c1 * r1 * (pbest - X) + c2 * r2 * (gbest.reshape(-1, 1) - X)
        X = X + V
        X[0] = np.clip(X[0], x_bounds[0], x_bounds[1])
        X[1] = np.clip(X[1], y_bounds[0], y_bounds[1])
        obj = func(X[0], X[1])
        mask = obj < pbest_obj
        pbest[:, mask] = X[:, mask]
        pbest_obj[mask] = obj[mask]
        if obj.min() < gbest_obj:
            gbest = X[:, obj.argmin()]
            gbest_obj = obj.min()

    def animate(i):
        update_pso()
        ax.set_title(f"Iteration {i+1}")
        p_plot.set_offsets(X.T)
        pbest_plot.set_offsets(pbest.T)
        gbest_plot.set_offsets(gbest.reshape(1, -1))
        p_arrow.set_offsets(X.T)
        p_arrow.set_UVC(V[0], V[1])
        return ax, pbest_plot, p_plot, p_arrow, gbest_plot

    anim_pso = FuncAnimation(fig_contour, animate, frames=iterations, interval=300, blit=False)

    if generate_gif:
        def rotate_animation(frame):
            if frame < 180:
                ax_anim.view_init(elev=frame, azim=270)
            else:
                ax_anim.view_init(elev=360-frame, azim=270)

        fig_anim = plt.figure(figsize=(10, 7))
        ax_anim = fig_anim.add_subplot(111, projection='3d')
        surf_anim = ax_anim.plot_surface(x, y, z, cmap='viridis', edgecolor='none')
        ani_3d = FuncAnimation(fig_anim, rotate_animation, frames=np.linspace(0, 360, 180), interval=50)

        ani_3d.save('rotating_plot.gif', dpi=120, writer='pillow', fps=20)
        anim_pso.save("pso_animation.gif", dpi=120, writer="pillow")
        messagebox.showinfo("GIFs Saved", "Animations saved as 'rotating_plot.gif' and 'pso_animation.gif'.")

    plt.show()
    result_message = (
        f"PSO found the best solution at f({gbest[0]:.4f}, {gbest[1]:.4f}) = {gbest_obj:.4f}\n"
        f"Global minimum in the function is at f({x_min:.4f}, {y_min:.4f}) = {func_min_value:.4f}")
    messagebox.showinfo("PSO Solution", result_message)
    


# GUI
def run_pso():
    try:
        iterations = int(iterations_entry.get())
        n_particles = int(particles_entry.get())
        w = float(w_entry.get())
        c1 = float(c1_entry.get())
        c2 = float(c2_entry.get())
        x_bounds = (float(x_min_entry.get()), float(x_max_entry.get()))
        y_bounds = (float(y_min_entry.get()), float(y_max_entry.get()))
        func_name = func_selector.get()
        func = objective_functions[func_name]
        generate_gif = gif_var.get()
        particle_swarm_optimization(func, iterations, n_particles, w, c1, c2, x_bounds, y_bounds, generate_gif)
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

def update_defaults(event):
    selected_function = func_selector.get()
    if selected_function == "Example":
        iterations_entry.delete(0, tk.END)
        iterations_entry.insert(0, "50")
        particles_entry.delete(0, tk.END)
        particles_entry.insert(0, "30")
        w_entry.delete(0, tk.END)
        w_entry.insert(0, "0.8")
        c1_entry.delete(0, tk.END)
        c1_entry.insert(0, "0.1")
        c2_entry.delete(0, tk.END)
        c2_entry.insert(0, "0.2")
        x_min_entry.delete(0, tk.END)
        x_min_entry.insert(0, "0")
        x_max_entry.delete(0, tk.END)
        x_max_entry.insert(0, "5")
        y_min_entry.delete(0, tk.END)
        y_min_entry.insert(0, "0")
        y_max_entry.delete(0, tk.END)
        y_max_entry.insert(0, "5")
    elif selected_function == "Rastrigin":
        iterations_entry.delete(0, tk.END)
        iterations_entry.insert(0, "50")
        particles_entry.delete(0, tk.END)
        particles_entry.insert(0, "30")
        w_entry.delete(0, tk.END)
        w_entry.insert(0, "0.8")
        c1_entry.delete(0, tk.END)
        c1_entry.insert(0, "0.1")
        c2_entry.delete(0, tk.END)
        c2_entry.insert(0, "0.2")
        x_min_entry.delete(0, tk.END)
        x_min_entry.insert(0, "-5.12")
        x_max_entry.delete(0, tk.END)
        x_max_entry.insert(0, "5.12")
        y_min_entry.delete(0, tk.END)
        y_min_entry.insert(0, "-5.12")
        y_max_entry.delete(0, tk.END)
        y_max_entry.insert(0, "5.12")
    elif selected_function == "Ackley":
        iterations_entry.delete(0, tk.END)
        iterations_entry.insert(0, "50")
        particles_entry.delete(0, tk.END)
        particles_entry.insert(0, "30")
        w_entry.delete(0, tk.END)
        w_entry.insert(0, "0.8")
        c1_entry.delete(0, tk.END)
        c1_entry.insert(0, "0.1")
        c2_entry.delete(0, tk.END)
        c2_entry.insert(0, "0.2")
        x_min_entry.delete(0, tk.END)
        x_min_entry.insert(0, "-5")
        x_max_entry.delete(0, tk.END)
        x_max_entry.insert(0, "5")
        y_min_entry.delete(0, tk.END)
        y_min_entry.insert(0, "-5")
        y_max_entry.delete(0, tk.END)
        y_max_entry.insert(0, "5")
    elif selected_function == "Sphere":
        iterations_entry.delete(0, tk.END)
        iterations_entry.insert(0, "50")
        particles_entry.delete(0, tk.END)
        particles_entry.insert(0, "30")
        w_entry.delete(0, tk.END)
        w_entry.insert(0, "0.8")
        c1_entry.delete(0, tk.END)
        c1_entry.insert(0, "0.1")
        c2_entry.delete(0, tk.END)
        c2_entry.insert(0, "0.2")
        x_min_entry.delete(0, tk.END)
        x_min_entry.insert(0, "-5.12")
        x_max_entry.delete(0, tk.END)
        x_max_entry.insert(0, "5.12")
        y_min_entry.delete(0, tk.END)
        y_min_entry.insert(0, "-5.12")
        y_max_entry.delete(0, tk.END)
        y_max_entry.insert(0, "5.12")
    elif selected_function == "Rosenbrock":
        iterations_entry.delete(0, tk.END)
        iterations_entry.insert(0, "50")
        particles_entry.delete(0, tk.END)
        particles_entry.insert(0, "30")
        w_entry.delete(0, tk.END)
        w_entry.insert(0, "0.8")
        c1_entry.delete(0, tk.END)
        c1_entry.insert(0, "0.1")
        c2_entry.delete(0, tk.END)
        c2_entry.insert(0, "0.2")
        x_min_entry.delete(0, tk.END)
        x_min_entry.insert(0, "-5")
        x_max_entry.delete(0, tk.END)
        x_max_entry.insert(0, "5")
        y_min_entry.delete(0, tk.END)
        y_min_entry.insert(0, "-5")
        y_max_entry.delete(0, tk.END)
        y_max_entry.insert(0, "5")
    elif selected_function == "Beale":
        iterations_entry.delete(0, tk.END)
        iterations_entry.insert(0, "50")
        particles_entry.delete(0, tk.END)
        particles_entry.insert(0, "30")
        w_entry.delete(0, tk.END)
        w_entry.insert(0, "0.8")
        c1_entry.delete(0, tk.END)
        c1_entry.insert(0, "0.1")
        c2_entry.delete(0, tk.END)
        c2_entry.insert(0, "0.2")
        x_min_entry.delete(0, tk.END)
        x_min_entry.insert(0, "-4.5")
        x_max_entry.delete(0, tk.END)
        x_max_entry.insert(0, "4.5")
        y_min_entry.delete(0, tk.END)
        y_min_entry.insert(0, "-4.5")
        y_max_entry.delete(0, tk.END)
        y_max_entry.insert(0, "4.5")
    elif selected_function == "Goldstein-Price":
        iterations_entry.delete(0, tk.END)
        iterations_entry.insert(0, "50")
        particles_entry.delete(0, tk.END)
        particles_entry.insert(0, "30")
        w_entry.delete(0, tk.END)
        w_entry.insert(0, "0.8")
        c1_entry.delete(0, tk.END)
        c1_entry.insert(0, "0.1")
        c2_entry.delete(0, tk.END)
        c2_entry.insert(0, "0.2")
        x_min_entry.delete(0, tk.END)
        x_min_entry.insert(0, "-2")
        x_max_entry.delete(0, tk.END)
        x_max_entry.insert(0, "2")
        y_min_entry.delete(0, tk.END)
        y_min_entry.insert(0, "-2")
        y_max_entry.delete(0, tk.END)
        y_max_entry.insert(0, "2")
    elif selected_function == "Booth":
        iterations_entry.delete(0, tk.END)
        iterations_entry.insert(0, "50")
        particles_entry.delete(0, tk.END)
        particles_entry.insert(0, "30")
        w_entry.delete(0, tk.END)
        w_entry.insert(0, "0.8")
        c1_entry.delete(0, tk.END)
        c1_entry.insert(0, "0.1")
        c2_entry.delete(0, tk.END)
        c2_entry.insert(0, "0.2")
        x_min_entry.delete(0, tk.END)
        x_min_entry.insert(0, "-10")
        x_max_entry.delete(0, tk.END)
        x_max_entry.insert(0, "10")
        y_min_entry.delete(0, tk.END)
        y_min_entry.insert(0, "-10")
        y_max_entry.delete(0, tk.END)
        y_max_entry.insert(0, "10")
    elif selected_function == "Bukin function N.6":
        iterations_entry.delete(0, tk.END)
        iterations_entry.insert(0, "50")
        particles_entry.delete(0, tk.END)
        particles_entry.insert(0, "30")
        w_entry.delete(0, tk.END)
        w_entry.insert(0, "0.8")
        c1_entry.delete(0, tk.END)
        c1_entry.insert(0, "0.1")
        c2_entry.delete(0, tk.END)
        c2_entry.insert(0, "0.2")
        x_min_entry.delete(0, tk.END)
        x_min_entry.insert(0, "-15")
        x_max_entry.delete(0, tk.END)
        x_max_entry.insert(0, "-5")
        y_min_entry.delete(0, tk.END)
        y_min_entry.insert(0, "-3")
        y_max_entry.delete(0, tk.END)
        y_max_entry.insert(0, "3")
    elif selected_function == "Matyas":
        iterations_entry.delete(0, tk.END)
        iterations_entry.insert(0, "50")
        particles_entry.delete(0, tk.END)
        particles_entry.insert(0, "30")
        w_entry.delete(0, tk.END)
        w_entry.insert(0, "0.8")
        c1_entry.delete(0, tk.END)
        c1_entry.insert(0, "0.1")
        c2_entry.delete(0, tk.END)
        c2_entry.insert(0, "0.2")
        x_min_entry.delete(0, tk.END)
        x_min_entry.insert(0, "-10")
        x_max_entry.delete(0, tk.END)
        x_max_entry.insert(0, "10")
        y_min_entry.delete(0, tk.END)
        y_min_entry.insert(0, "-10")
        y_max_entry.delete(0, tk.END)
        y_max_entry.insert(0, "10")
    elif selected_function == "Levi function N.13":
        iterations_entry.delete(0, tk.END)
        iterations_entry.insert(0, "50")
        particles_entry.delete(0, tk.END)
        particles_entry.insert(0, "30")
        w_entry.delete(0, tk.END)
        w_entry.insert(0, "0.8")
        c1_entry.delete(0, tk.END)
        c1_entry.insert(0, "0.1")
        c2_entry.delete(0, tk.END)
        c2_entry.insert(0, "0.2")
        x_min_entry.delete(0, tk.END)
        x_min_entry.insert(0, "-10")
        x_max_entry.delete(0, tk.END)
        x_max_entry.insert(0, "10")
        y_min_entry.delete(0, tk.END)
        y_min_entry.insert(0, "-10")
        y_max_entry.delete(0, tk.END)
        y_max_entry.insert(0, "10")
    elif selected_function == "Himmelblau":
        iterations_entry.delete(0, tk.END)
        iterations_entry.insert(0, "50")
        particles_entry.delete(0, tk.END)
        particles_entry.insert(0, "30")
        w_entry.delete(0, tk.END)
        w_entry.insert(0, "0.8")
        c1_entry.delete(0, tk.END)
        c1_entry.insert(0, "0.1")
        c2_entry.delete(0, tk.END)
        c2_entry.insert(0, "0.2")
        x_min_entry.delete(0, tk.END)
        x_min_entry.insert(0, "-5")
        x_max_entry.delete(0, tk.END)
        x_max_entry.insert(0, "5")
        y_min_entry.delete(0, tk.END)
        y_min_entry.insert(0, "-5")
        y_max_entry.delete(0, tk.END)
        y_max_entry.insert(0, "5")
    elif selected_function == "Three hump camel":
        iterations_entry.delete(0, tk.END)
        iterations_entry.insert(0, "50")
        particles_entry.delete(0, tk.END)
        particles_entry.insert(0, "30")
        w_entry.delete(0, tk.END)
        w_entry.insert(0, "0.8")
        c1_entry.delete(0, tk.END)
        c1_entry.insert(0, "0.1")
        c2_entry.delete(0, tk.END)
        c2_entry.insert(0, "0.2")
        x_min_entry.delete(0, tk.END)
        x_min_entry.insert(0, "-5")
        x_max_entry.delete(0, tk.END)
        x_max_entry.insert(0, "5")
        y_min_entry.delete(0, tk.END)
        y_min_entry.insert(0, "-5")
        y_max_entry.delete(0, tk.END)
        y_max_entry.insert(0, "5")
    elif selected_function == "Easom":
        iterations_entry.delete(0, tk.END)
        iterations_entry.insert(0, "50")
        particles_entry.delete(0, tk.END)
        particles_entry.insert(0, "30")
        w_entry.delete(0, tk.END)
        w_entry.insert(0, "0.8")
        c1_entry.delete(0, tk.END)
        c1_entry.insert(0, "0.1")
        c2_entry.delete(0, tk.END)
        c2_entry.insert(0, "0.2")
        x_min_entry.delete(0, tk.END)
        x_min_entry.insert(0, "-100")
        x_max_entry.delete(0, tk.END)
        x_max_entry.insert(0, "100")
        y_min_entry.delete(0, tk.END)
        y_min_entry.insert(0, "-100")
        y_max_entry.delete(0, tk.END)
        y_max_entry.insert(0, "100")
    elif selected_function == "Cross-in-tray":
        iterations_entry.delete(0, tk.END)
        iterations_entry.insert(0, "50")
        particles_entry.delete(0, tk.END)
        particles_entry.insert(0, "30")
        w_entry.delete(0, tk.END)
        w_entry.insert(0, "0.8")
        c1_entry.delete(0, tk.END)
        c1_entry.insert(0, "0.1")
        c2_entry.delete(0, tk.END)
        c2_entry.insert(0, "0.2")
        x_min_entry.delete(0, tk.END)
        x_min_entry.insert(0, "-10")
        x_max_entry.delete(0, tk.END)
        x_max_entry.insert(0, "10")
        y_min_entry.delete(0, tk.END)
        y_min_entry.insert(0, "-10")
        y_max_entry.delete(0, tk.END)
        y_max_entry.insert(0, "10")
    elif selected_function == "Eggholder":
        iterations_entry.delete(0, tk.END)
        iterations_entry.insert(0, "50")
        particles_entry.delete(0, tk.END)
        particles_entry.insert(0, "30")
        w_entry.delete(0, tk.END)
        w_entry.insert(0, "0.8")
        c1_entry.delete(0, tk.END)
        c1_entry.insert(0, "0.1")
        c2_entry.delete(0, tk.END)
        c2_entry.insert(0, "0.2")
        x_min_entry.delete(0, tk.END)
        x_min_entry.insert(0, "-512")
        x_max_entry.delete(0, tk.END)
        x_max_entry.insert(0, "512")
        y_min_entry.delete(0, tk.END)
        y_min_entry.insert(0, "-512")
        y_max_entry.delete(0, tk.END)
        y_max_entry.insert(0, "512")
    elif selected_function == "Holder table":
        iterations_entry.delete(0, tk.END)
        iterations_entry.insert(0, "50")
        particles_entry.delete(0, tk.END)
        particles_entry.insert(0, "30")
        w_entry.delete(0, tk.END)
        w_entry.insert(0, "0.8")
        c1_entry.delete(0, tk.END)
        c1_entry.insert(0, "0.1")
        c2_entry.delete(0, tk.END)
        c2_entry.insert(0, "0.2")
        x_min_entry.delete(0, tk.END)
        x_min_entry.insert(0, "-10")
        x_max_entry.delete(0, tk.END)
        x_max_entry.insert(0, "10")
        y_min_entry.delete(0, tk.END)
        y_min_entry.insert(0, "-10")
        y_max_entry.delete(0, tk.END)
        y_max_entry.insert(0, "10")
    elif selected_function == "McCormick":
        iterations_entry.delete(0, tk.END)
        iterations_entry.insert(0, "50")
        particles_entry.delete(0, tk.END)
        particles_entry.insert(0, "30")
        w_entry.delete(0, tk.END)
        w_entry.insert(0, "0.8")
        c1_entry.delete(0, tk.END)
        c1_entry.insert(0, "0.1")
        c2_entry.delete(0, tk.END)
        c2_entry.insert(0, "0.2")
        x_min_entry.delete(0, tk.END)
        x_min_entry.insert(0, "-1.5")
        x_max_entry.delete(0, tk.END)
        x_max_entry.insert(0, "4")
        y_min_entry.delete(0, tk.END)
        y_min_entry.insert(0, "-3")
        y_max_entry.delete(0, tk.END)
        y_max_entry.insert(0, "4")
    elif selected_function == "Schaffer function N.2":
        iterations_entry.delete(0, tk.END)
        iterations_entry.insert(0, "50")
        particles_entry.delete(0, tk.END)
        particles_entry.insert(0, "30")
        w_entry.delete(0, tk.END)
        w_entry.insert(0, "0.8")
        c1_entry.delete(0, tk.END)
        c1_entry.insert(0, "0.1")
        c2_entry.delete(0, tk.END)
        c2_entry.insert(0, "0.2")
        x_min_entry.delete(0, tk.END)
        x_min_entry.insert(0, "-100")
        x_max_entry.delete(0, tk.END)
        x_max_entry.insert(0, "100")
        y_min_entry.delete(0, tk.END)
        y_min_entry.insert(0, "-100")
        y_max_entry.delete(0, tk.END)
        y_max_entry.insert(0, "100")
    elif selected_function == "Schaffer function N.4":
        iterations_entry.delete(0, tk.END)
        iterations_entry.insert(0, "50")
        particles_entry.delete(0, tk.END)
        particles_entry.insert(0, "30")
        w_entry.delete(0, tk.END)
        w_entry.insert(0, "0.8")
        c1_entry.delete(0, tk.END)
        c1_entry.insert(0, "0.1")
        c2_entry.delete(0, tk.END)
        c2_entry.insert(0, "0.2")
        x_min_entry.delete(0, tk.END)
        x_min_entry.insert(0, "-100")
        x_max_entry.delete(0, tk.END)
        x_max_entry.insert(0, "100")
        y_min_entry.delete(0, tk.END)
        y_min_entry.insert(0, "-100")
        y_max_entry.delete(0, tk.END)
        y_max_entry.insert(0, "100")
    elif selected_function == "Styblinski-Tang":
        iterations_entry.delete(0, tk.END)
        iterations_entry.insert(0, "50")
        particles_entry.delete(0, tk.END)
        particles_entry.insert(0, "30")
        w_entry.delete(0, tk.END)
        w_entry.insert(0, "0.8")
        c1_entry.delete(0, tk.END)
        c1_entry.insert(0, "0.1")
        c2_entry.delete(0, tk.END)
        c2_entry.insert(0, "0.2")
        x_min_entry.delete(0, tk.END)
        x_min_entry.insert(0, "-5")
        x_max_entry.delete(0, tk.END)
        x_max_entry.insert(0, "5")
        y_min_entry.delete(0, tk.END)
        y_min_entry.insert(0, "-5")
        y_max_entry.delete(0, tk.END)
        y_max_entry.insert(0, "5")
    


# Tkinter Setup
root = tk.Tk()
root.title("Particle Swarm Optimization")

# Ustawienie stałej wielkości okna
root.geometry("350x300")  # Ustawienie rozmiaru okna
root.resizable(False, False)  # Zablokowanie zmiany rozmiaru

tk.Label(root, text="Iterations:").grid(row=0, column=0, sticky="e")
iterations_entry = tk.Entry(root)
iterations_entry.insert(0, "50")
iterations_entry.grid(row=0, column=1)

tk.Label(root, text="Particles:").grid(row=1, column=0, sticky="e")
particles_entry = tk.Entry(root)
particles_entry.insert(0, "30")
particles_entry.grid(row=1, column=1)

tk.Label(root, text="Inertia (w):").grid(row=2, column=0, sticky="e")
w_entry = tk.Entry(root)
w_entry.insert(0, "0.8")
w_entry.grid(row=2, column=1)

tk.Label(root, text="Cognitive (c1):").grid(row=3, column=0, sticky="e")
c1_entry = tk.Entry(root)
c1_entry.insert(0, "0.1")
c1_entry.grid(row=3, column=1)

tk.Label(root, text="Social (c2):").grid(row=4, column=0, sticky="e")
c2_entry = tk.Entry(root)
c2_entry.insert(0, "0.2")
c2_entry.grid(row=4, column=1)

tk.Label(root, text="X Min:").grid(row=5, column=0, sticky="e")
x_min_entry = tk.Entry(root)
x_min_entry.insert(0, "0")
x_min_entry.grid(row=5, column=1)

tk.Label(root, text="X Max:").grid(row=6, column=0, sticky="e")
x_max_entry = tk.Entry(root)
x_max_entry.insert(0, "5")
x_max_entry.grid(row=6, column=1)

tk.Label(root, text="Y Min:").grid(row=7, column=0, sticky="e")
y_min_entry = tk.Entry(root)
y_min_entry.insert(0, "0")
y_min_entry.grid(row=7, column=1)

tk.Label(root, text="Y Max:").grid(row=8, column=0, sticky="e")
y_max_entry = tk.Entry(root)
y_max_entry.insert(0, "5")
y_max_entry.grid(row=8, column=1)

tk.Label(root, text="Function:").grid(row=9, column=0, sticky="e")
func_selector = ttk.Combobox(root, values=list(objective_functions.keys()), state="readonly")
func_selector.grid(row=9, column=1)
func_selector.current(0)

gif_var = tk.BooleanVar()
gif_checkbox = tk.Checkbutton(root, text="Generate GIF", variable=gif_var)
gif_checkbox.grid(row=10, column=1, sticky="w")

run_button = tk.Button(root, text="Run PSO", command=run_pso)
run_button.grid(row=11, column=0, columnspan=2)

# Powiązanie Comboboxa z funkcją aktualizującą domyślne wartości
func_selector.bind("<<ComboboxSelected>>", update_defaults)

root.mainloop()
