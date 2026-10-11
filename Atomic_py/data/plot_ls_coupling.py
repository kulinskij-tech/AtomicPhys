from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse


def arrow(axis, start, end, color, label, offset, dashed=False):
    axis.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "-|>", "color": color, "lw": 2.4, "linestyle": "--" if dashed else "-", "mutation_scale": 17})
    axis.text(end[0] + offset[0], end[1] + offset[1], label, color=color, fontsize=14)


def build_figures(destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "svg.fonttype": "none"})
    orbital_squared = 2.0
    spin_squared = 0.75
    total_squared = 3.75
    total_length = np.sqrt(total_squared)
    orbital_parallel = (total_squared + orbital_squared - spin_squared) / (2 * total_length)
    spin_parallel = total_length - orbital_parallel
    transverse = np.sqrt(orbital_squared - orbital_parallel**2)
    orbital = np.array([transverse, orbital_parallel])
    spin = np.array([-transverse, spin_parallel])
    total = orbital + spin
    moment_orbital = -orbital
    moment_spin = -2 * spin
    moment = moment_orbital + moment_spin
    mean_moment = np.array([0, moment[1]])
    assert np.isclose(spin @ spin, spin_squared)
    assert np.allclose(total, [0, total_length])
    assert np.isclose(-mean_moment[1] / total_length, 4 / 3)

    figure, axes = plt.subplots(1, 2, figsize=(11.5, 5.2), constrained_layout=True)
    for axis in axes:
        axis.set_aspect("equal")
        axis.axis("off")
        axis.plot(0, 0, "ko", markersize=4)
    first, second = axes
    first.set_title("Внутрішня прецесія навколо J", pad=16)
    arrow(first, [0, 0], total, "#7030a0", r"$\mathbf{J}$", [0.10, 0.05])
    arrow(first, [0, 0], orbital, "#0072b2", r"$\mathbf{L}$", [0.09, 0.03])
    arrow(first, [0, 0], spin, "#d55e00", r"$\mathbf{S}$", [-0.24, 0.02])
    arrow(first, orbital, total, "#d55e00", r"$\mathbf{S}$", [0.10, -0.32], dashed=True)
    for height, color in [(orbital_parallel, "#0072b2"), (spin_parallel, "#d55e00")]:
        first.add_patch(Ellipse((0, height), 2 * transverse, 0.22, fill=False, edgecolor=color, linestyle="--", alpha=0.55))
    first.text(0, -0.24, r"$\mathbf{L}_{\perp}+\mathbf{S}_{\perp}=0$", ha="center", fontsize=13)
    first.text(0, -0.52, r"$L=1,\ S=1/2,\ J=3/2$", ha="center")
    first.text(0, -0.77, "Моменти у одиницях ħ", ha="center", color="#555555")
    first.set_xlim(-1.15, 1.15)
    first.set_ylim(-0.95, 2.30)

    second.set_title("Магнітний момент і його середнє", pad=16)
    second.plot([0, 0], [0.38, -2.95], color="#7030a0", linewidth=1, alpha=0.35)
    second.text(0.06, 0.20, "вісь J", color="#7030a0")
    arrow(second, [0, 0], moment_orbital, "#0072b2", r"$\boldsymbol{\mu}_L$", [-0.50, -0.09])
    arrow(second, [0, 0], moment_spin, "#d55e00", r"$\boldsymbol{\mu}_S$", [0.06, 0.06])
    arrow(second, [0, 0], moment, "#009e73", r"$\boldsymbol{\mu}$", [0.07, 0.04])
    arrow(second, [0, 0], mean_moment, "#a91d45", r"$\overline{\boldsymbol{\mu}}$", [-0.60, -0.07], dashed=True)
    second.plot([mean_moment[0], moment[0]], [mean_moment[1], moment[1]], "--", color="#777777", linewidth=1)
    second.text(0.20, -3.15, r"$g_L=1,\quad g_S=2,\quad g_J=4/3$", ha="center")
    second.text(0.20, -3.45, r"Магнітні моменти у одиницях $\mu_B$", ha="center", color="#555555")
    second.set_xlim(-1.45, 1.55)
    second.set_ylim(-3.65, 0.65)
    figure.savefig(destination / "fig_lande_vector_model.png", dpi=180, facecolor="white")
    plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(12.5, 4.4), constrained_layout=True)
    titles = ["1. Конфігурація", "2. Електростатичні терми", "3. LS-рівні без зовнішнього поля"]
    for axis, title in zip(axes, titles):
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 3.5)
        axis.axis("off")
        axis.set_title(title, fontsize=12, pad=12)

    def level(axis, energy, label, color):
        axis.plot([0.10, 0.62], [energy, energy], color=color, linewidth=3)
        axis.text(0.10, energy + 0.13, label, fontsize=14, color=color)

    level(axes[0], 1.55, r"$1s2p$: 12 станів", "#555555")
    axes[0].text(0.10, 0.72, "Обидва електрони\nберуть участь у класифікації", fontsize=11)
    level(axes[1], 2.70, r"$^{1}P$: 3 стани", "#0072b2")
    level(axes[1], 1.15, r"$^{3}P$: 9 станів", "#d55e00")
    axes[1].text(0.10, 0.30, "Синглет–триплетне розділення:\nкулонівська енергія та обмін", fontsize=11)
    level(axes[2], 2.70, r"$^{1}P_1$: $2J+1=3$", "#0072b2")
    for energy, total_number in [(1.65, 0), (1.10, 1), (0.55, 2)]:
        level(axes[2], energy, rf"$^{{3}}P_{total_number}$: $2J+1={2 * total_number + 1}$", "#d55e00")
    figure.text(0.5, 0.025, "Схема класифікації: порядок і відстані між рівнями не відтворюють реальні енергії He.", ha="center", fontsize=10, color="#555555")
    figure.savefig(destination / "fig_he_ls_hierarchy.png", dpi=180, facecolor="white")
    plt.close(figure)


if __name__ == "__main__":
    build_figures(Path(__file__).resolve().parents[1] / "figs")
