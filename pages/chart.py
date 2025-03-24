from sqlite3 import connect
import flet as ft
from matplotlib.figure import Figure
from matplotlib.pyplot import close as plt_close

from analysis import analys, read_last
from utils import get_uptime_str, smooth_resize, translate

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import flet as ft
from flet.matplotlib_chart import MatplotlibChart

matplotlib.use("svg")

def page_chart(page: ft.Page):
    smooth_resize(page, 500, 700)
    
    
    def __get_plot() -> Figure:
        analys()
        ut = read_last()[0]
        sut = ut - 120000
        
        with connect('data.db') as conn:
            cur = conn.cursor()
            cur.execute('select * from compinfo where uptime > ?', (sut,))
            r = cur.fetchall()
            cur.close()
        
        points = np.array(sorted(r, key=lambda x: x[0]))

        fig, axs = plt.subplots(1, 1)
        
        axs.plot(points[:, 0], points[:, 1], label=translate('[data.temp_cpu] (°C)'), color='yellow')
        axs.plot(points[:, 0], points[:, 2], color='red')
        if points[0, 3] != -1:
            axs.plot(points[:, 0], points[:, 3], label=translate('[data.temp_gpu] (°C)'), color='green')
        axs.plot(points[:, 0], points[:, 4], label=translate('[data.cpu_usage] (%)'), color='orange')
        if points[0, 5] != -1:
            axs.plot(points[:, 0], points[:, 5], label=translate('[data.gpu_usage] (%)'), color='blue')
        axs.plot(points[:, 0], points[:, 6], label=translate('[data.ram_usage] (%)'), color='purple')
        axs.legend(loc='upper left')
        axs.set_xlim(sut, ut)
        axs.set_xlabel(translate("[chart.xaxis_title]"))
        axs.grid(True, color="white", linestyle="--", linewidth=0.5)
        axs.tick_params(colors="white", which="both")
        axs.xaxis.label.set_color("white")
        axs.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: get_uptime_str(int(x))))
        axs.yaxis.label.set_color("white")
        axs.spines["bottom"].set_color("white")
        axs.spines["left"].set_color("white")

        fig.tight_layout()
        return fig

    plt.rcParams['legend.labelcolor'] = 'white'
    plt.rcParams['legend.facecolor'] = 'black'
    plt.rcParams['legend.edgecolor'] = 'black'
    chart = MatplotlibChart(__get_plot(), isolated=True, transparent=True, expand=True)
    
    def update_chart():
        while page.navigation_bar.selected_index == 1:
            try:
                fig = chart.figure
                chart.figure = __get_plot()
                chart.update()
                plt_close(fig)
            except AssertionError:
                return

    page.add(chart)
    update_chart()
