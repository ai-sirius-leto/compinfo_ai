from sqlite3 import connect
import flet as ft
from matplotlib.figure import Figure

from analys import analys, read_last
from utils import get_uptime_str, smooth_resize

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
        mut = ut - 60000
        
        with connect('data.db') as conn:
            cur = conn.cursor()
            cur.execute('select * from compinfo where uptime > ?', (sut,))
            r = cur.fetchall()
            cur.close()
        
        points = np.array(sorted([[i[0], i[1]] for i in r], key=lambda x: x[0]))

        fig, axs = plt.subplots(1, 1)
        
        axs.plot(points[:, 0], points[:, 1])
        axs.set_xlim(sut, ut)
        axs.set_xlabel("время с момента включения компьютера")
        axs.grid(True, color="white", linestyle="--", linewidth=0.5)
        axs.tick_params(colors="white", which="both")
        axs.xaxis.label.set_color("white")
        axs.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: get_uptime_str(int(x))))
        axs.yaxis.label.set_color("white")
        axs.spines["bottom"].set_color("white")
        axs.spines["left"].set_color("white")

        fig.tight_layout()
        return fig

    chart = MatplotlibChart(__get_plot(), isolated=True, transparent=True, expand=True)
    
    def update_chart():
        while page.navigation_bar.selected_index == 1:
            chart.figure = __get_plot()
            chart.update()

    page.add(chart)
    update_chart()
