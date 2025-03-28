from os import read
from sqlite3 import connect
import flet as ft
from matplotlib.figure import Figure
from matplotlib.pyplot import close as plt_close
import psutil
import time

from analysis import analysis, read_all, read_last
from arbeitAI import all_predict
from utils import get_settings, get_uptime_str, smooth_resize, translate

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import flet as ft
from flet.matplotlib_chart import MatplotlibChart

matplotlib.use("svg")

def page_chart(page: ft.Page):
    smooth_resize(page, 500, 700)
    
    
    def __get_plot() -> Figure:
        analysis()
        ut = read_last()[0]
        
        p = psutil.Process()        
        sut = max((psutil.boot_time() - p.create_time()) * 1000, ut - 60000)
 
        sets = get_settings()
        predict_ms = 1000 * sets['pr_sc']

        r = [i for i in read_all() if i[0] > sut]
        
        points = np.array(sorted(r, key=lambda x: x[0]))
        predicted_points = np.array([])

        fig, axs = plt.subplots()
        
        # Predict `predict_ms` milliseconds
        predicted_points = all_predict(False, predict_ms // 1000)

        def get_x_y(x_arr_column, y_arr_column, start, end):
            m = (x_arr_column >= start) & (x_arr_column <= end)
            return x_arr_column[m], y_arr_column[m]

        # Temp cpu (real / predicted)
        axs.plot(*get_x_y(points[:, 0], points[:, 1], sut, ut), label=translate('[data.temp_cpu] (°C)'), color='yellow')
        axs.plot(*get_x_y(predicted_points[:, 0], predicted_points[:, 1], ut - 1000, ut + predict_ms), color=(1, 1, 0.458823529))

        axs.plot(points[:, 0], points[:, 2], color='red') # Critical temp cpu
        
        # Temp gpu (if exist)
        if points[0, 3] != -1:
            axs.plot(points[:, 0], points[:, 3], label=translate('[data.temp_gpu] (°C)'), color='green')
        
        # Cpu usage (real / predicted)
        axs.plot(*get_x_y(points[:, 0], points[:, 4], sut, ut), label=translate('[data.cpu_usage] (%)'), color='orange')
        axs.plot(*get_x_y(predicted_points[:, 0], predicted_points[:, 4], ut - 1000, ut + predict_ms), color=(1, 0.811764706, 0.509803922))
        
        
        # Gpu usage (if exist)
        if points[0, 5] != -1:
            axs.plot(points[:, 0], points[:, 5], label=translate('[data.gpu_usage] (%)'), color='blue')
        
        # Ram usage (real / predicted)
        axs.plot(*get_x_y(points[:, 0], points[:, 6], sut, ut), label=translate('[data.ram_usage] (%)'), color=(1, 0, 1))
        axs.plot(*get_x_y(predicted_points[:, 0], predicted_points[:, 6], ut - 1000, ut + predict_ms), color=(1, 0.490196078, 1))
        
        # Line which's dividing real an predicted data
        axs.plot([ut] * 100, np.arange(100), color='grey')
        
        # Settings
        axs.legend(loc='upper left')
        axs.set_xlim(sut, ut + predict_ms)
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
