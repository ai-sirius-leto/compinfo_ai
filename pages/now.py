import flet as ft
from analys import analys, read_all, read_last
from utils import get_uptime_str, smooth_resize

INFO_FStr = '''
Время работы: `{uptime}`
# **Процессор**
Температура процессора: `{t_cpu}`\\
Использование процессора: `{u_cpu}`

# **Видеокарта**
Температура видеокарты: `{t_gpu}`\\
Использование видеокарты: `{u_gpu}`

Использоание ОЗУ: `{u_ram}`
'''

def format(s, **kwargs):
    for k, v in kwargs.items():
        s = s.replace('{'+k+'}', str(v))
    return s

def format_data(data: tuple):
    uptime, temp_cpu, crit_temp_cpu, temp_gpu, cpu_usage, gpu_usage, ram_usage = [round(i, 2) for i in data]
    uptime_str = get_uptime_str(uptime)
        
    return format(
        INFO_FStr, uptime=uptime_str,
        t_cpu=f'{temp_cpu}°C', t_gpu=f'{temp_gpu}°C' if temp_gpu != -1 else '❌',
        u_cpu=f'{cpu_usage}%', u_gpu=f'{gpu_usage}%' if gpu_usage != -1 else '❌',
        u_ram=f'{ram_usage}%'
    )

def page_now(page: ft.Page):
    smooth_resize(page, 350, 300)
    if read_all():
        txt = ft.Markdown(format_data(read_last()))
    else:
        txt = ft.Markdown('Анализ')
    
    def update_txt():
        while page.navigation_bar.selected_index == 0:
            try:
                analys()
                
                txt.value = format_data(read_last())
                
                txt.update()
                update_txt()
            except AssertionError:
                return
    
    page.add(txt)
    
    update_txt()
