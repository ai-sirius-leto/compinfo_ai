import flet as ft
from analys import analys, read_all

INFO_FStr = '''
Время работы: `{uptime}`
# **Процессор**
Температура процессора: `{t_cpu}`\\
Использование процессора: `{u_cpu}`

# **Видеокарта**
Температура видеокарты: `{t_gpu}`\\
Использование видеокарты: `{u_gpu}`

Использоание ОЗУ: `{u_ram}`\\
Использование диска: `{u_disk}`
'''

def format(s, **kwargs):
    for k, v in kwargs.items():
        s = s.replace('{'+k+'}', str(v))
    return s

def format_data(data: tuple):
    uptime, temp_cpu, crit_temp_cpu, temp_gpu, cpu_usage, gpu_usage, ram_usage, disk_usage = [round(i, 2) for i in data]
    ut_ms = uptime % 1000
    uptime //= 1000
    ut_s = uptime % 60
    uptime //= 60
    ut_min = uptime % 60
    uptime //= 60
    ut_h = uptime % 24
    uptime //= 24
    ut_d = uptime
    
    uptime_l = [ut_d, ut_h, ut_min, ut_s]
    uptime_l2 = [str(i) for i in uptime_l if i > 0]
    uptime_str = ':'.join(uptime_l2)
    
    return format(
        INFO_FStr, uptime=uptime_str,
        t_cpu=f'{temp_cpu}°C', t_gpu=f'{temp_gpu}°C' if temp_gpu != -1 else '❌',
        u_cpu=f'{cpu_usage}%', u_gpu=f'{gpu_usage}%' if gpu_usage != -1 else '❌',
        u_ram=f'{ram_usage}%', u_disk=f'{disk_usage}%'
    )

def page_now(page: ft.Page):
    page.window.height = 350
    page.window.min_height = 445
    page.window.width = 300
    page.window.min_width = 370
    
    if read_all():
        txt = ft.Markdown(format_data(read_all()[-1]))
    else:
        txt = ft.Markdown('Анализ')
    
    def update_txt():
        if page.navigation_bar.selected_index == 0:
            try:
                analys()
                
                txt.value = format_data(read_all()[-1])
                
                txt.update()
                update_txt()
            except AssertionError:
                return
        else:
            return
    
    page.add(txt)
    
    update_txt()
