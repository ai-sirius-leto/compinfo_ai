import flet as ft

def smooth_resize(page: ft.Page, nh: int, nw: int):
    smoothness = 1000
    
    sh = page.window.height
    sw = page.window.width
    
    dh = nh - sh
    dw = nw - sw
    
    for i in range(smoothness):
        page.window.height = sh + (dh * (i + 1)) // smoothness
        page.window.width = sw + (dw * (i + 1)) // smoothness
        page.update()
    
    page.window.height = nh
    page.window.width = nw
    page.window.min_height = nh + 100
    page.window.min_width = nw + 55
    page.update()

def get_uptime_str(uptime: int) -> str:
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
    return ':'.join(uptime_l2)
