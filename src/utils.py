from ujson import dumps, loads
from typing import Any
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
    uptime = int(uptime)
    ut_ms = uptime % 1000
    uptime //= 1000
    ut_s = uptime % 60
    uptime //= 60
    ut_min = uptime % 60
    uptime //= 60
    ut_h = uptime % 24
    uptime //= 24
    ut_d = uptime
    
    def __format_num(x):
        x = str(x)
        return "0" * (2 - len(x)) + x

    uptime_l = map(__format_num, [ut_d, ut_h, ut_min, ut_s])
    uptime_l2 = []

    for i in uptime_l:
        if i != "00" or len(uptime_l2) > 0:
            uptime_l2.append(i)

    return ':'.join(uptime_l2)

settings_path = 'pages/settings.json'

def get_settings() -> dict[str, Any]:
    with open(settings_path, encoding='utf8') as f:
        d = f.read()
    return loads(d)

def set_settings(new_data: dict[str, Any]) -> None:
    with open(settings_path, "w", encoding='utf8') as f:
        f.write(dumps(new_data, indent=4))
        
loc_files = {
    'en': 'pages/localisation/en.json',
    'ru': 'pages/localisation/ru.json'
}

def __get_onedim_dict(d: dict[str, dict[str, Any] | Any]) -> dict[str, Any]:
    '''
    {
        "a": {"b": 3, "c": 9},
        "b": 18
    }
    ->
    {
        "a.b": 3,
        "a.c": 9,
        "b": 18
    }
    '''
    
    changed = False
    
    nd = {}
    
    for k, v in d.items():
        if isinstance(v, dict):
            changed = True
            for k2, v2 in v.items():
                nd[f"{k}.{k2}"] = v2
        else:
            nd[k] = v
    
    if changed:
        return __get_onedim_dict(nd)
    return nd

# auto set translated text
class _translations(list):
    def __init__(self, iterable = list()):
        super().__init__(iterable)
    
    def append(self, object):
        p, n = object
        if p.data:
            p.__setattr__(n, translate(p.data))
        return super().append(object)
    
    def __iadd__(self, value):
        for i in value:
            self.append(i)
        return self

class State:
    translations = _translations()

s = State()

def translate(s: str) -> None:
    curr_lang = get_settings()['lang']
    
    with open(loc_files[curr_lang], encoding='utf8') as f:
        loc = __get_onedim_dict(loads(f.read()))
    with open(loc_files[list(loc_files.keys())[0]], encoding='utf8') as f:
        def_loc = __get_onedim_dict(loads(f.read()))
    
    for k, v in loc.items():
        s = s.replace('['+k+']', v)
    for k, v in def_loc.items():
        s = s.replace('['+k+']', v)
    
    return s
    
