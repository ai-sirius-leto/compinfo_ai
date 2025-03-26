from os import close
import flet as ft
from utils import get_settings, set_settings, s, smooth_resize, translate
from pynput import keyboard
from pynput.keyboard import Key, KeyCode

class State:
    shift = False

s2 = State()

def page_settings(page: ft.Page):
    page.update()
    
    smooth_resize(page, 350, 300)
    sets = get_settings()
    curr_lang = sets['lang']
    curr_theme = sets['theme']
    
    flag_ru = ft.Image('/flag_ru.png', height=40, opacity=1 if curr_lang == 'ru' else 0.5)
    flag_uk = ft.Image('/flag_uk.png', height=40, opacity=1 if curr_lang == 'en' else 0.5)
    lang_row = ft.Row([flag_ru, flag_uk], width=90, height=45)
    
    light_theme = ft.Image('/light_theme.png', height=40, opacity=1 if curr_theme == 'light' else 0.5)
    system_theme = ft.Image('/system_theme.png', height=40, opacity=1 if curr_theme == 'system' else 0.5)
    dark_theme = ft.Image('/dark_theme.png', height=40, opacity=1 if curr_theme == 'dark' else 0.5)
    theme_row = ft.Row([light_theme, system_theme, dark_theme], width=145, height=50)
    
    
    def change_lang(*_):
        sets = get_settings()
        curr_lang = sets['lang']
        
        # ru -> uk
        match curr_lang:
            case 'ru':
                sets['lang'] = 'en'
                flag_uk.opacity = 1
                flag_ru.opacity = 0.5
            case 'en':
                sets['lang'] = 'ru'
                flag_ru.opacity = 1
                flag_uk.opacity = 0.5
        
        set_settings(sets)
        
        for p, n in s.translations:
            if p.data:
                p.__setattr__(n, translate(p.data))
            if p.page:
                p.update()
        
        lang_row.update()
    
    def change_theme(*_):
        print(s2.shift)
        l = ['light', 'system', 'dark'] * 3
        sets = get_settings()
        curr_theme = sets['theme']
        
        d = -1 if s2.shift else 1
        new_theme = l[l.index(curr_theme, 3) + d]

        light_theme.opacity = 1 if new_theme == 'light' else 0.5
        system_theme.opacity = 1 if new_theme == 'system' else 0.5
        dark_theme.opacity = 1 if new_theme == 'dark' else 0.5
        theme_row.update()
        
        sets['theme'] = new_theme
        set_settings(sets)
        
        # Change theme
        match sets['theme']:
            case 'light':
                page.theme_mode = ft.ThemeMode.LIGHT
            case 'system':
                page.theme_mode = ft.ThemeMode.SYSTEM
            case 'dark':
                page.theme_mode = ft.ThemeMode.DARK
        page.update()
        
        
    title = ft.Container(ft.Text(data='[settings.title]', size=30, weight=ft.FontWeight.BOLD), width=page.window.width, alignment=ft.Alignment(0, 0), padding=ft.Padding(0, 0, 0, 20))
    lang_setting = ft.Row([ft.Text(data='[settings.lang]', size=20, height=30), ft.FilledTonalButton(content=lang_row, on_click=change_lang)], alignment=ft.MainAxisAlignment.SPACE_AROUND)
    theme_setting = ft.Row([ft.Text(data='[settings.theme]', size=20, height=30), ft.FilledTonalButton(content=theme_row, on_click=change_theme)], alignment=ft.MainAxisAlignment.SPACE_AROUND)
    theme_tip = ft.Text(data='[settings.theme_tip]', opacity=0.5, size=12)
    s.translations += [
        [title.content, 'value'],
        [lang_setting.controls[0], 'value'],
        [theme_setting.controls[0], 'value'],
        [theme_tip, 'value']
    ]
    
    page.add(
        title,
        lang_setting,
        theme_setting,
        theme_tip
    )
    
    def on_press(key: Key | KeyCode):
        if key == Key.shift:
            s2.shift = True
    def on_release(key):
        if key == keyboard.Key.shift:
            s2.shift = False
            
    with keyboard.Listener(
            on_press=on_press,
            on_release=on_release) as listener:
        listener.join()

