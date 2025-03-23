import flet as ft
from utils import get_settings, set_settings

def page_settings(page: ft.Page):
    curr_lang = get_settings()['lang']
    flag_ru = ft.Image('/flag_ru.png', height=40, opacity=1 if curr_lang == 'ru' else 0.5)
    flag_uk = ft.Image('/flag_uk.png', height=40, opacity=1 if curr_lang == 'uk' else 0.5)
    row = ft.Row([flag_ru, flag_uk], width=90)
    
    def change_lang(*_):
        sets = get_settings()
        curr_lang = sets['lang']
        
        # ru -> uk
        match curr_lang:
            case 'ru':
                sets['lang'] = 'uk'
                flag_uk.opacity = 1
                flag_ru.opacity = 0.5
            case 'uk':
                sets['lang'] = 'ru'
                flag_ru.opacity = 1
                flag_uk.opacity = 0.5
        set_settings(sets)
        row.update()
    
    page.add(
        ft.Container(ft.Text('Настройки', size=30, weight=ft.FontWeight.BOLD), width=page.window.width, alignment=ft.Alignment(0, 0), padding=ft.Padding(0, 0, 0, 20)),
        ft.Row([ft.Text('Язык', size=20, height=30), ft.FilledTonalButton(content=row, on_click=change_lang)], alignment=ft.MainAxisAlignment.SPACE_AROUND)
    )
