import flet as ft
from utils import get_settings, set_settings, s, smooth_resize, translate

def page_settings(page: ft.Page):
    smooth_resize(page, 350, 300)
    curr_lang = get_settings()['lang']
    flag_ru = ft.Image('/flag_ru.png', height=40, opacity=1 if curr_lang == 'ru' else 0.5)
    flag_uk = ft.Image('/flag_uk.png', height=40, opacity=1 if curr_lang == 'en' else 0.5)
    row = ft.Row([flag_ru, flag_uk], width=90)
    
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
        
        row.update()
    
    title = ft.Container(ft.Text(data='[settings.title]', size=30, weight=ft.FontWeight.BOLD), width=page.window.width, alignment=ft.Alignment(0, 0), padding=ft.Padding(0, 0, 0, 20))
    lang_setting = ft.Row([ft.Text(data='[settings.lang]', size=20, height=30), ft.FilledTonalButton(content=row, on_click=change_lang)], alignment=ft.MainAxisAlignment.SPACE_AROUND)
    
    s.translations += [
        [title.content, 'value'],
        [lang_setting.controls[0], 'value']
    ]
    
    page.add(
        title,
        lang_setting
    )
