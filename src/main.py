import flet as ft

from pages.chart import page_chart
from pages.now import page_now
from pages.settings import page_settings
from utils import get_settings, s


def main(page: ft.Page):
    page.data = '[wintitle]'
    page.on_disconnect = lambda *a, **k: exit(0)
    page.window.height = 10
    page.window.width = 10
    page.on_close = lambda *_: exit(0)

    def on_keyboard(e: ft.KeyboardEvent):
        # Change navigation bar destinations using 'Arrow Left' and 'Arrow Right' keys
        dests = list(range(len(page.navigation_bar.destinations))) * 3
        curr_dest_index = 3 + page.navigation_bar.selected_index
        match e.key:
            case 'Arrow Left':
                page.navigation_bar.selected_index = dests[curr_dest_index - 1]
            case 'Arrow Right':
                page.navigation_bar.selected_index = dests[curr_dest_index + 1]
        change_destination()

    page.on_keyboard_event = on_keyboard
    
    sets = get_settings()
    match sets['theme']:
        case 'light':
            page.theme_mode = ft.ThemeMode.LIGHT
        case 'system':
            page.theme_mode = ft.ThemeMode.SYSTEM
        case 'dark':
            page.theme_mode = ft.ThemeMode.DARK
    
    def change_destination(*_):
        page.controls.clear()
        match page.navigation_bar.selected_index:
            case 0:
                page_now(page)
            case 1:
                page_chart(page)
            case 2:
                page_settings(page)
    
    page.navigation_bar = ft.NavigationBar(
        destinations=[
            ft.NavigationBarDestination(
                icon=ft.Icons.TIMER_OUTLINED,
                selected_icon=ft.Icons.TIMER,
                data='[navbar.now]'
            ),
            ft.NavigationBarDestination(
                icon=ft.Icons.TIMELINE_OUTLINED,
                selected_icon=ft.Icons.TIMELINE,
                data='[navbar.chart]'
            ),
            ft.NavigationBarDestination(
                icon=ft.Icons.SETTINGS_OUTLINED,
                selected_icon=ft.Icons.SETTINGS,
                data='[navbar.settings]'
            ),
        ],
        on_change=change_destination
    )
    s.translations += [
        [page, 'title'],
        [page.navigation_bar.destinations[0], 'label'],
        [page.navigation_bar.destinations[1], 'label'],
        [page.navigation_bar.destinations[2], 'label']
    ]
    page.update()
    
    page_now(page)

if __name__ == '__main__':
    ft.app(main, use_color_emoji=True, assets_dir="pages/assets")

