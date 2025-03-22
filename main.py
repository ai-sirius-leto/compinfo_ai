import flet as ft

from pages.chart import page_chart
from pages.now import page_now
from pages.settings import page_settings


def main(page: ft.Page):
    page.title = 'Анализатор компьютера'
    page.on_disconnect = lambda *a, **k: exit(0)
    page.window.height = 10
    page.window.width = 10
    
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
                label="Сейчас"
            ),
            ft.NavigationBarDestination(
                icon=ft.Icons.TIMELINE_OUTLINED,
                selected_icon=ft.Icons.TIMELINE,
                label="График"
            ),
            ft.NavigationBarDestination(
                icon=ft.Icons.SETTINGS_OUTLINED,
                selected_icon=ft.Icons.SETTINGS,
                label="Настройки",
            ),
        ],
        on_change=change_destination
    )
    page.update()
    
    page_now(page)

if __name__ == '__main__':
    ft.app(main, use_color_emoji=True)

