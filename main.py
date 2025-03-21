import flet as ft
from analys import analys

# for i in range(1, 11):
#     analys()
#     print(f"{i}/10")

INFO_FStr = 'Время работы: {uptime}\nТемпература процессора: {t_cpu}\nТемпература видеокарты: {t_gpu}\nЗагруженность процессора: {u_cpu}\nЗагруженность видеокарты: {u_gpu}\nЗагруженность диска:{u_disk}'


def main(page: ft.Page):
    page.title = 'Анализатор компьютера'
    page.add(ft.Text('))

if __name__ == '__main__':
    ft.app(main)

