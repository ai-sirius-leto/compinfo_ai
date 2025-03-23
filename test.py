class A:
    def __init__(self):
        self.__lol = 'lol'
    
    @property
    def lol(self) -> str:
        return self.__lol

    @lol.setter
    def lol(self, new_lol: str):
        self.__lol = new_lol
        
a = A()
l = {a: ['lol']}

for k, v in l.items():
    for i in v:
        k.__setattr__(i, '12')

print(a.lol)    
