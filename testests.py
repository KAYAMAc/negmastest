class A:

    def __next__(self) :
        print("aaa")
        return

    def __iter__(self):
        return self

    def run(self):
        for _ in self:
            print("bbb")
a=A()
a.run()