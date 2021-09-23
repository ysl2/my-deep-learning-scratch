#!/usr/bin/env python3

# * 在形参，表示封包
# * 在调用处，表示解包
# 元组需要一个*进行封/解包
# 字典需要两个*进行封/解包

def accept1(*s):
    print(s)


def accept2(**s):
    print(s)


if __name__ == '__main__':
    a = (0, 1, 2, 3)
    b = {'a': 0, 'b': 1, 'c': 2}

    accept1(a)
    accept1(*a)
    # accept1(**a)
    accept1(b)
    accept1(*b)
    # accept1(**b)

    # =========================

    # accept2(a)
    # accept2(*a)
    # accept2(**a)
    # accept2(b)
    # accept2(*b)
    accept2(**b)
