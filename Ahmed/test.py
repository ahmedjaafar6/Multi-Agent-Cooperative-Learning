# x = 5



def foo():
    global reward
    reward = 0
    def ss():
        global reward
        reward+=1
    ss()
    print(reward)

foo()
# print(x)


# x , y = 3, 5

# print(x, y)