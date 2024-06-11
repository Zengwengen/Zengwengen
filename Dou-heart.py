from turtle import *
import turtle


def go_to(x, y):
    up()
    goto(x, y)
    down()


def big_circle(size):
    speed(1600)
    for i in range(150):
        forward(size)
        right(0.3)
    turtle.hideturtle()


def small_circle(size):
    speed(1600)
    for i in range(210):
        forward(size)
        right(0.786)


def line(size):
    speed(1100)
    forward(51*size)


def heart(x, y, size):
    go_to(x, y)
    left(150)
    begin_fill()
    line(size)
    big_circle(size)
    small_circle(size)
    left(120)
    small_circle(size)
    big_circle(size)
    line(size)
    end_fill()


def arrow():
    pensize(10)
    setheading(0)
    go_to(-400, 0)
    left(15)
    forward(150)
    go_to(339, 178)
    forward(150)


def arrow_head():
    pensize(1)
    #speed(1200)
    color('black', 'red')
    begin_fill()
    left(120)
    forward(20)
    right(150)
    forward(35)
    right(120)
    forward(35)
    right(150)
    forward(20)
    end_fill()


def main():
    pensize(4)
    color('black', 'red')
    # getscreen().tracer(30, 0)
    heart(200, 0, 1)
    setheading(0)
    heart(-80, -100, 1.5)
    arrow()
    arrow_head()
    go_to(400, -300)
    done()


main()
