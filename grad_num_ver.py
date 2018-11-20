import sympy as sy
import math


def myformula(formula, **kwargs):
    expr = sy.sympify(formula)
    return expr.evalf(subs=kwargs)


def ver_grad(val, e):
    formula1 = "x^2+4.5*x^3-sin(x)"
    value = (myformula(formula1, x=val+e)-myformula(formula1, x=val-e))/(2*e)
    return value


def main():

    value = ver_grad(0.7, 0.01)
    print(value)


if __name__ == "__main__":
    main()