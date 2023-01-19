"""python wrapper"""

# decorator is a high-order function that returns a function
def log_wrapper(func):
    def wrapped_func(*args, **kargs):
        print('I got wrapped!')
        return func(*args, **kargs)
    return wrapped_func

def say_hello(x):
    print('hello')
    return x + 1

if __name__ == '__main__':
    f = log_wrapper(say_hello)
    print(f.__name__, say_hello.__name__)
    y = f(1)
    print(y)
    
