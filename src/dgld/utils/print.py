def cprint(x, color='green'):
    from termcolor import colored
    if color == 'info':
        color = 'green'
        x = 'INFO:' + str(x)
    if color == 'debug':
        color = 'red'
        x = 'DEBUG:' + str(x)
        
    print(colored(x, color))

def lcprint(*arr, color='green'):
    x = ':: '.join([str(i) for i in arr])
    from termcolor import colored
    if color.lower() == 'info':
        color = 'green'
        x = 'INFO:' + str(x)
    if color.lower() == 'debug':
        color = 'red'
        x = 'DEBUG:' + str(x)
        
    print(colored(x, color))