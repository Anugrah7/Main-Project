import os
import time
import sys
import pyfiglet

def run_PE():
    file = sys.argv[2]
    os.system("python Extract/PE_main.py {}".format(file))

def run_URL():
    url=sys.argv[2]
    os.system("python Extract/url_main.py {}".format(url))

def start():
    select = sys.argv[1]

    if select == '1':
        run_PE()
    elif select == '2': 
        run_URL()
    elif select == '3':
        exit()
    else:
        print("Bad input\nExiting...")
        time.sleep(3)
        exit()



start()
