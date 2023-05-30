python -m venv venv

venv\Scripts\pip.exe install discord
venv\Scripts\pip.exe install -r requirements.txt

::CLI arguments can be added below as 'bot.py --xformers --extensions superbooga'

venv\Scripts\python.exe bot.py 
