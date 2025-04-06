# How to create the project as an executable. i.e create a file- opening it will directly open the chat :D
#### Assumes conda env.
#### create a file with .bat  extension and put this code inside it with necessary modifications. 


```
rem @echo OFF

#set conda path
set CONDAPATH=C:\Users\user\anaconda3   

#input env_name for your project
set ENVNAME=llm_pdf                     

if %ENVNAME%==base (set ENVPATH=%CONDAPATH%) else (set ENVPATH=%CONDAPATH%\envs\%ENVNAME%)

call %CONDAPATH%\Scripts\activate.bat %ENVPATH%

#modify accordingly!!! 
streamlit run path_to_this_folder\project\Home.py

```
