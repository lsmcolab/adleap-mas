import inspect
import os
import psutil
import datetime
import warnings

######
# EXCEPTION METHODS
######
# returns the current line number in our program.
def lineno():
    return inspect.currentframe().f_back.f_lineno

######
# WRITE METHODS
######
class LogFile:

    def __init__(self,env=None,file_name="",*args):
        # self.filename = './results/'+ str(self.n_agents) + '_' +\
        #     str(self.n_tasks) + '_' + self.start_time.strftime("%d-%m-%Y_%Hh%Mm%Ss") + '.csv'
        self.env = env
        self.start_time = datetime.datetime.now()
        self.header = args
        if(file_name ==""):
            self.filename = "./results/"+self.start_time.strftime("%d-%m-%Y_%Hh%Mm%Ss")+ ".csv"
        else:
            self.filename = file_name
        if(not os.path.isdir("results")):
            os.mkdir("results")
        self.write_header()

    def write_header(self):
        with open(self.filename, 'w') as logfile:
            #logfile.write('Iteration;')
            #logfile.write('CPU Usage (%);')
            #logfile.write('Memory Usage (RAM) (%);')
            #logfile.write('Additional Info;')
            for header in self.header[0]:
                logfile.write(str(header)+";")
            logfile.write('\n')

    def write(self,env=None,*args):
        with open(self.filename, 'a') as logfile:
            if(not len(args) ==len(self.header)):
                warnings.warn("Initialisation and writing have different sizes .")

            for key in args[0]:
                logfile.write(str(args[0][key])+";")

            logfile.write('\n')

class BashLogFile:

    def __init__(self,file_name=""):
        if(not os.path.isdir("./bashlog")):
            os.mkdir("./bashlog")

        self.start_time = datetime.datetime.now()
        if(file_name ==""):
            self.filename = "./bashlog/"+self.start_time.strftime("%d-%m-%Y_%Hh%Mm%Ss")+ ".csv"
        else:
            self.filename = file_name
            
        file = open(self.filename,'w')
        file.close()

    def write(self,log):
        print(log)
        with open(self.filename, 'a') as blfile:
            blfile.write(log+'\n')
