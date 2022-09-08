import inspect
import os
import datetime
import warnings
import sys

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

    def __init__(self,env,scenario_id,method,exp_num,*args):
        # creating the path
        if(not os.path.isdir("results")):
            os.mkdir("results")
        self.path = "./results/"

        self.env = env
        self.header = args

        # defining filename
        self.start_time = datetime.datetime.now()
        self.filename = str(method)+'_'+str(env)+str(scenario_id)+'_'+str(exp_num)+'.csv'

        # creating the result file
        self.write_header()

    def write_header(self):
        with open(self.path+self.filename, 'w') as logfile:
            for header in self.header[0]:
                logfile.write(str(header)+";")
            logfile.write('\n')

    def write(self,*args):
        with open(self.path+self.filename, 'a') as logfile:
            if(not len(args) ==len(self.header)):
                warnings.warn("Initialisation and writing have different sizes .")

            for key in args[0]:
                logfile.write(str(args[0][key])+";")

            logfile.write('\n')

class BashLogFile:

    def __init__(self,file_name=""):
        # creating the path
        if(not os.path.isdir("./bashlog")):
            os.mkdir("./bashlog")
        self.path = "./bashlog/"

        # defining filename
        self.start_time = datetime.datetime.now()
        if(file_name ==""):
            self.filename = self.start_time.strftime("%d-%m-%Y_%Hh%Mm%Ss")+ ".csv"
        else:
            self.filename = file_name
        
        # saving original stderr
        self.original_stderr = sys.stderr

        # creating the log files
        ofile = open(self.path+'OUTPUT_'+self.filename,'w')
        ofile.close()

        efile = open(self.path+'ERROR_'+self.filename,'w')
        efile.close()

    def redirect_stderr(self):
        file = open(self.path+'ERROR_'+self.filename,'a')
        sys.stderr = file
    
    def reset_stderr(self):
        sys.stderr = self.original_stderr

    def write(self,log):
        print(log)
        with open(self.path+'OUTPUT_'+self.filename, 'a') as blfile:
            blfile.write(log+'\n')
