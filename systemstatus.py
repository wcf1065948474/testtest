import numpy as np
import pynvml
import psutil
import time
import threading

class GetSystemStatus(object):
    def __init__(self):
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        self.cpu_util_list = []
        self.mem_util_list = []
        self.mem_used_list = []
        self.gpu_util_list = []
        self.gpu_mem_util_list = []
        self.gpu_mem_used_list = []
        self.thread = threading.Thread(target=self.run,name='sys_status')
        self.thread.setDaemon(True)
        self.thread.start()
        
    
    def get_gpu_util_status(self):
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
            gpu_util = util.gpu
            mem_util = util.memory

            memInfo = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            mem_total = memInfo.total / 1024 / 1024
            mem_used = memInfo.used / 1024 / 1024
        except pynvml.NVMLError as err:
            return None
        return gpu_util,mem_util,mem_total,mem_used

    def get_cpu_util_status(self):
        cpu_util = psutil.cpu_percent(interval=None)
        return cpu_util

    def get_mem_util_status(self):
        mem = psutil.virtual_memory()
        return mem
    
    def record_status(self):
        # svmem(total=10367352832, available=6472179712, percent=37.6, used=8186245120, free=2181107712, active=4748992512,
        # inactive=2758115328, buffers=790724608, cached=3500347392, shared=787554304, slab=199348224)
        gpu_status = self.get_gpu_util_status()
        self.gpu_mem_total = gpu_status[2]
        if gpu_status is not None:
            self.gpu_util_list.append(gpu_status[0])
            self.gpu_mem_util_list.append(gpu_status[1])
            # self.gpu_mem_total.append(gpu_status[2])
            self.gpu_mem_used_list.append(gpu_status[3])
        self.cpu_util_list.append(self.get_cpu_util_status())
        mem_status = self.get_mem_util_status()
        self.mem_total = mem_status.total/1024/1024
        self.mem_util_list.append(mem_status.percent)
        self.mem_used_list.append(mem_status.used/1024/1024)
        
    def show_status(self):
        cpu_mem_status = "CPU:{:.2f}% | MEM:{:.2f}MiB/{:.2f}MiB({:.2f}%)".format(np.mean(self.cpu_util_list),np.mean(self.mem_used_list),self.mem_total,np.mean(self.mem_util_list))
        gpu_status = "GPU:{:.2f}% | {:.2f}MiB/{:.2f}MiB({:.2f}%)".format(np.mean(self.gpu_util_list),np.mean(self.gpu_mem_used_list),self.gpu_mem_total,np.mean(self.gpu_mem_util_list))
        print(cpu_mem_status)
        print(gpu_status)
        self.cpu_util_list.clear()
        self.mem_util_list.clear()
        self.mem_used_list.clear()
        self.gpu_util_list.clear()
        self.gpu_mem_util_list.clear()
        self.gpu_mem_used_list.clear()

    def run(self):
        count = 0
        while True:
            time.sleep(0.5)
            self.record_status()
            count += 1
            if count == 100:
                self.show_status()
                count = 0
                
    def __del__(self):
        pynvml.nvmlShutdown()