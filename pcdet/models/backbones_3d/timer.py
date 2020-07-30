import time
import torch

def avg_onthefly(cnt, old_avg, new_val):
    return  old_avg + (new_val - old_avg) / cnt

class Timer():
    def __init__(self, cuda_sync=True):
        self.cnt = 0
        self.start_time = 0
        self.end_time = 0
        self.events : [str] = [ ]
        self.last_time = 0
        self.event_time = {}

        self.cuda_sync = cuda_sync
        pass

    def start(self):
        if self.cuda_sync:
            torch.cuda.synchronize()

        self.cnt += 1
        self.start_time = self.last_time =  time.time()

    def record(self, event:str):
        if self.cuda_sync:
            torch.cuda.synchronize()

        if not event in self.event_time:
            self.events.append(event)
        now = time.time()
        event_time = now - self.last_time
        self.last_time = now

        if self.cnt > 1:
            # avg
            event_time = avg_onthefly(self.cnt, self.event_time[event], event_time)

        self.event_time[event] = event_time

    def end(self):
        if self.cuda_sync:
            torch.cuda.synchronize()

        now = time.time()

        total_time = now - self.start_time
        if self.cnt > 1:
            # avg
            total_time = avg_onthefly(self.cnt, self.total_time, total_time)
        self.total_time = total_time

        print(" loop cnt = ", self.cnt)
        for e in self.events:
            print(e,"\t", end='')
        print("total")
        print()
        for e in self.events:
            print("{:5.5f}".format(self.event_time[e]), "\t", end='')
        print("{:5.5f}".format(self.total_time))
        print()


if __name__ == '__main__':
    timer = Timer(cuda_sync=False)
    for i in range(4):
        timer.start()
        time.sleep(0.3)
        timer.record("a")
        time.sleep(0.2)
        timer.record("b")
        time.sleep(0.5)
        timer.end()



