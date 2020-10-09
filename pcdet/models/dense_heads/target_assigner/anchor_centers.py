
from math import floor

class Centers:
    """ a evenly spaced array float """
    def __init__(self, start:float, delta:float, num:float):
        self.start = start
        self.delta = delta
        self.num = num

    def conv(self, kernel_size, stride, padding):
        start = self.start - padding * self.delta + (kernel_size - 1) / 2 * self.delta
        delta = self.delta * stride
        num = floor((self.num + 2 * padding - (kernel_size - 1) - 1) / stride + 1)
        return Centers(start, delta, num)

    def deconv(self, kernel_size, stride, padding):
        pass

    def __str__(self):
        return "start = {}, delta = {}, num = {}".format(self.start, self.delta, self.num)

def range_res2centers(value_range:[float], res:int) -> Centers:
    """
    Args:
        value_range : low and high
        res: resolution

    """
    low, high = value_range
    delta = (high - low) / res
    start = low + delta / 2
    return Centers(start, delta, res)


from math import log
from math import radians

def test_centers():
    # centers = Centers(0.1, 0.1, 4);
    # centers.conv(3, 2, 1)
    # print(centers.conv(3, 2, 1))
    centers_logr = range_res2centers([log(6), log(70.4)], 512)
    centers_logr = centers_logr.conv(3, 2, 1)\
                               .conv(3, 2, 1)\
                               .conv(3, 1, 1)

    print(centers_logr)

    phi = range_res2centers([radians(-45), radians(45)], 512)
    phi = phi.conv(3,2,1).conv(3,2,1).conv(3,1,1)
    print(phi)

    print(range_res2centers([-45, 45], 3))

if __name__ == '__main__':
    test_centers()