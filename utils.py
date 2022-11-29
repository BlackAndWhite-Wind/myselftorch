import matplotlib.pyplot as plt
import imageio

def draw_function(draw_count=None):
    data = []
    plt.ion()
    ims=[]
    fig=plt.figure()
    def __draw(_data, keep=False):
        data.append(_data)
        length = len(data)
        start = 0
        if (draw_count is not None) and isinstance(draw_count, int) and (draw_count < length):
            draw_data = data[-draw_count:]
            start = length - draw_count
        else:
            draw_data = data
        idx = list(range(start, length))
        plt.clf()
        plt.title("Loss Of Train")
        plt.xlabel("Epoh")
        plt.ylabel("Loss")
        plt.plot(idx, draw_data)
        plt.savefig('temp.png')
        ims.append(imageio.imread('temp.png'))
        plt.pause(0.1)
        if keep:
            plt.show()
        else:
            plt.savefig("train loss.png")
            imageio.mimsave('pic1.gif', ims, duration=0.5)
            plt.ioff()

    return __draw