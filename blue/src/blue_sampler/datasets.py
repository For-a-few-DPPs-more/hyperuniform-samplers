from squarenet.sampler import samplepoints, list_methods

def generate_dataset(size = (1000, 2), method = "barbara", plot_points = True, list_method = True):
    if list_method:
        print("source: squarenet.sampler from module squarenet")
        print("available datasets:", list_methods())
    return samplepoints(method = method, size = size, plot_points = plot_points)