import pickle, pickletools
import numpy as np
import tensorflow as tf
from timeit import default_timer as timer
import PIL.Image
import reconnector
import os

class ProgressiveGAN_Handler(object):
    """
    Handles a trained Progressive GAN model, loads from
    """

    def __init__(self, settings, args):
        # Initialization, should create the model, load it and also run one inference (to build the graph)
        self.settings = settings
        print("Init handler with path =", args.model_path)

        _, name = os.path.split(args.model_path)
        name = str(name).replace(".pkl", "")
        name = str(name).replace(".pt", "")
        self.model_name_id = name

        # Load and create a model
        self._create_model(args.model_path)

        self.latent_vector_size = self._Gs.input_shapes[0][1:][0]

        self._mode_intermediate_output = False
        self._mode_intermediate_output_layer_name = "" # "Gs/Grow_lod1/add:0"

        # Infer once to build up the graph (takes like 2 sec extra time is the first one)
        print("Testing with an example to build the graph")
        self._example_input = self.example_input(verbose=False)
        self._example_output = self.infer(self._example_input, verbose=False)

        # Load other networks (if we set these)
        self.multiple_nets_paths = args.multiple_nets
        self.multiple_nets = None
        if self.multiple_nets_paths is not "":
            self.prepare_multiple_nets()

        # Network altering:
        self.original_weights = {}

    def load_model(self,path):
        #tf.InteractiveSession()
        with open(path, 'rb') as file:
            loaded_networks = pickle.load(file)
            if isinstance(loaded_networks, tuple):
                # Original model saves a tuple (Generator, Discriminator, weighted avg Generator)
                _G, _D, _Gs = loaded_networks
            else:
                _Gs = loaded_networks
        return _Gs

    def _create_model(self, model_path):
        tf.InteractiveSession()
        with open(model_path, 'rb') as file:
            print(file)

            proto_op = next(pickletools.genops(file))
            assert proto_op[0].name == 'PROTO'
            proto_ver = proto_op[1]
            print("Pickled with version", proto_ver)

            loaded_networks = pickle.load(file)
            if isinstance(loaded_networks, tuple):
                # Original model saves a tuple (Generator, Discriminator, weighted avg Generator)
                self._G, self._D, self._Gs = loaded_networks
            else:
                self._Gs = loaded_networks

            """
            # DEBUG if we wanted to save something!
            def save_pkl(obj, filename):
                with open(filename, 'wb') as file:
                    pickle.dump(obj, file, protocol=pickle.HIGHEST_PROTOCOL)

            save_pkl(self._Gs, "_GsOnly.pkl")
            """

    def report(self):

        print("[ProgressiveGAN_Handler Status report]")
        print("\t- latent_vector_size:",self.latent_vector_size)
        print("\t- typical input shape is:",self._example_input.shape)
        print("\t- typical output shape is:",self._example_output.shape)

    def example_input(self, how_many=1, seed=None, verbose=True):
        if seed:
            if verbose:
                print("Generating random input (from seed=",seed,")")
            latents = np.random.RandomState(seed).randn(how_many, self.latent_vector_size)
        else:
            if verbose:
                print("Generating random input")
            latents = np.random.randn(how_many, self.latent_vector_size)

        if verbose:
            print("example input is ...", latents.shape)
        example_input = latents
        return example_input

    def infer(self, input_latents, verbose=True):
        labels = np.zeros([input_latents.shape[0]] + self._Gs.input_shapes[1][1:])

        if self._mode_intermediate_output:
            images = self._Gs.custom_tinkered_layer_output_run(self._mode_intermediate_output_layer_name,False,input_latents,labels)
        else:
            images = self._Gs.run(input_latents,labels)

        # Convert images to PIL-compatible format.
        images = np.clip(np.rint((images + 1.0) / 2.0 * 255.0), 0.0, 255.0).astype(np.uint8)  # [-1,1] => [0,255]
        images = images.transpose(0, 2, 3, 1)  # NCHW => NHWC

        if verbose:
            print("Generated",images.shape)

        return images

    def set_mode(self, intermediate_output_layer=None):
        print("Setting mode 'intermediate_output_layer' to", intermediate_output_layer)
        if intermediate_output_layer is None:
            self._mode_intermediate_output = False
        else:
            self._mode_intermediate_output = True
            self._mode_intermediate_output_layer_name = intermediate_output_layer

    def save_image(self, image, name="foo.jpg"):
        im = PIL.Image.fromarray(image)
        im.save(name)

    def times_a(self, a, np_arr):
        return a * np_arr

    def change_net(self, target_tensor, operation, *kwargs):
        # PS: Changing the weights is faster than generating an image... (which is good news)
        np_arr = None
        net = self._Gs

        # restore old weights
        for tensor_key in self.original_weights.keys():
            orig_val = self.original_weights[tensor_key]
            net.set_var(tensor_key, orig_val)

        if target_tensor not in self.original_weights:
            # we haven't change this tensor yet - load it from the NN
            np_arr = net.get_var(target_tensor)  # <slow
            self.original_weights[target_tensor] = np_arr

        np_arr = self.original_weights[target_tensor]
        print("tensor as np_arr:", type(np_arr), np_arr.shape)
        np_arr = operation(np_arr, kwargs)
        net.set_var(target_tensor, np_arr)

        self._Gs = net

    def restore(self):
        net = self._Gs
        editednet = reconnector.restore_net(net)
        self._Gs = editednet

    def reconnect(self, target_tensor, percent_change = 30):
        net = self._Gs
        editednet = reconnector.reconnect(net, target_tensor, percent_change)
        self._Gs = editednet

    def reconnect_simulate_random_weights(self, target_tensor, percent_change = 30):
        net = self._Gs
        editednet = reconnector.reconnect_simulate_random_weights(net, target_tensor, percent_change)
        self._Gs = editednet

    def savenet(self):
        # usually haxed net
        net = self._Gs
        def save_pkl(obj, filename):
            with open(filename, 'wb') as file:
                pickle.dump(obj, file, protocol=pickle.HIGHEST_PROTOCOL)
        save_pkl(self._Gs, "renders/haxed_gs.pkl")

    def DEBUG(self):
        print("####DEBUG####")
        print(self._Gs)
        print("#############")

    def prepare_multiple_nets(self):
        multiple_nets = str(self.multiple_nets_paths).split(",")
        print("Loading multiple nets from", len(multiple_nets), "additional nets:")

        self.multiple_nets = []
        self.multiple_nets_i = 0 # marks the currently loaded net ...
        self.multiple_nets.append(self._Gs)

        for path in multiple_nets:
            print("-", path)

            generator_net = self.load_model(path)
            self.multiple_nets.append(generator_net)

        print("Successfully loaded ", len(self.multiple_nets), "additional nets!")

        # We can preload them like this:
        preload = True # slower window load, but then very speedy cycling ...
        if preload:
            for i in range(len(self.multiple_nets)):
                self.cycle_multi_nets()
                self.infer(self._example_input, verbose=False)

    def cycle_multi_nets(self, direction=1):
        multiple_nets_i = self.multiple_nets_i + direction
        if multiple_nets_i >= len(self.multiple_nets):
            multiple_nets_i = 0
        if multiple_nets_i < 0:
            multiple_nets_i = len(self.multiple_nets)-1
        to_load = self.multiple_nets[multiple_nets_i]

        print("Cycled the Generator network!")
        self._Gs = to_load
        self.multiple_nets_i = multiple_nets_i

# Example of usage:
"""
pro_path = "aerials512vectors1024px_snapshot-010200.pkl"
pro_handler = ProgressiveGAN_Handler(pro_path)

pro_handler.report()

example_input = pro_handler.example_input()
example_output = pro_handler.infer(example_input)

print("example_output:", example_output.shape)

from timeit import default_timer as timer

# Basic measurements

repeats = 15
times = []
for repeat_i in range(repeats):
    t_infer = timer()

    example_input = pro_handler.example_input(verbose=False)
    example_output = pro_handler.infer(example_input, verbose=False)

    t_infer = timer() - t_infer
    if repeat_i > 0:
        times.append(t_infer)
    #print("Prediction (of 1 sample) took", t_infer, "sec.")

times = np.asarray(times)
print("Statistics:")
print("prediction time - avg +- std =", np.mean(times), "+-", np.std(times), "sec.")

# Batch measurements

how_many = 4 # too big? gpu mem explodes
repeats = 15
times = []
for repeat_i in range(repeats):
    t_infer = timer()

    example_inputs = pro_handler.example_input(how_many = how_many, verbose=False)
    example_outputs = pro_handler.infer(example_inputs, verbose=False)

    t_infer = timer() - t_infer

    if repeat_i > 0:
        times.append(t_infer)
    #print("Prediction (of",how_many,"samples) took", t_infer, "sec.")

times = np.asarray(times)
print("Statistics:")
print("prediction of whole",how_many," took time - avg +- std =", np.mean(times), "+-", np.std(times), "sec.")
print("prediction as divided for one - avg +- std =", np.mean(times/how_many), "+-", np.std(times/how_many), "sec.")
"""