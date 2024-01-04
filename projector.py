# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/stylegan2/license.html
#
#
# import pickle
# import numpy as np
# import tensorflow as tf
# import dnnlib
# import dnnlib.tflib as tflib
#
# #----------------------------------------------------------------------------
#
# def open_file_or_url(file_or_url):
#     if dnnlib.util.is_url(file_or_url):
#         return dnnlib.util.open_url(file_or_url, cache_dir='.stylegan2-cache')
#     return open(file_or_url, 'rb')
#
# def load_pkl(file_or_url):
#     with open_file_or_url(file_or_url) as file:
#         return pickle.load(file, encoding='latin1')
#
# class Projector:
#     def __init__(self):
#         self.num_steps                  = 1000
#         self.dlatent_avg_samples        = 10000
#         self.initial_learning_rate      = 0.1
#         self.initial_noise_factor       = 0.05
#         self.lr_rampdown_length         = 0.25
#         self.lr_rampup_length           = 0.05
#         self.noise_ramp_length          = 0.75
#         self.regularize_noise_weight    = 1e5
#         self.verbose                    = False
#         self.clone_net                  = True
#
#         self._Gs                    = None
#         self._minibatch_size        = None
#         self._dlatent_avg           = None
#         self._dlatent_std           = None
#         self._noise_vars            = None
#         self._noise_init_op         = None
#         self._noise_normalize_op    = None
#         self._dlatents_var          = None
#         self._noise_in              = None
#         self._dlatents_expr         = None
#         self._images_expr           = None
#         self._target_images_var     = None
#         self._lpips                 = None
#         self._dist                  = None
#         self._loss                  = None
#         self._reg_sizes             = None
#         self._lrate_in              = None
#         self._opt                   = None
#         self._opt_step              = None
#         self._cur_step              = None
#
#     def _info(self, *args):
#         if self.verbose:
#             print('Projector:', *args)
#
#     def set_network(self, Gs, minibatch_size=1):
#         assert minibatch_size == 1
#         self._Gs = Gs
#         self._minibatch_size = minibatch_size
#         if self._Gs is None:
#             return
#         if self.clone_net:
#             self._Gs = self._Gs.clone()
#
#         # Find dlatent stats.
#         self._info('Finding W midpoint and stddev using %d samples...' % self.dlatent_avg_samples)
#         latent_samples = np.random.RandomState(123).randn(self.dlatent_avg_samples, *self._Gs.input_shapes[0][1:])
#         print("latent_samples.shape", latent_samples.shape)
#         dlatent_samples = self._Gs.run(latent_samples, None)[:, :1, :] # [N, 1, 512]
#         print("dlatent_samples.shape", dlatent_samples.shape)
#         self._dlatent_avg = np.mean(dlatent_samples, axis=0, keepdims=True) # [1, 1, 512]
#         self._dlatent_std = (np.sum((dlatent_samples - self._dlatent_avg) ** 2) / self.dlatent_avg_samples) ** 0.5
#         self._info('std = %g' % self._dlatent_std)
#
#         # Find noise inputs.
#         self._info('Setting up noise inputs...')
#         self._noise_vars = []
#         noise_init_ops = []
#         noise_normalize_ops = []
#         while True:
#             n = 'G_synthesis/noise%d' % len(self._noise_vars)
#             if not n in self._Gs.vars:
#                 break
#             v = self._Gs.vars[n]
#             self._noise_vars.append(v)
#             noise_init_ops.append(tf.assign(v, tf.random_normal(tf.shape(v), dtype=tf.float32)))
#             noise_mean = tf.reduce_mean(v)
#             noise_std = tf.reduce_mean((v - noise_mean)**2)**0.5
#             noise_normalize_ops.append(tf.assign(v, (v - noise_mean) / noise_std))
#             self._info(n, v)
#         self._noise_init_op = tf.group(*noise_init_ops)
#         self._noise_normalize_op = tf.group(*noise_normalize_ops)
#
#         # Image output graph.
#         self._info('Building image output graph...')
#         self._dlatents_var = tf.Variable(tf.zeros([self._minibatch_size] + list(self._dlatent_avg.shape[1:])), name='dlatents_var')
#         self._noise_in = tf.placeholder(tf.float32, [], name='noise_in')
#         dlatents_noise = tf.random.normal(shape=self._dlatents_var.shape) * self._noise_in
#         self._dlatents_expr = tf.tile(self._dlatents_var + dlatents_noise, [1, self._Gs.components.synthesis.input_shape[1], 1])
#         self._images_expr = self._Gs.components.synthesis.get_output_for(self._dlatents_expr, randomize_noise=False)
#
#         # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
#         proc_images_expr = (self._images_expr + 1) * (255 / 2)
#         sh = proc_images_expr.shape.as_list()
#         if sh[2] > 256:
#             factor = sh[2] // 256
#             proc_images_expr = tf.reduce_mean(tf.reshape(proc_images_expr, [-1, sh[1], sh[2] // factor, factor, sh[2] // factor, factor]), axis=[3,5])
#
#         # Loss graph.
#         self._info('Building loss graph...')
#         self._target_images_var = tf.Variable(tf.zeros(proc_images_expr.shape), name='target_images_var')
#         if self._lpips is None:
#             self._lpips = misc.load_pkl('https://nvlabs-fi-cdn.nvidia.com/stylegan/networks/metrics/vgg16_zhang_perceptual.pkl')
#         self._dist = self._lpips.get_output_for(proc_images_expr, self._target_images_var)
#         self._loss = tf.reduce_sum(self._dist)
#
#         # Noise regularization graph.
#         self._info('Building noise regularization graph...')
#         reg_loss = 0.0
#         for v in self._noise_vars:
#             sz = v.shape[2]
#             while True:
#                 reg_loss += tf.reduce_mean(v * tf.roll(v, shift=1, axis=3))**2 + tf.reduce_mean(v * tf.roll(v, shift=1, axis=2))**2
#                 if sz <= 8:
#                     break # Small enough already
#                 v = tf.reshape(v, [1, 1, sz//2, 2, sz//2, 2]) # Downscale
#                 v = tf.reduce_mean(v, axis=[3, 5])
#                 sz = sz // 2
#         self._loss += reg_loss * self.regularize_noise_weight
#
#         # Optimizer.
#         self._info('Setting up optimizer...')
#         self._lrate_in = tf.placeholder(tf.float32, [], name='lrate_in')
#         self._opt = dnnlib.tflib.Optimizer(learning_rate=self._lrate_in)
#         self._opt.register_gradients(self._loss, [self._dlatents_var] + self._noise_vars)
#         self._opt_step = self._opt.apply_updates()
#
#     def run(self, target_images):
#         # Run to completion.
#         self.start(target_images)
#         while self._cur_step < self.num_steps:
#             self.step()
#
#         # Collect results.
#         pres = dnnlib.EasyDict()
#         pres.dlatents = self.get_dlatents()
#         pres.noises = self.get_noises()
#         pres.images = self.get_images()
#         return pres
#
#     def start(self, target_images):
#         assert self._Gs is not None
#
#         # Prepare target images.
#         self._info('Preparing target images...')
#         target_images = np.asarray(target_images, dtype='float32')
#         target_images = (target_images + 1) * (255 / 2)
#         sh = target_images.shape
#         assert sh[0] == self._minibatch_size
#         if sh[2] > self._target_images_var.shape[2]:
#             factor = sh[2] // self._target_images_var.shape[2]
#             target_images = np.reshape(target_images, [-1, sh[1], sh[2] // factor, factor, sh[3] // factor, factor]).mean((3, 5))
#
#         # Initialize optimization state.
#         self._info('Initializing optimization state...')
#         tflib.set_vars({self._target_images_var: target_images, self._dlatents_var: np.tile(self._dlatent_avg, [self._minibatch_size, 1, 1])})
#         tflib.run(self._noise_init_op)
#         self._opt.reset_optimizer_state()
#         self._cur_step = 0
#
#     def step(self):
#         assert self._cur_step is not None
#         if self._cur_step >= self.num_steps:
#             return
#         if self._cur_step == 0:
#             self._info('Running...')
#
#         # Hyperparameters.
#         t = self._cur_step / self.num_steps
#         noise_strength = self._dlatent_std * self.initial_noise_factor * max(0.0, 1.0 - t / self.noise_ramp_length) ** 2
#         lr_ramp = min(1.0, (1.0 - t) / self.lr_rampdown_length)
#         lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
#         lr_ramp = lr_ramp * min(1.0, t / self.lr_rampup_length)
#         learning_rate = self.initial_learning_rate * lr_ramp
#
#         # Train.
#         feed_dict = {self._noise_in: noise_strength, self._lrate_in: learning_rate}
#         _, dist_value, loss_value = tflib.run([self._opt_step, self._dist, self._loss], feed_dict)
#         tflib.run(self._noise_normalize_op)
#
#         # Print status.
#         self._cur_step += 1
#         if self._cur_step == self.num_steps or self._cur_step % 10 == 0:
#             self._info('%-8d%-12g%-12g' % (self._cur_step, dist_value, loss_value))
#         if self._cur_step == self.num_steps:
#             self._info('Done.')
#
#     def get_cur_step(self):
#         return self._cur_step
#
#     def get_dlatents(self):
#         return tflib.run(self._dlatents_expr, {self._noise_in: 0})
#
#     def get_noises(self):
#         return tflib.run(self._noise_vars)
#
#     def get_images(self):
#         return tflib.run(self._images_expr, {self._noise_in: 0})

#----------------------------------------------------------------------------

# examples at:
# - https://github.com/tkarras/progressive_growing_of_gans/pull/4
# - https://discuss.pytorch.org/t/how-to-load-a-pt-pretained-model-using-scipy-misc/72045

network_pkl = "/home/vitek/Vitek/Art/GAN_explorer/models/_model_downloads_/progressive_growing_of_gans-CuriosityHazCam/CuriosityHazCam-013770-NejFrekvence_SumAOpakovaniSliti.pkl"
image_path = "/home/vitek/Vitek/Art/GAN_explorer/renders/CuriosityHazCam-013770-NejFrekvence_SumAOpakovaniSliti/kinda nice/saved_000024.png"
image_path = "/home/vitek/Vitek/Art/GAN_explorer/latents/CuriosityHazCam-013370-KonkretnejsiTvarovejsiPohybovejsiPlynulejsi_A/Asaved_000018.png"
init_latent = None
init_latent = "/home/vitek/Vitek/Art/GAN_explorer/renders/CuriosityHazCam-013770-NejFrekvence_SumAOpakovaniSliti/kinda nice/saved_000024.txt"
init_latent = "/home/vitek/Vitek/Art/GAN_explorer/latents/CuriosityHazCam-013370-KonkretnejsiTvarovejsiPohybovejsiPlynulejsi_A/Asaved_000014.txt"

target_folder = "renders/projected/"

import numpy as np
from PIL import Image
img = Image.open(image_path)
np_img = np.asarray(img)
np_img = np_img[:,:,0] # rgb->gray
print("image shape,", np_img.shape)
print("min,mean,max", np.min(np_img), np.mean(np_img), np.max(np_img), ) # min,mean,max 0 107.44881439208984 255

# def load_images(images_list, img_size):
#     loaded_images = list()
#     for img_path in images_list:
#         img = image.load_img(img_path, target_size=(img_size, img_size))
#         img = np.expand_dims(img, 0)
#         loaded_images.append(img)
#     loaded_images = np.vstack(loaded_images)
#     preprocessed_images = preprocess_input(loaded_images)
#     return preprocessed_images
#

from progressive_gan_handler import ProgressiveGAN_Handler
from settings import Settings
import mock

args = mock.Mock()
args.architecture = "ProgressiveGAN"
args.model_path = network_pkl
print(" ... loading from ... ", args.model_path)
settings = Settings()
pro_handler = ProgressiveGAN_Handler(settings, args)
pro_handler.report()

example_input = pro_handler.example_input()
example_output = pro_handler.infer(example_input)

print("example_input:", example_input.shape)
print("example_output:", example_output.shape)
print("min,mean,max", np.min(example_output), np.mean(example_output), np.max(example_output), ) # min,mean,max 0 57.41764163970947 255

#
# import tensorflow as tf
#
# def printTensors(pb_file):
#     # read pb into graph_def
#     with tf.gfile.GFile(pb_file, "rb") as f:
#         graph_def = tf.GraphDef()
#         graph_def.ParseFromString(f.read())
#
#     # import graph_def
#     with tf.Graph().as_default() as graph:
#         tf.import_graph_def(graph_def)
#
#     # print operations
#     for op in graph.get_operations():
#         print(op.name)
#
#
# printTensors("VGG16_beauty_rates.pt")
# sadadsa

Gs = pro_handler._Gs

import cv2
import numpy as np
# random start:
if init_latent is None:
    latents = np.random.RandomState(1).randn(1000, *Gs.input_shapes[0][1:]) # 1000 random latents
    latents = latents[[0]] # hand-picked top-1
else:
    latent = np.loadtxt(init_latent)
    print("loaded", latent.shape, "from", init_latent)
    latents = np.asarray([latent])

labels = np.zeros([latents.shape[0]] + Gs.input_shapes[1][1:])

# img = load_image(image_path)
# img = np.random.RandomState(1).randn(1,1, 1024, 1024)



img = np.asarray([[np_img]])
# itterations = 2000
# itterations = 100
# learning_rate = 0.001
# learning_rate = 0.000001
# history = Gs.reverse_gan_for_etalons_v1(latents, labels, img, itterations=itterations, learning_rate=learning_rate)

iters = 2000
iters = 20000
iters = 10000
# without vgg
learning_rate=0.1 # all same - maybe jumped over?
learning_rate=0.001 # trosku slo ale mozna pomalu?
# learning_rate=0.01 # again maybe a bit jumpy ...

alpha=0.7 # < turned off anyway
# for now turned off "beholder beauty"
history = Gs.reverse_gan_for_etalons_v2(latents, labels, img, iters=iters, learning_rate=learning_rate, alpha=alpha)

### ^ imo very slow and doesnt much work!

for idx, h in enumerate(history):
    loss, latent = h
    print(idx, loss, "=>", latent.shape)

    image = pro_handler.infer(latent)[0]

    filename = target_folder+"projected_"+str(idx).zfill(3)+".png"
    cv2.imwrite(filename, image)
    np.savetxt(filename.replace(".png", ".txt"), latents[0])  # also save the latent

# proj = Projector()
# proj.set_network(Gs)
# noise_vars = [var for name, var in Gs.components.synthesis.vars.items() if name.startswith('noise')]
#
#
# truncation_psi = 1.0
# num_snapshots = 5
# seeds = [0,1,5] # ...
#
# Gs_kwargs = dnnlib.EasyDict()
# Gs_kwargs.randomize_noise = False
# Gs_kwargs.truncation_psi = truncation_psi


# #----------------------------------------------------------------------------
# # Image utils.
# import PIL.Image
# import PIL.ImageFont
#
# def adjust_dynamic_range(data, drange_in, drange_out):
#     if drange_in != drange_out:
#         scale = (np.float32(drange_out[1]) - np.float32(drange_out[0])) / (np.float32(drange_in[1]) - np.float32(drange_in[0]))
#         bias = (np.float32(drange_out[0]) - np.float32(drange_in[0]) * scale)
#         data = data * scale + bias
#     return data
#
# def create_image_grid(images, grid_size=None):
#     assert images.ndim == 3 or images.ndim == 4
#     num, img_w, img_h = images.shape[0], images.shape[-1], images.shape[-2]
#
#     if grid_size is not None:
#         grid_w, grid_h = tuple(grid_size)
#     else:
#         grid_w = max(int(np.ceil(np.sqrt(num))), 1)
#         grid_h = max((num - 1) // grid_w + 1, 1)
#
#     grid = np.zeros(list(images.shape[1:-2]) + [grid_h * img_h, grid_w * img_w], dtype=images.dtype)
#     for idx in range(num):
#         x = (idx % grid_w) * img_w
#         y = (idx // grid_w) * img_h
#         grid[..., y : y + img_h, x : x + img_w] = images[idx]
#     return grid
#
# def convert_to_pil_image(image, drange=[0,1]):
#     assert image.ndim == 2 or image.ndim == 3
#     if image.ndim == 3:
#         if image.shape[0] == 1:
#             image = image[0] # grayscale CHW => HW
#         else:
#             image = image.transpose(1, 2, 0) # CHW -> HWC
#
#     image = adjust_dynamic_range(image, drange, [0,255])
#     image = np.rint(image).clip(0, 255).astype(np.uint8)
#     fmt = 'RGB' if image.ndim == 3 else 'L'
#     return PIL.Image.fromarray(image, fmt)
#
# def save_image_grid(images, filename, drange=[0,1], grid_size=None):
#     convert_to_pil_image(create_image_grid(images, grid_size), drange).save(filename)
#
# def apply_mirror_augment(minibatch):
#     mask = np.random.rand(minibatch.shape[0]) < 0.5
#     minibatch = np.array(minibatch)
#     minibatch[mask] = minibatch[mask, :, :, ::-1]
#     return minibatch
#
# def project_image(proj, targets, png_prefix, num_snapshots):
#     snapshot_steps = set(proj.num_steps - np.linspace(0, proj.num_steps, num_snapshots, endpoint=False, dtype=int))
#     misc.save_image_grid(targets, png_prefix + 'target.png', drange=[-1,1])
#     proj.start(targets)
#     while proj.get_cur_step() < proj.num_steps:
#         print('\r%d / %d ... ' % (proj.get_cur_step(), proj.num_steps), end='', flush=True)
#         proj.step()
#         if proj.get_cur_step() in snapshot_steps:
#             misc.save_image_grid(proj.get_images(), png_prefix + 'step%04d.png' % proj.get_cur_step(), drange=[-1,1])
#     print('\r%-30s\r' % '', end='', flush=True)
#
#
# for seed_idx, seed in enumerate(seeds):
#     print('Projecting seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
#     rnd = np.random.RandomState(seed)
#     z = rnd.randn(1, *Gs.input_shape[1:])
#     tflib.set_vars({var: rnd.randn(*var.shape.as_list()) for var in noise_vars})
#     images = Gs.run(z, None, **Gs_kwargs)
#     project_image(proj, targets=images, png_prefix=dnnlib.make_run_dir_path('seed%04d-' % seed), num_snapshots=num_snapshots)
#
