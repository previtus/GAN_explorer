import matplotlib
matplotlib.use('TKAgg')

import mock
import os
import numpy as np

def all(model_path, architecture = 'StyleGAN2'):

    args = mock.Mock()
    args.architecture = architecture
    args.mode = 'explore'
    args.steps_speed = '120'
    args.conv_reconnect_str = '0.3'
    args.deploy = 'False'
    args.port = '8000'
    args.model_path = model_path

    from getter_functions import Getter
    from interaction_handler import Interaction_Handler

    steps_speed = int(args.steps_speed)
    version = "v2"  # "game"

    server_deployed = (args.deploy == "True")
    port = str(args.port) #port = "8000" # -> Uses a link for REST requests: "http://localhost:"+PORT+"/get_image"
    getter = Getter(args, USE_SERVER_INSTEAD=server_deployed, PORT=port)
    initial_resolution = 1024

    interaction_handler = Interaction_Handler(getter, initial_resolution)
    interaction_handler.convolutional_layer_reconnection_strength = float(args.conv_reconnect_str)

    pretrained_model = ("karras2018iclr" in args.model_path)
    if args.architecture == "ProgressiveGAN":
        if not pretrained_model:
            # << Pre-trained PGGAN models have tensors named as: "16x16/Conv0/weight" while our custom models have "16x16/Conv0_up/weight" -> probably due to the used tf versions
            interaction_handler.target_tensors = [tensor.replace("Conv0", "Conv0_up") for tensor in interaction_handler.target_tensors]
            interaction_handler.plotter.target_tensors = [tensor.replace("Conv0", "Conv0_up") for tensor in interaction_handler.plotter.target_tensors]
        if "-256x256.pkl" in args.model_path:
            interaction_handler.plotter.font_multiplier = 0.25
    ### StyleGAN2 layer naming is different:
    if args.architecture == "StyleGAN2":
        interaction_handler.target_tensors = ["G_synthesis/"+tensor.replace("Conv0", "Conv0_up") for tensor in interaction_handler.target_tensors]
        interaction_handler.plotter.target_tensors = ["G_synthesis/"+tensor.replace("Conv0", "Conv0_up") for tensor in interaction_handler.plotter.target_tensors]


    # plotter allowed only in local run
    if not server_deployed:
        pass
        #interaction_handler.plotter.prepare_with_set_tensors()

    interaction_handler.latent_vector_size = getter.get_vec_size_localServerSwitch()

    interaction_handler.shuffle_random_points(steps=steps_speed)

    ######### ACTION !!!!!!!
    global get_image_function, frame, number_of_frames, counting_run

    get_image_function = interaction_handler.get_interpolated_image_key_input
    interaction_handler.selected_feature_i = int(interaction_handler.latent_vector_size / 2.0)
    # interaction_handler.selected_feature_i = 10 # hmm is there an ordering?
    interaction_handler.previous = interaction_handler.p0
    interaction_handler.move_by = 1.0
    interaction_handler.SHIFT = False
    interaction_handler.ALT = False

    interaction_handler.saved = [None] * 10

    interaction_handler.renderer.show_fps = False
    name = interaction_handler.getter.model_name_id + "____client"
    folder = "renders/" + name + "/"
    return interaction_handler, folder, name



def LatentRiderV2_03_2021(interaction_handler):
    global get_image_function, frame, number_of_frames, counting_run
    ####v6 speed ups - mooore
    # 30 fps

    from matplotlib import pyplot as plt
    import cv2

    def showfce(frame):
        plt.figure(figsize = (10,10))

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        plt.imshow(img)
        #plt.show()

    key_code = "r"
    #key_code = ""
    frame = interaction_handler.renderer.return_frame_client_mode(get_image_function, key_code)
    showfce(frame)

    def savefce(image):
        global file_format
        folder = "renders/"
        if not os.path.exists(folder):
            os.mkdir(folder)
        name = interaction_handler.getter.model_name_id + "____client"
        folder = "renders/" + name + "/"
        if not os.path.exists(folder):
            os.mkdir(folder)

        filename = folder+"saved_" + str(interaction_handler.saved_already).zfill(4) + "." + file_format
        interaction_handler.saved_already += 1
        print("Saving in good quality as ", filename)

        cv2.imwrite(filename, image)

    def click_key_and_save(key_code):
        global get_image_function, frame, number_of_frames, counting_run
        if not counting_run:
            frame = interaction_handler.renderer.return_frame_client_mode(get_image_function, key_code)
            savefce(frame)
        
        number_of_frames += 1
        
    def black_frame():
        global number_of_frames, counting_run
        if not counting_run:
            frame = np.zeros((1024, 1024, 3))
            savefce(frame)
        number_of_frames += 1
        
    def black_frames(n = 10):
        for i in range(n):
            black_frame()


    ### Go from slow to fast and kinda peak
    ### Then after near-flickering end with black frames

    def alsogood30sec_endstoofastmaybe(): # P3
        upto = 4
        m = 15
        for k in range(m):
            # from 0 to 1
            speed = 0.05 + (float(k)/10.0) # slowly speeding up?
            speed /= (m/10.)
            speed *= upto
            speed += 0.17
            print(speed)

            parametrized(speed)
            

            
            
    def parametrized_from_list(list_of_speeds, onlyprintspeeds=False):
        for speed in list_of_speeds:
            print(speed)
            if not onlyprintspeeds: parametrized(speed)
    def parametrized_from_startend(start_at,end_at,number_of_steps, onlyprintspeeds=False):
        l = list(np.arange(start_at,end_at, (end_at-start_at)/number_of_steps))
        #print(l)
        parametrized_from_list(l, onlyprintspeeds)

            
    def nice40sec(m_len = 5, onlyprintspeeds = False): # P2
        m = m_len
        for k in range(m):
            # from 0 to 1
            speed = 0.15 + (float(k)/10.0) # slowly speeding up?
            speed /= (m/10.)
            print(speed)
            if not onlyprintspeeds: parametrized(speed)
        prev = speed
        for k in range(m):
            # from 0 to 1
            speed = 0.15 + (float(k)/10.0) # slowly speeding up?
            speed /= (m/10.)
            speed += prev
            print(speed)
            if not onlyprintspeeds: parametrized(speed)
        prev = speed
        for k in range(m):
            # from 0 to 1
            speed = 0.15 + (float(k)/10.0) # slowly speeding up?
            speed /= (m/10.)
            speed += prev
            print(speed)
            if not onlyprintspeeds: parametrized(speed)

        
    def nice30sec_speedup_from_slow_BACKUP(): # P1 nice, could I get it working multiple times?
        m = 5
        for k in range(m):
            # from 0 to 1
            speed = 0.15 + (float(k)/10.0) # slowly speeding up?
            speed /= (m/10.)
            print(speed)
            parametrized(speed)


    def slow_slow_then_suddenly_speedup():      # ST2b
        interaction_handler.move_by = 1.0 * 0.5 * 0.5 * 0.5 # - - -
        times = 3
        for j in range(times): # repeats
            click_key_and_save("r")
            for i in range(3* 3*3*4): # each one will have
                click_key_and_save("w")

        interaction_handler.move_by = 1.0 * 0.5 * 0.5 # - -
        times = 3
        for j in range(times): # repeats
            click_key_and_save("r")
            for i in range(2* 3*3*4): # each one will have
                click_key_and_save("w")


        interaction_handler.move_by = 1.0 * 0.5 # - # 0.5
        times = 5
        for j in range(times): # repeats
            click_key_and_save("r")
            for i in range(3*3*4): # each one will have
                click_key_and_save("w")

        interaction_handler.move_by = 0.75
        times = 5
        for j in range(times): # repeats
            click_key_and_save("r")
            for i in range(2*3*4): # each one will have
                click_key_and_save("w")

        interaction_handler.move_by = 1.0
        times = 4
        for j in range(times): # repeats
            click_key_and_save("r")
            for i in range(3*4): # each one will have
                click_key_and_save("w")


    def waggly_speedy_10sec(): # looks like beating   #~T5
        
        interaction_handler.move_by = 1.0 * 2 * 2 * 2 # + + +
        for j in range(10):
            click_key_and_save("r")
            for i in range(20+12 - 2*j):
                click_key_and_save("w")


    def peak_30sec(fps = 30, times = [4,3,2]):  ###~T4
        interaction_handler.move_by = 1.0
        
        for j in range(int(fps*2)):
            click_key_and_save("r")
            for i in range(times[0]):
                click_key_and_save("w")

        direction = True # flipping w/s
        for j in range(int(fps*2)):
            click_key_and_save("r")
            direction = not direction
            for i in range(times[1]):
                if direction:
                    click_key_and_save("w")
                else:
                    click_key_and_save("s")

        for j in range(int(fps*4)): # good but hard to follow, has to be the peak
            click_key_and_save("r")
            direction = not direction
            for i in range(times[2]):
                if direction:
                    click_key_and_save("w")
                else:
                    click_key_and_save("s")
    ### Finally run a slow one which slows even a bit more



            
    def standartized(JUMPSIZE = 1.0, REPEATS = 3, ONELOCATION = 30):
        interaction_handler.move_by = JUMPSIZE
        for j in range(REPEATS): # repeats
            click_key_and_save("r")
            for i in range(ONELOCATION): # each one will have
                click_key_and_save("w")

    def parametrized(speed = 1.0, REPEATS = 3, onelocation_def = 30.0):
        JUMPSIZE = 1.0 * speed
        ONELOCATION = int(onelocation_def / speed)
        interaction_handler.move_by = JUMPSIZE
        print(REPEATS,"*",ONELOCATION,"moving by", JUMPSIZE)
        for j in range(REPEATS): # repeats
            click_key_and_save("r")
            for i in range(ONELOCATION): # each one will have
                click_key_and_save("w")


    # slowdowns from slow to even slower ~

    def slowdown_end(m_len = 5, onlyprintspeeds=False): # P2
        # from 1 to ... 0.5?
        start_at = 0.075
        end_at = 0.04
        
        step_by = (start_at - end_at) / (m_len-1)
        speed = start_at
        for k in range(m_len):
            print(speed)
            if not onlyprintspeeds: parametrized(speed = speed, REPEATS = 1, onelocation_def= (9.0)) 
                # for m=3 -> 13.0 => 25 sec, 9 => 17 sec
                # what about m=4? 9=>23 and thats good i think
            
            speed = speed - step_by
            
        # maybe too slow?
       

    ## <<<[ FINAL CUT: C3 with vC and probably a later selected end scene. ]>>>


    # Composition 3 = C1, faster start, add black screen and slowed down end + C2 tweaking
    # C3 ~ < run next + rerun above
    # vC ... this setup works pretty good!
    #    - generate few iteration on this setup (several runs of the same)
    #    - optionally we can add a whole bunch of the end bits (slowdown_end(3)) as individual folders ... do this after (+- easy cut with black screen)

    number_of_frames = 0
    counting_run = False # False = real / True = only count frames!
    #counting_run = True

    #nice40sec(m_len = 10) # 1:20
    # tuned into:
    startat = 0.15 + 0.05 # kick a bit # not too much
    endat = 3.15 - 0.15
    #steps = 21
    steps = 18
    stepby = (endat-startat)/(steps)
    l = list(np.arange(startat,endat+stepby, stepby))
    l[1:10] = [v - .11 for v in l[1:8]]
    parametrized_from_list(l, False) # ~approx like nice40sec(m_len = 10) but faster?
    print("!!! Early Exit ",number_of_frames," frames ", int(100 * ((number_of_frames/30)/60)) / 100, "min !!!")
    return True

    ##################
    ## also could fully replace by P1 (would be much shorter and would need to think how to cut to wabbly parrt)
    ####

    waggly_speedy_10sec()
    peak_30sec(times = [3,2,2]) # thats about right.... prev [4,3,2]

    black_frames(40) # 1 sec black
    slowdown_end(3)    # ? sec  # slowdown_end(4)    # 23 sec
                       # 3 is kinda perfect, just need to wait for a good one though

    # 1:20 + 0:10 + 0:30 + 0:02 + 0:23 ~= 2:25 cca (with mix 3 min?)


    print("!!! ",number_of_frames," frames ", int(100 * ((number_of_frames/30)/60)) / 100, "min !!!")
    # !!!  3938  frames  2.18 min !!! <<< C3 .... do I want less than 2min?

    ### END
    # Only ends:
    number_of_frames = 0
    counting_run = False

    slowdown_end(3)    # ? sec  # slowdown_end(4)    # 23 sec
                       # 3 is kinda perfect, just need to wait for a good one though

    print("!!! ",number_of_frames," frames ", int(100 * ((number_of_frames/30)/60)) / 100, "min !!!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Project: VideoGen for GAN Explorer.')
    parser.add_argument('-network',
                        help='Path to the model (.pkl file) - this can be a pretrained ProgressiveGAN model, or just the Generator network (Gs).',
                        default='models/sg_bus35k_network-snapshot-001882-good.pkl')
    parser.add_argument('-architecture', help='GAN architecture type (support for "ProgressiveGAN"; work-in-progress also "StyleGAN2"). Defaults to "ProgressiveGAN".', default='ProgressiveGAN')
    parser.add_argument('-format', help='Format png or jpg', default='jpg')


    global file_format
    args = parser.parse_args()
    args.model_path = args.network
    
    file_format = args.format

    interaction_handler, folder, name = all(args.model_path, args.architecture)
    print("Folder:", folder)

    LatentRiderV2_03_2021(interaction_handler)

    command = "ffmpeg -r 30/1 -pattern_type glob -i '"+folder+"*."+file_format+"' -c:v libx264 -vf fps=30 -crf 15 -pix_fmt yuv420p vid_"+name+"_30fps.mp4"
    import subprocess

    print(command)
    #subprocess.call(['ffmpeg', '-i', 'test%d0.png', 'output.avi'])
    subprocess.call(command, shell=True)


    """
    #aerials 114k / 35k
    #args.model_path = "models/sg_aerials114k_network-snapshot-000982.pkl"
    #args.model_path = "models/sg_aerials35k_network-snapshot-000982.pkl"
    #bus
    args.model_path = "models/sg_bus35k_network-snapshot-001882-good.pkl" # later model - try the same v1/v6 tricks
    #boat
    #args.model_path = "models/sg_boat35k_network-snapshot-001473.pkl" # later model - try the same with flickering around maybe??
    #walk
    #args.model_path = "models/sg_walk35k_network-snapshot-001309.pkl"

    # previous model - tested it for experimental film, but seems like the cuts are slightly worse ...
    #args.architecture = 'ProgressiveGAN'
    #args.model_path = "/home/vitek/Vitek/python_codes/GAN_explorer/models/bus-snapshot-010300.pkl"
    """