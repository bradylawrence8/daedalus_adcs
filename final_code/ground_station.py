import customtkinter as ctk
import tkinter as tk
from PIL import Image
import math
import sys
import socket
from threading import Thread, Event
import time


tcpsocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 

tcpsocket.connect(('172.30.185.61', 8000)) 

def getdata():
    global ax
    global ay
    global az
    global gx
    global gy
    global gz
    global rpm1
    global rpm2
    global rpm3
    while True:
        data = tcpsocket.recv(1024)
        strings = data.decode('utf8')
        if ";" in strings:
            strings = strings.split(';', 1)[0]
            stringlist = strings.split(',')
            if len(stringlist) == 9 and not '' in stringlist:
                #print(stringlist)
                if '.' in stringlist[0]:
                    ax = float(stringlist[0])
                ay = float(stringlist[1])
                az = float(stringlist[2])
                gx = float(stringlist[3])
                gy = float(stringlist[4])
                gz = float(stringlist[5])
                rpm1 = float(stringlist[6])
                rpm2 = float(stringlist[7])
                rpm3 = float(stringlist[8])
                time.sleep(.001)

def appwindow():
    global ax
    global ay
    global az
    global gx
    global gy
    global gz
    global rpm1
    global rpm2
    global rpm3
    ax = 0.0
    ay = 0.0
    az = 0.0
    gx = 0.0
    gy = 0.0
    gz = 0.0
    rpm1 = 0.0
    rpm2 = 0.0
    rpm3 = 0.0
    # define tkinter app window
    app = ctk.CTk()
    #app.minsize(1536, 864)
    #app.maxsize(1536, 864)

    # center window on screen with a size of half of the screen in each dimension
    w = 1536
    h = 864
    appwidth = math.floor(w/2)
    appheight = math.floor(h/2)
    centerx = math.floor(w/4)
    centery = math.floor(h/4)
    geo = str(appwidth) + 'x' + str(appheight) + '+' + str(centerx) + '+' + str(centery)
    app.geometry(geo)

    # aesthetics stuff
    app.after(201, lambda :app.iconbitmap('capstone\\daedalus_transparent.ico')) # set window icon to daedalus logo :D
    ctk.set_appearance_mode('dark') # make dark mode so it looks nice
    ctk.set_default_color_theme('capstone\\theme.json') # set theme, custom theme contained in json file
    app.title('Daedalus Ground Station') # set title.

    # fonts setup
    title = ('', 15)
    header = ('', 13)
    body = ('', 12)

    # setup tabs for different use functions
    tabs = ctk.CTkTabview(master=app, width=appwidth, height=appheight, anchor='w', fg_color='#000A47', segmented_button_selected_color='darkblue') # sizing is going to look goofy until i actually fix it
    tabs.pack()
    inputs = tabs.add('Inputs')
    outputs = tabs.add('Outputs')
    tabs.set('Inputs')
    # tabs._segmented_button.grid_forget() # hide selector bar to make my own, still haven't done that yet bc im busy

    # setup functions when changing tab
    def select_inputs():
        tabs.configure(fg_color='#000A47', segmented_button_selected_color='darkblue', segmented_button_selected_hover_color='darkblue')

    def select_outputs():
        tabs.configure(fg_color='#A16B0E', segmented_button_selected_color='darkorange', segmented_button_selected_hover_color='darkorange')

    def tab_change():
        active_tab = tabs.get()
        if active_tab == "Inputs":
            select_inputs()
        if active_tab == "Outputs":
            select_outputs()

    tabs.configure(command=tab_change)

    # setup inputs tab

    # box for reaction wheel commands
    rw_commands = ctk.CTkFrame(master=inputs, width=math.floor(appwidth/3.2), height=math.floor(appheight*0.886), fg_color='darkblue')
    rw_commands.grid(padx=5, row=0, column=0)
    rw_commands.grid_propagate(0)
    rw_commands.grid_columnconfigure((0, 1, 2), weight=1)
    rw_title = ctk.CTkLabel(master=rw_commands, text='Reaction Wheel Commands', font=title)
    rw_title.grid(row=0, column=0, padx=25, columnspan=3)
    # torque:
    def torque_cmd():
        torque_label = ctk.CTkLabel(master=rw_commands, text='Torque Input:', font=header)
        torque_label.grid(padx=5, pady=5, row=1, column=0, sticky='w')
        torque_label_1 = ctk.CTkLabel(master=rw_commands, text='X Motor:', font=body)
        torque_label_1.grid(padx=5, pady=5, row=2, column=0, sticky='w')
        torque_input_1 = ctk.CTkEntry(master=rw_commands, width=100)
        torque_input_1.grid(padx=5, pady=5, row=2, column=1, sticky='w')
        torque_button_1 = ctk.CTkButton(master=rw_commands, text='✔', width=40, fg_color='#000A47')
        torque_button_1.grid(padx=5, pady=5, row=2, column=2, sticky='w')
        torque_label_2 = ctk.CTkLabel(master=rw_commands, text='Y Motor:', font=body)
        torque_label_2.grid(padx=5, pady=5, row=3, column=0, sticky='w')
        torque_input_2 = ctk.CTkEntry(master=rw_commands, width=100)
        torque_input_2.grid(padx=5, pady=5, row=3, column=1, sticky='w')
        torque_button_2 = ctk.CTkButton(master=rw_commands, text='✔', width=40, fg_color='#000A47')
        torque_button_2.grid(padx=5, pady=5, row=3, column=2, sticky='w')
        torque_label_3 = ctk.CTkLabel(master=rw_commands, text='Z Motor:', font=body)
        torque_label_3.grid(padx=5, pady=5, row=4, column=0, sticky='w')
        torque_input_3 = ctk.CTkEntry(master=rw_commands, width=100)
        torque_input_3.grid(padx=5, pady=5, row=4, column=1, sticky='w')
        torque_button_3 = ctk.CTkButton(master=rw_commands, text='✔', width=40, fg_color='#000A47')
        torque_button_3.grid(padx=5, pady=5, row=4, column=2, sticky='w')
    torque_cmd()

    # rpm:
    def rpm_cmd():
        rpm_label = ctk.CTkLabel(master=rw_commands, text='RPM Input:', font=header)
        rpm_label.grid(padx=5, pady=5, row=5, column=0, sticky='w')
        rpm_label_1 = ctk.CTkLabel(master=rw_commands, text='X Motor:', font=body)
        rpm_label_1.grid(padx=5, pady=5, row=6, column=0, sticky='w')
        rpm_input_1 = ctk.CTkEntry(master=rw_commands, width=100)
        rpm_input_1.grid(padx=5, pady=5, row=6, column=1, sticky='w')
        rpm_button_1 = ctk.CTkButton(master=rw_commands, text='✔', width=40, fg_color='#000A47')
        rpm_button_1.grid(padx=5, pady=5, row=6, column=2, sticky='w')
        rpm_label_2 = ctk.CTkLabel(master=rw_commands, text='Y Motor:', font=body)
        rpm_label_2.grid(padx=5, pady=5, row=7, column=0, sticky='w')
        rpm_input_2 = ctk.CTkEntry(master=rw_commands, width=100)
        rpm_input_2.grid(padx=5, pady=5, row=7, column=1, sticky='w')
        rpm_button_2 = ctk.CTkButton(master=rw_commands, text='✔', width=40, fg_color='#000A47')
        rpm_button_2.grid(padx=5, pady=5, row=7, column=2, sticky='w')
        rpm_label_3 = ctk.CTkLabel(master=rw_commands, text='Z Motor:', font=body)
        rpm_label_3.grid(padx=5, pady=5, row=8, column=0, sticky='w')
        rpm_input_3 = ctk.CTkEntry(master=rw_commands, width=100)
        rpm_input_3.grid(padx=5, pady=5, row=8, column=1, sticky='w')
        rpm_button_3 = ctk.CTkButton(master=rw_commands, text='✔', width=40, fg_color='#000A47')
        rpm_button_3.grid(padx=5, pady=5, row=8, column=2, sticky='w')
    rpm_cmd()

    # box for attitude commands
    att_commands = ctk.CTkFrame(master=inputs, width=math.floor(appwidth/3.2), height=math.floor(1.8*appheight/3*0.886), fg_color='darkblue')
    att_commands.grid_propagate(0)
    rw_commands.grid_columnconfigure((0, 1), weight=1)
    att_commands.grid(padx=5, row=0, column=1, sticky='n')
    ctk.CTkLabel(master=att_commands, text='Attitude Commands', font=title).grid(row=0, column=0, columnspan=2, padx=55)

    def eulerangles():
        parameterization_label.configure(text='Euler Angles:')
        attitude_label_1.configure(text='Yaw:')
        attitude_label_2.configure(text='Pitch:')
        attitude_label_3.configure(text='Roll:')
        attitude_label_1.grid(padx=5, pady=5, row=3, column=0, sticky='w')
        attitude_input_1.grid(padx=5, pady=5, row=3, column=1, sticky='e')
        attitude_label_2.grid(padx=5, pady=5, row=4, column=0, sticky='w')
        attitude_input_2.grid(padx=5, pady=5, row=4, column=1, sticky='e')
        attitude_label_3.grid(padx=5, pady=5, row=5, column=0, sticky='w')
        attitude_input_3.grid(padx=5, pady=5, row=5, column=1, sticky='e')
        attitude_label_4.grid_forget()
        attitude_input_4.grid_forget()

    def quaternion():
        parameterization_label.configure(text='Quaternion:')
        attitude_label_1.configure(text='q\u2081:')
        attitude_label_2.configure(text='q\u2082:')
        attitude_label_3.configure(text='q\u2083:')
        attitude_label_1.grid(padx=5, pady=1, row=3, column=0, sticky='w')
        attitude_input_1.grid(padx=5, pady=1, row=3, column=1, sticky='e')
        attitude_label_2.grid(padx=5, pady=1, row=4, column=0, sticky='w')
        attitude_input_2.grid(padx=5, pady=1, row=4, column=1, sticky='e')
        attitude_label_3.grid(padx=5, pady=1, row=5, column=0, sticky='w')
        attitude_input_3.grid(padx=5, pady=1, row=5, column=1, sticky='e')
        attitude_label_4.grid(padx=5, pady=1, row=6, column=0, sticky='w')
        attitude_input_4.grid(padx=5, pady=1, row=6, column=1, sticky='e')

    def changeparameterization(parameterization):
        if parameterization == 'Euler Angles':
            eulerangles()
        if parameterization == 'Quaternion':
            quaternion()

    parameterization_select = ctk.CTkOptionMenu(master=att_commands, values=('Euler Angles', 'Quaternion'), width=20, command=changeparameterization, dropdown_fg_color='#000A47', fg_color='#000A47', button_color='#000A47', button_hover_color='#000A47')
    parameterization_select.grid(row=1, column=0, columnspan=2)
    parameterization_label = ctk.CTkLabel(master=att_commands, text='Euler Angles:', font=header)
    parameterization_label.grid(padx=5, pady=5, row=2, column=0, columnspan=1, sticky='w')
    parameterization_button = ctk.CTkButton(master=att_commands, text='Enter', fg_color='#000A47', width=100)
    parameterization_button.grid(padx=5, pady=5, row=2, column=1, sticky='e')
    attitude_label_1 = ctk.CTkLabel(master=att_commands, text='Yaw: ', font=body)
    attitude_label_1.grid(padx=5, pady=5, row=3, column=0, sticky='w')
    attitude_input_1 = ctk.CTkEntry(master=att_commands, width=100)
    attitude_input_1.grid(padx=5, pady=5, row=3, column=1, sticky='e')
    attitude_label_2 = ctk.CTkLabel(master=att_commands, text='Pitch: ', font=body)
    attitude_label_2.grid(padx=5, pady=5, row=4, column=0, sticky='w')
    attitude_input_2 = ctk.CTkEntry(master=att_commands, width=100)
    attitude_input_2.grid(padx=5, pady=5, row=4, column=1, sticky='e')
    attitude_label_3 = ctk.CTkLabel(master=att_commands, text='Roll: ', font=body)
    attitude_label_3.grid(padx=5, pady=5, row=5, column=0, sticky='w')
    attitude_input_3 = ctk.CTkEntry(master=att_commands, width=100)
    attitude_input_3.grid(padx=5, pady=5, row=5, column=1, sticky='e')
    attitude_label_4 = ctk.CTkLabel(master=att_commands, text='q\u2084: ', font=body)
    attitude_input_4 = ctk.CTkEntry(master=att_commands, width=100)

    parameterization_select.set('Euler Angles')

    # box for control mode
    mode_commands = ctk.CTkFrame(master=inputs, width=math.floor(appwidth/3.2), height=math.floor(1.15*appheight/3*0.886), fg_color='darkblue')
    mode_commands.grid_propagate(0)
    mode_commands.grid(padx=5, row=0, column=1, sticky='s')
    ctk.CTkLabel(master=mode_commands, text='Set Mode', font=title).grid(row=0, column=0, padx=90)
    mode_select = tk.IntVar(value=0)
    mode_default = ctk.CTkRadioButton(master=mode_commands, variable=mode_select, value=1, text='Target Pointing', hover_color='#000A47', fg_color='white')
    mode_tracking = ctk.CTkRadioButton(master=mode_commands, variable=mode_select, value=2, text='Sun Tracking', hover_color='#000A47', fg_color='white')
    mode_filein = ctk.CTkRadioButton(master=mode_commands, variable=mode_select, value=3, text='File Input', hover_color='#000A47', fg_color='white')
    mode_default.grid(padx=15, pady=5, row=1, column=0, sticky='w')
    mode_tracking.grid(padx=15, pady=5, row=2, column=0, sticky='w')
    mode_filein.grid(padx=15, pady=5, row=3, column=0, sticky='w')

    # box for additional buttons
    button_commands = ctk.CTkFrame(master=inputs, width=math.floor(appwidth/3.2), height=math.floor(appheight*0.886), fg_color='darkblue')
    button_commands.grid_propagate(0)
    button_commands.grid(padx=5, row=0, column=2)
    button_commands.grid_columnconfigure((0, 1), weight=1)
    ctk.CTkLabel(master=button_commands, text='Other Buttons', font=title).grid(row=0, column=0, columnspan=2, padx=70)
    estop_label = ctk.CTkLabel(master=button_commands, text='Emergency Stop:    Esc + Space', font=header, text_color='red')
    estop_label.grid(row=1, column=0, columnspan=2, padx=15, pady=5, sticky='w')
    shutoff_label = ctk.CTkLabel(master=button_commands, text='Shutoff Switch: ', font=header)
    shutoff_label.grid(row=2, column=0, padx=15, pady=5, sticky='w')
    shutoff_button = ctk.CTkButton(master=button_commands, text='Stop', fg_color='#000A47', width=100)
    shutoff_button.grid(row=2, column=1, padx=15, pady=5, sticky='e')
    calibrate_label = ctk.CTkLabel(master=button_commands, text='Zero Attitude: ', font=header)
    calibrate_label.grid(row=3, column=0, padx=15, pady=5, sticky='w')
    calibrate_button = ctk.CTkButton(master=button_commands, text='Calibrate', fg_color='#000A47', width=100)
    calibrate_button.grid(row=3, column=1, padx=15, pady=5, sticky='e')

    # setup outputs tab

    # box for raw sensor output data
    sensor_outputs = ctk.CTkFrame(master=outputs, width=math.floor(appwidth/3.2), height=math.floor(appheight*0.886), fg_color='orange')
    sensor_outputs.grid_propagate(0)
    sensor_outputs.grid(padx=5, row=0, column=0)
    sensor_outputs.grid_columnconfigure((0, 1), weight=1)
    ctk.CTkLabel(master=sensor_outputs, text='Sensor Output Data', font=title, text_color='black').grid(row=0, column=0, columnspan=1, padx=40)

    gyro_output_label = ctk.CTkLabel(master=sensor_outputs, text='Gyroscope:', font=header, text_color='black')
    gyro_output_label.grid(padx=5, pady=2, row=1, column=0, sticky='w')
    gyro_label_1 = ctk.CTkLabel(master=sensor_outputs, text='X: ', font=body, text_color='black')
    gyro_label_1.grid(padx=5, pady=0, row=2, column=0, sticky='w')
    gyro_label_2 = ctk.CTkLabel(master=sensor_outputs, text='Y: ', font=body, text_color='black')
    gyro_label_2.grid(padx=5, pady=0, row=3, column=0, sticky='w')
    gyro_label_3 = ctk.CTkLabel(master=sensor_outputs, text='Z: ', font=body, text_color='black')
    gyro_label_3.grid(padx=5, pady=0, row=4, column=0, sticky='w')
    accel_output_label = ctk.CTkLabel(master=sensor_outputs, text='Accelerometer:', font=header, text_color='black')
    accel_output_label.grid(padx=5, pady=2, row=5, column=0, sticky='w')
    accel_label_1 = ctk.CTkLabel(master=sensor_outputs, text='X: ', font=body, text_color='black')
    accel_label_1.grid(padx=5, pady=0, row=6, column=0, sticky='w')
    accel_label_2 = ctk.CTkLabel(master=sensor_outputs, text='Y: ', font=body, text_color='black')
    accel_label_2.grid(padx=5, pady=0, row=7, column=0, sticky='w')
    accel_label_3 = ctk.CTkLabel(master=sensor_outputs, text='Z: ', font=body, text_color='black')
    accel_label_3.grid(padx=5, pady=0, row=8, column=0, sticky='w')
    ss_output_label = ctk.CTkLabel(master=sensor_outputs, text='Sun Direction:', font=header, text_color='black')
    ss_output_label.grid(padx=5, pady=2, row=9, column=0, sticky='w')
    ss_label_1 = ctk.CTkLabel(master=sensor_outputs, text='X: ', font=body, text_color='black')
    ss_label_1.grid(padx=5, pady=0, row=10, column=0, sticky='w')
    ss_label_2 = ctk.CTkLabel(master=sensor_outputs, text='Y: ', font=body, text_color='black')
    ss_label_2.grid(padx=5, pady=0, row=11, column=0, sticky='w')
    ss_label_3 = ctk.CTkLabel(master=sensor_outputs, text='Z: ', font=body, text_color='black')
    ss_label_3.grid(padx=5, pady=0, row=12, column=0, sticky='w')

    # box for other plots
    plot_outputs = ctk.CTkFrame(master=outputs, width=math.floor(appwidth/3.2), height=math.floor(1.8*appheight/3*0.886), fg_color='orange')
    plot_outputs.grid_propagate(0)
    plot_outputs.grid(padx=5, row=0, column=1, sticky='n')
    plot_outputs.grid_columnconfigure((0, 1), weight=1)
    ctk.CTkLabel(master=plot_outputs, text='Output Plots', font=title, text_color='black').grid(row=0, column=0, columnspan=1, padx=70)
    #image_path='Camera0.jpg'
    #image = ctk.CTkImage(
    #    light_image=Image.open(image_path),
    #    dark_image=Image.open(image_path),
    #    size=(250, 200)
    #)
    #image_label = ctk.CTkLabel(master=plot_outputs, image=image, text="")
    #image_label.grid(row=1, column=0, columnspan=2)


    # box for reaction wheel outputs
    rw_outputs = ctk.CTkFrame(master=outputs, width=math.floor(appwidth/3.2), height=math.floor(1.15*appheight/3*0.886), fg_color='orange')
    rw_outputs.grid_propagate(0)
    rw_outputs.grid(padx=5, row=0, column=1, sticky='s')
    rw_outputs.grid_columnconfigure((0, 1), weight=1)
    ctk.CTkLabel(master=rw_outputs, text='Reaction Wheel Outputs', font=title, text_color='black').grid(row=0, column=0, pady=5, columnspan=1, padx=25)

    #def motoroutputs():
    motor_label_1 = ctk.CTkLabel(master=rw_outputs, text=f'X: {rpm1} RPM', font=body, text_color='black')
    motor_label_1.grid(padx=5, pady=1, row=1, column=0, sticky='w')
    motor_label_2 = ctk.CTkLabel(master=rw_outputs, text=f'Y: {rpm2} RPM', font=body, text_color='black')
    motor_label_2.grid(padx=5, pady=1, row=2, column=0, sticky='w')
    motor_label_3 = ctk.CTkLabel(master=rw_outputs, text=f'Z: {rpm3} RPM', font=body, text_color='black')
    motor_label_3.grid(padx=5, pady=1, row=3, column=0, sticky='w')
    #motoroutputs()

    # box for attitude outputs
    attitude_outputs = ctk.CTkFrame(master=outputs, width=math.floor(appwidth/3.2), height=math.floor(appheight*0.886), fg_color='orange')
    attitude_outputs.grid_propagate(0)
    attitude_outputs.grid(padx=5, row=0, column=2)
    attitude_outputs.grid_columnconfigure((0, 1), weight=1)
    ctk.CTkLabel(master=attitude_outputs, text='Attitude Output', font=title, text_color='black').grid(row=0, column=0, columnspan=3, padx=10)

    def eaoutput():
        output_label.configure(text='Euler Angles:')
        attitude_outlabel_1.configure(text='Yaw:')
        attitude_outlabel_2.configure(text='Pitch:')
        attitude_outlabel_3.configure(text='Roll:')
        attitude_outlabel_1.grid(padx=5, pady=5, row=3, column=0, sticky='w')
        attitude_output_1.grid(padx=5, pady=5, row=3, column=1, sticky='e')
        attitude_outlabel_2.grid(padx=5, pady=5, row=4, column=0, sticky='w')
        attitude_output_2.grid(padx=5, pady=5, row=4, column=1, sticky='e')
        attitude_outlabel_3.grid(padx=5, pady=5, row=5, column=0, sticky='w')
        attitude_output_3.grid(padx=5, pady=5, row=5, column=1, sticky='e')
        attitude_outlabel_4.grid_forget()
        attitude_output_4.grid_forget()

    def qoutput():
        output_label.configure(text='Quaternion:')
        attitude_outlabel_1.configure(text='q\u2081:')
        attitude_outlabel_2.configure(text='q\u2082:')
        attitude_outlabel_3.configure(text='q\u2083:')
        attitude_outlabel_1.grid(padx=5, pady=1, row=3, column=0, sticky='w')
        attitude_output_1.grid(padx=5, pady=1, row=3, column=1, sticky='e')
        attitude_outlabel_2.grid(padx=5, pady=1, row=4, column=0, sticky='w')
        attitude_output_2.grid(padx=5, pady=1, row=4, column=1, sticky='e')
        attitude_outlabel_3.grid(padx=5, pady=1, row=5, column=0, sticky='w')
        attitude_output_3.grid(padx=5, pady=1, row=5, column=1, sticky='e')
        attitude_outlabel_4.grid(padx=5, pady=1, row=6, column=0, sticky='w')
        attitude_output_4.grid(padx=5, pady=1, row=6, column=1, sticky='e')

    def changeparameteroutput(parameterization):
        if parameterization == 'Euler Angles':
            eaoutput()
        if parameterization == 'Quaternion':
            qoutput()

    output_select = ctk.CTkOptionMenu(master=attitude_outputs, values=('Euler Angles', 'Quaternion'), width=20, command=changeparameteroutput, dropdown_fg_color='#000A47', fg_color='#000A47', button_color='#000A47', button_hover_color='#000A47')
    output_select.grid(row=1, column=0, columnspan=2)
    output_label = ctk.CTkLabel(master=attitude_outputs, text='Euler Angles:', font=header, text_color='black')
    output_label.grid(padx=5, pady=5, row=2, column=0, columnspan=1, sticky='w')
    attitude_outlabel_1 = ctk.CTkLabel(master=attitude_outputs, text='Yaw: ', font=body, text_color='black')
    attitude_outlabel_1.grid(padx=5, pady=5, row=3, column=0, sticky='w')
    attitude_output_1 = ctk.CTkLabel(master=attitude_outputs, text='0', width=100, text_color='black')
    attitude_output_1.grid(padx=5, pady=5, row=3, column=1, sticky='e')
    attitude_outlabel_2 = ctk.CTkLabel(master=attitude_outputs, text='Pitch: ', font=body, text_color='black')
    attitude_outlabel_2.grid(padx=5, pady=5, row=4, column=0, sticky='w')
    attitude_output_2 = ctk.CTkLabel(master=attitude_outputs, text='0', width=100, text_color='black')
    attitude_output_2.grid(padx=5, pady=5, row=4, column=1, sticky='e')
    attitude_outlabel_3 = ctk.CTkLabel(master=attitude_outputs, text='Roll: ', font=body, text_color='black')
    attitude_outlabel_3.grid(padx=5, pady=5, row=5, column=0, sticky='w')
    attitude_output_3 = ctk.CTkLabel(master=attitude_outputs, text='0', width=100, text_color='black')
    attitude_output_3.grid(padx=5, pady=5, row=5, column=1, sticky='e')
    attitude_outlabel_4 = ctk.CTkLabel(master=attitude_outputs, text='q\u2084: ', font=body, text_color='black')
    attitude_output_4 = ctk.CTkLabel(master=attitude_outputs, text='1', width=100, text_color='black')

    output_select.set('Euler Angles')

    # currentcam = 0
    # image_paths = ["Camera0.jpg", "Camera1.jpg", "Camera2.jpg", "Camera3.jpg"]
    # def updateimage(cam):
    #     takepicture(currentcam)
    #     image = ctk.CTkImage(
    #         light_image=Image.open(image_paths[cam]),
    #         dark_image=Image.open(image_paths[cam]),
    #         size=(250, 200) # (width, height)
    #     )
    #     image_label = ctk.CTkLabel(master=plot_outputs, image=image, text="")
    #     image_label.grid(row=1, column=0, columnspan=2)
    # updateimage(0)

    # def changecamera(cam):
    #     if cam == "Camera A":
    #         currentcam = 0
    #     if cam == "Camera B":
    #         currentcam = 1
    #     if cam == "Camera C":
    #         currentcam = 2
    #     if cam == "Camera D":
    #         currentcam = 3
    #     updateimage(currentcam)
    #     show_current_cam = ctk.CTkLabel(master=attitude_outputs, text=str(currentcam), font=body)
    #     show_current_cam.grid(row=9, column=0, columnspan=2)
    # camera_select = ctk.CTkOptionMenu(master=attitude_outputs, values=("Camera A", "Camera B", "Camera C", "Camera D"), command=changecamera)
    # camera_select.grid(row=8, column=0, columnspan=2)
    def updatedata():
        global ax
        global ay
        global az
        global gx
        global gy
        global gz
        global rpm1
        global rpm2
        global rpm3
        while True:
            accel_label_1.configure(text=f"X: {ax:.2f} m/s^2")
            accel_label_2.configure(text=f"Y: {ay:.2f} m/s^2")
            accel_label_3.configure(text=f"Z: {az:.2f} m/s^2")
            gyro_label_1.configure(text=f"X: {gx:.2f}")
            gyro_label_2.configure(text=f"Y: {gy:.2f}")
            gyro_label_3.configure(text=f"Z: {gz:.2f}")
            motor_label_1.configure(text=f"X: {rpm1:.0f} RPM")
            motor_label_2.configure(text=f"Y: {rpm2:.0f} RPM")
            motor_label_3.configure(text=f"Z: {rpm3:.0f} RPM")

    t3 = Thread(target=updatedata, args=())
    t3.start()

    app.mainloop()


t1 = Thread(target=getdata, args=())
t2 = Thread(target=appwindow, args=())
t1.start()
t2.start()
