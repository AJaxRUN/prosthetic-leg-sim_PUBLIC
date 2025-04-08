###################### Template Code, yet to complete #################### 

# import time
# import mujoco
# import mujoco.viewer

# m = mujoco.MjModel.from_xml_path('/Users/arjunchembairamachandran/Desktop/ROB 590/myo_sim/osl/myolegs_osl.xml')
# d = mujoco.MjData(m)

# with mujoco.viewer.launch_passive(m, d) as viewer:
#   start = time.time()
#   while viewer.is_running() and time.time() - start < 30:
#     step_start = time.time()

#     mujoco.mj_step(m, d)

#     with viewer.lock():
#       viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(d.time % 2)

#     viewer.sync()

#     time_until_next_step = m.opt.timestep - (time.time() - step_start)
#     if time_until_next_step > 0:
#       time.sleep(time_until_next_step)