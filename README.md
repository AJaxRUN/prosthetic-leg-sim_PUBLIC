# Amputee-Prosthetic Leg Simulation Environment

## Prerequisites 
You can use Conda or Pyenv, note that since requirements.txt is generated for pip (Pyenv),
the procedure for pyenv is given below.
1. Ensure pyenv and pyenv-virtualenv are installed on your system. 
      
   **MacOS**
   ```
      # Install pyenv
      brew install pyenv

      # Install pyenv-virtualenv (optional but recommended)
      brew install pyenv-virtualenv

      # Add pyenv to your shell
      echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc
      echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc
      echo 'eval "$(pyenv init --path)"' >> ~/.zshrc
      echo 'eval "$(pyenv init -)"' >> ~/.zshrc
      echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.zshrc

      # Reload shell configuration
      source ~/.zshrc
   ```

   **Linux** - Install pyenv on Linux using git
   ```
      # Install dependencies
      sudo apt update && sudo apt install -y \
      make build-essential libssl-dev zlib1g-dev \
      libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
      libncurses5-dev libncursesw5-dev xz-utils tk-dev \
      libffi-dev liblzma-dev git

      # Clone pyenv repository
      git clone https://github.com/pyenv/pyenv.git ~/.pyenv

      # Clone pyenv-virtualenv (optional)
      git clone https://github.com/pyenv/pyenv-virtualenv.git ~/.pyenv plugins/pyenv-virtualenv

      # Add pyenv to your shell
      echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
      echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
      echo 'eval "$(pyenv init --path)"' >> ~/.bashrc
      echo 'eval "$(pyenv init -)"' >> ~/.bashrc
      echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bashrc

      # Reload shell configuration
      source ~/.bashrc
   ```

   **Windows** On Windows, pyenv is not natively supported. Instead, use pyenv-win
   ```
   # Install pyenv-win
   git clone https://github.com/pyenv-win/pyenv-win.git $HOME\.pyenv

   # Add pyenv to your PATH
   $env:PYENV = "$HOME\.pyenv"
   [System.Environment]::SetEnvironmentVariable("PYENV", $env:PYENV, "User")
   [System.Environment]::SetEnvironmentVariable("Path", "$env:PYENV\bin;$env:PYENV\shims;$env:Path", "User")

   # Restart your terminal to apply changes
   ```

2. Create a virtual environment named prosthetic-leg-sim
   ```
   pyenv install 3.10.6  # Skip if already installed
   pyenv virtualenv 3.10.6 prosthetic-leg-sim
   ```
3. Activate the Virtual Environment
   ```
   pyenv activate prosthetic-leg-sim
   ```
4. Install Project Dependencies
   ```
   pip install -r requirements.txt
   ```

## Running the simulation
You can refer and modify config.yaml for the possible set of args - for running the simulation in different modes:

```
-env 
   type: String
   default: gym_hopper
   options: gym_hopper, gym_prosthetic, gym_hopper
   help: Specify the environment type.

-mode 
   type: String
   default: train
   options: train, test
   help: Specify the mode.
   note: Running this will replace the trained agent_models.

-agent_model 
   type: String
   help: Specify the relative path of the saved agent (.zip file). You can specify this to load a specific learnt model/policy for testing.

-clear_logs 
   type: Boolean
   help: Passing this flag will clear existing Tensorborad logs.
```

To run the simulation - MacOS, Linux

`mjpython main.py`

To run the simulation - Windows

`python main.py`
or
`python3 main.py`

For viewing the Tensorboard logs, open a separate terminal in the project root directory and run the following command:
`tensorboard --logdir=logs/`
## Motivation

The development of advanced prosthetic leg controllers is crucial for improving the quality of life for amputees. However, the process of designing, testing, and personalizing these controllers presents several challenges:

1. Safety Concerns: Real-world testing with amputee users can pose risks, especially during the early stages of controller development.

2. Time and Resource Constraints: Extensive physical testing is time-consuming and requires significant resources, including participant recruitment and specialized equipment.

3. Personalization Challenges: Each amputee has unique gait patterns and physical characteristics, making it difficult to create a one-size-fits-all solution.

4. Limited Iteration Capability: Physical testing limits the number of iterations possible for controller optimization.

5. Ethical Considerations: Minimizing discomfort and inconvenience for amputee participants during the development process is essential.

A comprehensive simulation environment addresses these challenges by:

- Providing a safe, virtual space for rapid prototyping and testing of prosthetic controllers.
- Enabling numerous iterations and scenarios to be tested quickly and cost-effectively.
- Allowing for the exploration of personalization strategies without requiring constant user involvement.
- Facilitating the development of more robust controllers before moving to physical testing.

By creating this simulation environment, we aim to accelerate the development of improved prosthetic leg controllers, ultimately leading to better outcomes for amputee users.

## Objective

Create a comprehensive simulation environment for an amputee-prosthetic leg system to enable future personalization of prosthetic controllers.

## Project Phases

### Phase 1: Basic Prosthetic Leg Model for Level Walking

1. Develop a dynamic model of the prosthetic leg, including knee and ankle joints.

2. Implement joint control:
   - Initially use torques from the dataset for knee and ankle control
   - (HKIC implementation will be deferred to a later stage)

3. Model amputee-prosthetic interaction through socket forces (Fx, Fy, and Moment).

4. Develop an RL-based "human controller" to simulate interaction forces.

5. Train the human controller using existing amputee walking data to mimic:
   - Normalized thigh motion (angle, velocity, acceleration)
   - Normalized ground reaction forces (GRF) at the prosthetic foot

6. Extend the model to incorporate stride-to-stride variations:
   - Begin with replicating normalized thigh data and GRF
   - Gradually introduce controlled variations to simulate natural gait differences

7. Validate the model's accuracy in reproducing both normalized data and stride variations

### Phase 2: Expanded Task Scenarios

1. Extend the simulation to include ramp walking scenarios.
2. Incorporate stair walking scenarios.
3. Adapt the human controller to handle these additional tasks.

### Phase 3: Full Leg Model Integration

1. Expand the model to include the sound leg.
2. Extend the human controller to manage both prosthetic and sound leg control.
3. Implement stride-to-stride variations to reflect real-world data:
   - Initially focus on normalized data for human controller learning
   - Introduce controlled disturbances to simulate natural gait variations

## Key Features

1. Data-driven approach using existing amputee walking data
2. Initial focus on reproducing dataset torques for knee and ankle control
3. Capability to model both normalized gait patterns and stride-to-stride variations
4. Flexibility to simulate various walking scenarios (level, ramp, stairs) in later phases

## Future Work: Personalization of Prosthetic Leg Controller

### Objective

Utilize the developed simulation environment to train and optimize a personalized HKIC for individual amputee users.

### Key Goals

1. Employ RL techniques to adapt HKIC parameters for individual users.
2. Improve symmetry index between prosthetic and sound leg joint angles.
3. Develop sim-to-real transfer methods for applying personalized controllers to actual prosthetic legs.

## Conclusion

This project aims to develop a robust simulation environment for amputee-prosthetic leg systems, paving the way for future advancements in personalized prosthetic control. By addressing the challenges of real-world testing and enabling rapid iteration, this simulation framework will accelerate the development of improved prosthetic technologies, ultimately enhancing the quality of life for amputee users.
